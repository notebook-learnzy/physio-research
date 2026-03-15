"""
research.py — Main autonomous research orchestrator for Learnzy Focus Score validation
Mirrors the karpathy/autoresearch loop for scientific hypothesis validation.

Autoresearch pattern:
  1. Try an experiment (search query → extract papers)
  2. If evidence_score IMPROVED → KEEP (merge findings)
  3. If evidence_score NOT improved → DISCARD (throw away findings)
  4. Only the best-quality evidence accumulates

Focus Score = 0.4 × HRV_Readiness + 0.6 × Sleep_Recovery (composite, not separate)

Fixed 5-minute budget per run. Crawls papers, extracts findings, scores hypothesis.
Logs results to results.tsv. Runs on GitHub Actions.
"""

import os
import sys
import time
import json
import random
import shutil
import logging
import argparse
import tempfile
import subprocess
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from search import search_all, rank_papers, Paper
from extract import extract_batch
from hypothesis import compute_evidence_score, print_summary, load_findings

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TIME_BUDGET_SECONDS = 5 * 60          # 5-minute wall clock budget
RESULTS_TSV         = Path("results.tsv")
FINDINGS_JSONL      = Path("findings.jsonl")
CITATIONS_MD        = Path("citations.md")
PROGRAM_MD          = Path("program.md")

# Max papers to fetch per source per run, then rank → top 10 go to LLM
MAX_PER_SOURCE = 25
TOP_N_EXTRACT  = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("research")


# ─── KEYWORD STRATEGY (Focus Score specific) ──────────────────────────────────
KEYWORD_SETS = [
    # Core: composite HRV+sleep → mental health
    "composite HRV sleep readiness metric mental health students depression anxiety",
    "RMSSD nighttime HRV wearable depression prediction PHQ-9 students",
    "sleep quality composite score PSQI academic performance GPA university",
    # GR receptor / cortisol boundary
    "glucocorticoid receptor activation threshold cortisol cognitive impairment stress",
    # Bedtime consistency → cognition
    "bedtime consistency circadian regularity learning retention recall students",
    # Composite readiness metrics (competitors)
    "HRV sleep composite readiness metric wearable wellbeing recovery score",
    # Two-pattern stress detection
    "acute chronic stress detection physiological monitoring wearable anomaly",
    # AASM sleep efficiency
    "sleep efficiency AASM 85% threshold cognitive performance next-day function",
    # Wearable HRV clinical validity
    "wearable HRV mental health screening clinical sensitivity specificity smartwatch",
    # Sleep → memory → academic
    "sleep duration optimal 7-9 hours memory consolidation learning exam performance",
    # Physiological readiness intervention
    "physiological readiness wearable intervention student wellbeing mental health",
    # HRV biofeedback → academic
    "autonomic recovery HRV biofeedback stress reduction academic performance",
    # Slow-wave sleep → memory
    "slow wave sleep RMSSD memory consolidation exam performance retention",
    # Digital biomarker early warning
    "digital biomarker composite mental health detection lead time early warning",
    # WHOOP / Oura validation
    "WHOOP recovery score Oura readiness clinical validation mental health outcomes",
    # Effect size benchmarks
    "HRV Cohen d effect size mental health biomarker student population",
    # ISI + wearable
    "insomnia severity index ISI wearable sleep monitoring prediction students",
    # HRV × anxiety correlation
    "heart rate variability anxiety GAD-7 correlation students longitudinal cohort",
    # Sleep recovery + cognitive load
    "sleep recovery score composite metric student cognitive load optimization",
    # Early warning systems
    "early warning mental health deterioration physiological wearable university 19 days",
    # Personal baseline vs population
    "HRV personal baseline intra-individual vs population norm stress detection",
    # RMSSD threshold
    "RMSSD threshold 50ms clinical significance autonomic function parasympathetic",
    # Sleep latency + WASO
    "sleep latency WASO wake after sleep onset cognitive impairment next-day students",
    # Validated composite metrics
    "composite readiness metric validated clinical outcomes effect size Cohen d",
    # Lead time prediction
    "physiological stress biomarker prediction mental health lead time early detection",
]


def get_next_query(iteration: int, previous_findings: list[dict]) -> str:
    """Rotate through keyword sets, mutate based on findings every 5 iterations."""
    base = KEYWORD_SETS[iteration % len(KEYWORD_SETS)]

    # Every 5 iterations, synthesize a focused query from top markers
    if iteration > 0 and iteration % 5 == 0 and previous_findings:
        from collections import Counter
        markers = []
        for f in previous_findings:
            for m in f.get("markers", []):
                if m.get("measure"):
                    markers.append(m["measure"])
        if markers:
            top_marker = Counter(markers).most_common(1)[0][0]
            focused_queries = [
                f"{top_marker} threshold mental health academic performance Focus Score",
                f"{top_marker} intervention randomized controlled trial students composite",
                f"{top_marker} predictive validity mental health biomarker wearable",
                f"{top_marker} cognitive load optimization student readiness composite HRV sleep",
            ]
            base = random.choice(focused_queries)
            log.info(f"[Strategy] Using marker-focused query: {base!r}")

    return base


def load_existing_paper_ids() -> set:
    existing = set()
    if FINDINGS_JSONL.exists():
        with open(FINDINGS_JSONL) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if d.get("paper_id"):
                            existing.add(d["paper_id"])
                    except Exception:
                        pass
    return existing


def write_findings_to_file(findings: list, path: Path) -> int:
    """Write findings to a JSONL file (append mode)."""
    if not findings:
        return 0
    with open(path, "a") as f:
        for finding in findings:
            f.write(json.dumps(asdict(finding)) + "\n")
    return len(findings)


def merge_findings(temp_path: Path) -> int:
    """Merge temp findings into the main findings.jsonl and update citations."""
    if not temp_path.exists():
        return 0
    new_findings = []
    with open(temp_path) as f:
        for line in f:
            if line.strip():
                new_findings.append(json.loads(line))
    if not new_findings:
        return 0
    # Append to main findings
    with open(FINDINGS_JSONL, "a") as f:
        for finding in new_findings:
            f.write(json.dumps(finding) + "\n")
    # Update citations
    update_citations(new_findings)
    return len(new_findings)


def compute_score_with_temp(temp_path: Path) -> float:
    """Compute evidence_score as if temp findings were merged into main findings."""
    # Create a combined temp file
    combined = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    try:
        # Copy existing findings
        if FINDINGS_JSONL.exists():
            with open(FINDINGS_JSONL) as f:
                combined.write(f.read())
        # Append temp findings
        if temp_path.exists():
            with open(temp_path) as f:
                combined.write(f.read())
        combined.close()
        report = compute_evidence_score(Path(combined.name))
        return report.evidence_score
    finally:
        os.unlink(combined.name)


def update_citations(findings: list):
    """Append newly extracted papers to citations.md with links."""
    if not findings:
        return
    header_needed = not CITATIONS_MD.exists()
    with open(CITATIONS_MD, "a") as f:
        if header_needed:
            f.write("# Citations — Papers Used in Focus Score Validation\n\n")
            f.write("Auto-generated by physio-research. Papers extracted and scored by Claude Haiku.\n\n")
            f.write("| # | Year | Title | Source | DOI / URL | H1 | H2 | Key Finding |\n")
            f.write("|---|------|-------|--------|-----------|----|----|-------------|\n")
        # Count existing lines to get next row number
        existing_count = 0
        if not header_needed:
            with open(CITATIONS_MD, "r") as rf:
                existing_count = sum(1 for line in rf if line.startswith("|") and not line.startswith("| #") and not line.startswith("|---"))
        for i, finding in enumerate(findings):
            fd = asdict(finding) if hasattr(finding, '__dataclass_fields__') else finding
            if not fd.get("extraction_ok", True):
                continue
            num = existing_count + i + 1
            year = fd.get("year", "?")
            title = fd.get("title", "?")[:80].replace("|", "/")
            source = fd.get("source", "?")
            doi = fd.get("doi", "")
            url = fd.get("url", "")
            link = f"[doi](https://doi.org/{doi})" if doi else (f"[link]({url})" if url else "—")
            h1 = "✓" if fd.get("supports_h1") else ("✗" if fd.get("supports_h1") is False else "—")
            h2 = "✓" if fd.get("supports_h2") else ("✗" if fd.get("supports_h2") is False else "—")
            kf = (fd.get("key_finding") or "")[:100].replace("|", "/").replace("\n", " ")
            f.write(f"| {num} | {year} | {title} | {source} | {link} | {h1} | {h2} | {kf} |\n")


def init_results_tsv():
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tevidence_score\tpapers_total\tstatus\tdescription\n")
        log.info("[Init] Created results.tsv")


def log_result(commit: str, score: float, papers_total: int, status: str, description: str):
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{score:.6f}\t{papers_total}\t{status}\t{description}\n")


def git_commit(message: str) -> str:
    try:
        subprocess.run(["git", "add", "findings.jsonl", "results.tsv", "citations.md"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log.warning(f"[Git] Commit failed: {e.stderr.decode() if e.stderr else e}")
        return "0000000"


def git_current_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "0000000"


def run_experiment(iteration: int, dry_run: bool = False) -> dict:
    """Run one experiment following the autoresearch keep/discard pattern.

    1. Search for papers with current query
    2. Extract findings to a TEMP file (not directly to findings.jsonl)
    3. Compute score WITH temp findings merged
    4. If score improved → KEEP (merge temp → findings.jsonl)
    5. If score not improved → DISCARD (delete temp file)
    """
    t_start = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"[Research] Iteration {iteration+1} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"[Research] Focus Score validation (0.4×HRV + 0.6×Sleep composite)")

    existing_ids = load_existing_paper_ids()
    prev_findings = load_findings(FINDINGS_JSONL)
    prev_score = compute_evidence_score(FINDINGS_JSONL).evidence_score if prev_findings else 0.0
    log.info(f"[Research] Current evidence_score: {prev_score:.6f} (from {len(prev_findings)} papers)")

    query = get_next_query(iteration, prev_findings)
    log.info(f"[Query] {query!r}")

    # ── Step 1: Search ──
    log.info("[Search] Crawling papers from all sources...")
    papers = search_all(query=query, max_per_source=MAX_PER_SOURCE, existing_ids=existing_ids)
    papers_found = len(papers)
    log.info(f"[Search] Found {papers_found} new papers")

    # Pre-score and rank — only top N go to LLM extraction
    if papers:
        ranked = rank_papers(papers, top_n=TOP_N_EXTRACT)
        log.info(f"[Ranking] Selected top {len(ranked)} of {papers_found} for extraction")
    else:
        ranked = []

    # ── Step 2: Extract to TEMP file ──
    temp_path = Path(tempfile.mktemp(suffix=".jsonl", prefix="findings_temp_"))
    papers_extracted = 0
    new_findings_count = 0
    all_failed = False

    if dry_run:
        log.info("[DryRun] Skipping LLM extraction")
        log.info(f"[DryRun] Would extract from {len(ranked)} top-ranked papers")
    elif ranked:
        paper_dicts = [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in ranked]
        findings = extract_batch(paper_dicts, existing_ids)
        papers_extracted = len(findings)

        # Check if ALL extractions failed (API error)
        ok_findings = [f for f in findings if f.extraction_ok]
        if papers_extracted > 0 and len(ok_findings) == 0:
            all_failed = True
            log.error(f"[Extract] ALL {papers_extracted} extractions failed! Likely an API error.")
        else:
            # Write successful findings to temp file
            new_findings_count = write_findings_to_file(ok_findings, temp_path)
            log.info(f"[Extract] {new_findings_count} successful extractions written to temp")

    # ── Step 3: Compute score with temp merged ──
    if new_findings_count > 0:
        new_score = compute_score_with_temp(temp_path)
    else:
        new_score = prev_score
    delta = new_score - prev_score

    # ── Step 4: Keep or Discard (autoresearch pattern) ──
    if all_failed:
        status = "api_error"
        log.error(f"[Decision] ❌ API_ERROR — all extractions failed, nothing to keep")
    elif new_findings_count == 0 and papers_found == 0:
        status = "no_papers"
        log.info(f"[Decision] — NO_PAPERS — no new papers found")
    elif new_findings_count == 0:
        status = "no_findings"
        log.info(f"[Decision] — NO_FINDINGS — extraction yielded nothing")
    elif delta > 0:
        # KEEP — merge temp findings into main file
        status = "keep"
        merged = merge_findings(temp_path)
        log.info(f"[Decision] ✅ KEEP — evidence_score {prev_score:.6f} → {new_score:.6f} (Δ={delta:+.6f}), merged {merged} findings")
    else:
        # DISCARD — throw away temp findings (autoresearch: git reset)
        status = "discard"
        log.info(f"[Decision] ❌ DISCARD — evidence_score {prev_score:.6f} → {new_score:.6f} (Δ={delta:+.6f}), no improvement")

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # ── Step 5: Report ──
    report = compute_evidence_score(FINDINGS_JSONL)
    desc = f"[iter{iteration+1}] {query[:50]} | +{new_findings_count} findings | Δ={delta:+.4f} | {status}"
    run_seconds = time.time() - t_start

    log.info(f"\n[Results] Iteration {iteration+1} complete in {run_seconds:.1f}s")
    print_summary(report)

    return {
        "iteration": iteration + 1,
        "query": query,
        "status": status,
        "papers_found": papers_found,
        "papers_extracted": papers_extracted,
        "new_findings": new_findings_count,
        "evidence_score": report.evidence_score,
        "delta": delta,
        "run_seconds": run_seconds,
        "description": desc,
        "all_failed": all_failed,
    }


def run_loop(
    max_wall_seconds: int = TIME_BUDGET_SECONDS,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
):
    t_wall_start = time.time()
    init_results_tsv()
    consecutive_api_fails = 0
    MAX_CONSECUTIVE_API_FAILS = 3  # stop loop after 3 consecutive all-fail iterations

    log.info(f"[Research] Starting Learnzy Focus Score validation loop")
    log.info(f"[Research] Pattern: autoresearch keep/discard — only improvements are kept")
    log.info(f"[Research] Hypothesis: Focus Score (0.4×HRV + 0.6×Sleep) → mental health + cognition")
    log.info(f"[Research] Time budget: {max_wall_seconds}s | Dry run: {dry_run}")
    log.info(f"[Research] Validated pilot: Cohen's d=1.536, PHQ-9 r=−0.452, ISI r=−0.591")

    iteration = 0

    while True:
        elapsed = time.time() - t_wall_start
        if elapsed >= max_wall_seconds:
            log.info(f"[Research] Time budget of {max_wall_seconds}s reached after {iteration} iterations.")
            break
        if max_iterations is not None and iteration >= max_iterations:
            log.info(f"[Research] Max iterations ({max_iterations}) reached.")
            break

        try:
            metrics = run_experiment(iteration, dry_run=dry_run)

            # Early-stop: if API is broken, don't burn through all iterations
            if metrics.get("all_failed"):
                consecutive_api_fails += 1
                if consecutive_api_fails >= MAX_CONSECUTIVE_API_FAILS:
                    log.error(f"[Research] {MAX_CONSECUTIVE_API_FAILS} consecutive API failures. Stopping loop.")
                    log.error(f"[Research] Check your ANTHROPIC_API_KEY and credit balance.")
                    break
            else:
                consecutive_api_fails = 0

            if not dry_run:
                # Only git commit on KEEP (like autoresearch advances the branch)
                if metrics["status"] == "keep":
                    commit = git_commit(metrics["description"])
                else:
                    commit = git_current_commit()
                log_result(
                    commit=commit,
                    score=metrics["evidence_score"],
                    papers_total=len(load_existing_paper_ids()),
                    status=metrics["status"],
                    description=metrics["description"],
                )

            iteration += 1
            log.info(f"[Research] Iteration {iteration} done. Status: {metrics['status']} | Evidence: {metrics['evidence_score']:.6f}")

        except KeyboardInterrupt:
            log.info("[Research] Interrupted by user.")
            break
        except Exception as e:
            log.error(f"[Research] Iteration {iteration} FAILED: {e}")
            import traceback; traceback.print_exc()
            log_result("0000000", 0.0, 0, "crash", f"[iter{iteration+1}] ERROR: {str(e)[:80]}")
            iteration += 1
            time.sleep(5)

    final_report = compute_evidence_score(FINDINGS_JSONL)
    log.info(f"\n{'='*60}")
    log.info(f"[Research] FINAL SUMMARY after {iteration} iterations:")
    print_summary(final_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learnzy Focus Score — Autonomous Research Agent")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM extraction (test crawling only)")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations (default: unlimited)")
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET_SECONDS, help="Wall clock budget in seconds")
    args = parser.parse_args()

    run_loop(
        max_wall_seconds=args.time_budget,
        dry_run=args.dry_run,
        max_iterations=args.iterations,
    )
