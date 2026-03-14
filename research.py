"""
research.py — Main autonomous research orchestrator for Learnzy Focus Score validation
Mirrors the karpathy/autoresearch loop but for scientific hypothesis validation.

Focus Score = 0.4 × HRV_Readiness + 0.6 × Sleep_Recovery (composite, not separate)

Fixed 5-minute budget per run. Crawls papers, extracts findings, scores hypothesis.
Logs results to results.tsv. Runs indefinitely on GitHub Actions.

Budget: 2000 tokens/minute via Claude Haiku → ~864 runs over 3 days.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from search import search_all, Paper
from extract import extract_batch
from hypothesis import compute_evidence_score, print_summary, load_findings

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TIME_BUDGET_SECONDS = 5 * 60          # 5-minute wall clock budget
RESULTS_TSV         = Path("results.tsv")
FINDINGS_JSONL      = Path("findings.jsonl")
PROGRAM_MD          = Path("program.md")

# Max papers to fetch per source per run
MAX_PER_SOURCE = 25

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


def append_findings(findings: list) -> int:
    if not findings:
        return 0
    with open(FINDINGS_JSONL, "a") as f:
        for finding in findings:
            f.write(json.dumps(asdict(finding)) + "\n")
    return len(findings)


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
        subprocess.run(["git", "add", "findings.jsonl", "results.tsv"], check=True, capture_output=True)
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
    t_start = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"[Research] Iteration {iteration+1} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"[Research] Focus Score validation (0.4×HRV + 0.6×Sleep composite)")

    existing_ids = load_existing_paper_ids()
    prev_findings = load_findings(FINDINGS_JSONL)
    prev_score = compute_evidence_score(FINDINGS_JSONL).evidence_score if prev_findings else 0.0

    query = get_next_query(iteration, prev_findings)
    log.info(f"[Query] {query!r}")

    log.info("[Search] Crawling papers from all sources...")
    papers = search_all(query=query, max_per_source=MAX_PER_SOURCE, existing_ids=existing_ids)
    papers_found = len(papers)
    log.info(f"[Search] Found {papers_found} new papers")

    if dry_run:
        log.info("[DryRun] Skipping LLM extraction")
        papers_extracted = 0
        new_findings_count = 0
    else:
        if papers:
            paper_dicts = [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in papers]
            findings = extract_batch(paper_dicts, existing_ids)
            papers_extracted = len(findings)
            new_findings_count = append_findings(findings)
        else:
            papers_extracted = 0
            new_findings_count = 0

    report = compute_evidence_score(FINDINGS_JSONL)
    delta = report.evidence_score - prev_score

    if new_findings_count == 0 and papers_found == 0:
        status = "no_papers"
    elif delta >= 0.005:
        status = "keep"
    elif delta >= 0:
        status = "marginal"
    else:
        status = "no_gain"

    desc = f"[iter{iteration+1}] {query[:60]} | +{new_findings_count} findings | delta={delta:+.4f}"
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
    }


def run_loop(
    max_wall_seconds: int = TIME_BUDGET_SECONDS,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
):
    t_wall_start = time.time()
    init_results_tsv()

    log.info(f"[Research] Starting Learnzy Focus Score validation loop")
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

            if not dry_run:
                commit = git_commit(metrics["description"])
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
