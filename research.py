"""
research.py — Main autonomous research orchestrator
Mirrors the karpathy/autoresearch loop but for scientific hypothesis validation.

Fixed 5-minute budget per run. Crawls papers, extracts findings, scores hypothesis.
Logs results to results.tsv. Runs indefinitely (or until manually stopped) on GitHub Actions.

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

from search import search_all, Paper
from extract import extract_batch
from hypothesis import compute_evidence_score, print_summary, load_findings

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TIME_BUDGET_SECONDS = 5 * 60          # 5-minute wall clock budget (excl. startup)
RESULTS_TSV         = Path("results.tsv")
FINDINGS_JSONL      = Path("findings.jsonl")
PROGRAM_MD          = Path("program.md")
LOG_FILE            = Path("run.log")

# Max papers to fetch per source per run (keep runs within 5 min)
MAX_PER_SOURCE = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("research")


# ─── KEYWORD STRATEGY ─────────────────────────────────────────────────────────
KEYWORD_SETS = [
    "HRV heart rate variability mental health students academic performance",
    "RMSSD SDNN anxiety depression university students",
    "sleep quality PSQI cognition GPA marks examination",
    "autonomic nervous system stress cortisol student performance",
    "HRV biofeedback intervention cognition randomized controlled trial",
    "wearable HRV monitoring mental health outcomes prospective study",
    "slow wave sleep memory consolidation learning students RCT",
    "sleep deprivation attention working memory executive function university",
    "heart rate variability anxiety prediction clinical threshold RMSSD",
    "polysomnography sleep stages academic success undergraduate",
    "HRV stress resilience mindfulness students intervention",
    "autonomic dysregulation depression HRV cohort students longitudinal",
    "sleep duration mental wellbeing GPA university systematic review",
    "vagal tone cognition attention concentration students HRV",
    "HRV SDNN > 100ms cardiovascular health students stress",
    "Pittsburgh Sleep Quality Index PSQI academic performance correlation",
    "HRV biofeedback anxiety reduction students randomized",
    "circadian rhythm sleep quality exam performance medical students",
    "HRV low frequency high frequency ratio stress students",
    "objective sleep monitoring actigraphy university student performance",
    "HRV mental health prediction machine learning wearable",
    "sleep restriction cognitive impairment medical students outcome",
    "parasympathetic activity resting HRV academic stress burnout",
    "sleep intervention academic performance randomized controlled trial",
    "heart rate variability resilience stress burnout student longitudinal",
]


def get_next_query(iteration: int, previous_findings: list[dict]) -> str:
    """Rotate through keyword sets, slightly mutating based on findings."""
    base = KEYWORD_SETS[iteration % len(KEYWORD_SETS)]

    # Every 5 iterations, try to synthesize a query from top markers found
    if iteration > 0 and iteration % 5 == 0 and previous_findings:
        # Find the most common marker type
        from collections import Counter
        markers = []
        for f in previous_findings:
            for m in f.get("markers", []):
                if m.get("measure"):
                    markers.append(m["measure"])
        if markers:
            top_marker = Counter(markers).most_common(1)[0][0]
            # Create a focused query
            focused_queries = [
                f"{top_marker} threshold mental health academic performance",
                f"{top_marker} intervention randomized controlled trial students",
                f"{top_marker} predictive validity mental health biomarker",
            ]
            base = random.choice(focused_queries)
            log.info(f"[Strategy] Using marker-focused query: {base!r}")

    return base


def load_existing_paper_ids() -> set:
    """Load paper_ids already in findings.jsonl to avoid re-processing."""
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
    """Append new findings to findings.jsonl. Returns count appended."""
    if not findings:
        return 0
    with open(FINDINGS_JSONL, "a") as f:
        for finding in findings:
            f.write(json.dumps(asdict(finding)) + "\n")
    return len(findings)


def init_results_tsv():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tevidence_score\tpapers_total\tstatus\tdescription\n")
        log.info("[Init] Created results.tsv")


def log_result(commit: str, score: float, papers_total: int, status: str, description: str):
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{score:.6f}\t{papers_total}\t{status}\t{description}\n")


def git_commit(message: str) -> str:
    """Stage findings.jsonl and results.tsv, commit, return short hash."""
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
    """
    Run one research experiment (one 5-minute cycle):
    1. Get query for this iteration
    2. Crawl papers
    3. Extract findings
    4. Score hypothesis
    5. Log results
    Returns dict with metrics.
    """
    t_start = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"[Research] Iteration {iteration+1} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load existing state
    existing_ids = load_existing_paper_ids()
    prev_findings = load_findings(FINDINGS_JSONL)
    prev_score = compute_evidence_score(FINDINGS_JSONL).evidence_score if prev_findings else 0.0

    # Step 1: Choose query
    query = get_next_query(iteration, prev_findings)
    log.info(f"[Query] {query!r}")

    # Step 2: Crawl papers
    log.info("[Search] Crawling papers from all sources...")
    papers = search_all(
        query=query,
        max_per_source=MAX_PER_SOURCE,
        existing_ids=existing_ids,
    )
    papers_found = len(papers)
    log.info(f"[Search] Found {papers_found} new papers")

    if dry_run:
        log.info("[DryRun] Skipping LLM extraction")
        papers_extracted = 0
        new_findings_count = 0
    else:
        # Step 3: Extract findings
        if papers:
            paper_dicts = [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in papers]
            findings = extract_batch(paper_dicts, existing_ids)
            papers_extracted = len(findings)
            new_findings_count = append_findings(findings)
        else:
            papers_extracted = 0
            new_findings_count = 0

    # Step 4: Score hypothesis
    report = compute_evidence_score(FINDINGS_JSONL)
    delta = report.evidence_score - prev_score

    # Step 5: Determine status
    if new_findings_count == 0 and papers_found == 0:
        status = "no_papers"
    elif delta >= 0.005:
        status = "keep"
    elif delta >= 0:
        status = "marginal"
    else:
        status = "no_gain"

    # Build description
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
    """
    Main research loop — runs until time budget exhausted or max_iterations reached.
    Mirrors the autoresearch LOOP FOREVER pattern.
    """
    t_wall_start = time.time()
    init_results_tsv()

    log.info(f"[Research] Starting physio-research loop")
    log.info(f"[Research] Time budget: {max_wall_seconds}s | Dry run: {dry_run}")
    log.info(f"[Research] Hypothesis: HRV/sleep → mental health + academic performance")

    iteration = 0

    while True:
        elapsed = time.time() - t_wall_start

        # Check time budget
        if elapsed >= max_wall_seconds:
            log.info(f"[Research] Time budget of {max_wall_seconds}s reached after {iteration} iterations. Stopping.")
            break

        # Check iteration limit (for dry-run testing)
        if max_iterations is not None and iteration >= max_iterations:
            log.info(f"[Research] Max iterations ({max_iterations}) reached. Stopping.")
            break

        try:
            metrics = run_experiment(iteration, dry_run=dry_run)

            # Git commit after each experiment
            if not dry_run:
                commit = git_commit(metrics["description"])
                log_result(
                    commit=commit,
                    score=metrics["evidence_score"],
                    papers_total=load_existing_paper_ids().__len__() if not dry_run else 0,
                    status=metrics["status"],
                    description=metrics["description"],
                )

            iteration += 1
            log.info(f"[Research] Iteration {iteration} done. Status: {metrics['status']} | Evidence: {metrics['evidence_score']:.6f}")

        except KeyboardInterrupt:
            log.info("[Research] Interrupted by user. Stopping.")
            break
        except Exception as e:
            log.error(f"[Research] Iteration {iteration} FAILED: {e}")
            import traceback; traceback.print_exc()
            # Log crash and continue
            log_result("0000000", 0.0, 0, "crash", f"[iter{iteration+1}] ERROR: {str(e)[:80]}")
            iteration += 1
            time.sleep(5)

    # Final summary
    final_report = compute_evidence_score(FINDINGS_JSONL)
    log.info(f"\n{'='*60}")
    log.info(f"[Research] FINAL SUMMARY after {iteration} iterations:")
    print_summary(final_report)


# ─── CLI ──────────────────────────────────────────────────────────────────────
from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physio-Research Autonomous Agent")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM extraction (test crawling only)")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations (default: unlimited within time budget)")
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET_SECONDS, help="Wall clock budget in seconds (default: 300)")
    args = parser.parse_args()

    run_loop(
        max_wall_seconds=args.time_budget,
        dry_run=args.dry_run,
        max_iterations=args.iterations,
    )
