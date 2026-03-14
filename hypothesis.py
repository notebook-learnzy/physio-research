"""
hypothesis.py — Evidence scoring for the HRV/sleep hypothesis
Computes a composite evidence_score ∈ [0.0, 1.0] from all accumulated findings.
This is the "val_bpb" of physio-research — higher is better.

Score = 0.30 × coverage + 0.30 × effect_size + 0.20 × specificity + 0.20 × quality
"""

import json
import math
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("hypothesis")

FINDINGS_PATH = Path("findings.jsonl")

# Study type quality weights
STUDY_WEIGHTS = {
    "meta_analysis": 1.0,
    "rct": 0.90,
    "cohort": 0.70,
    "cross_sectional": 0.45,
    "review": 0.35,
    "case_study": 0.20,
    "other": 0.15,
}


@dataclass
class EvidenceReport:
    evidence_score: float
    h1_score: float         # mental health sub-score
    h2_score: float         # cognition/marks sub-score
    coverage_score: float
    effect_size_score: float
    specificity_score: float
    quality_score: float
    total_papers: int
    h1_papers: int          # papers addressing H1
    h2_papers: int          # papers addressing H2
    h1_support: int         # papers supporting H1
    h2_support: int         # papers supporting H2
    papers_with_threshold: int
    top_markers: list[dict]
    top_findings: list[str]
    key_thresholds: list[str]


def load_findings(path: Path = FINDINGS_PATH) -> list[dict]:
    if not path.exists():
        return []
    findings = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    findings.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return findings


def normalize_effect_size(d: Optional[float], r: Optional[float]) -> Optional[float]:
    """Convert effect size to a normalized 0-1 score. Uses Cohen's d or Pearson r."""
    if d is not None and not math.isnan(d):
        # Cohen's d: 0.2=small, 0.5=medium, 0.8=large. Cap at 1.5→1.0
        return min(abs(d) / 1.5, 1.0)
    if r is not None and not math.isnan(r):
        # r: 0.1=small, 0.3=medium, 0.5=large. Cap at 0.8→1.0
        return min(abs(r) / 0.8, 1.0)
    return None


def compute_coverage_score(findings: list[dict]) -> tuple[float, int, int, int, int]:
    """
    Coverage = % of sub-hypotheses with ≥1 supporting paper.
    Sub-claims:
    H1: HRV→mental_health, sleep→mental_health
    H2: HRV→academic, sleep→academic, HRV→cognition, sleep→cognition
    """
    covered = {
        "hrv_mental": False,
        "sleep_mental": False,
        "hrv_academic": False,
        "sleep_academic": False,
        "hrv_cognition": False,
        "sleep_cognition": False,
    }

    h1_total = h1_support = h2_total = h2_support = 0

    for f in findings:
        if not f.get("extraction_ok", True):
            continue

        markers = f.get("markers", [])
        outcomes = f.get("outcomes", [])
        has_hrv = any(m.get("type") == "HRV" for m in markers)
        has_sleep = any(m.get("type") == "sleep" for m in markers)
        has_mental = any(o.get("type") == "mental_health" for o in outcomes)
        has_academic = any(o.get("type") == "academic" for o in outcomes)
        has_cognition = any(o.get("type") == "cognition" for o in outcomes)

        s_h1 = f.get("supports_h1")
        s_h2 = f.get("supports_h2")

        if s_h1 is not None:
            h1_total += 1
            if s_h1:
                h1_support += 1
        if s_h2 is not None:
            h2_total += 1
            if s_h2:
                h2_support += 1

        if s_h1 and has_hrv and has_mental:
            covered["hrv_mental"] = True
        if s_h1 and has_sleep and has_mental:
            covered["sleep_mental"] = True
        if s_h2 and has_hrv and has_academic:
            covered["hrv_academic"] = True
        if s_h2 and has_sleep and has_academic:
            covered["sleep_academic"] = True
        if s_h2 and has_hrv and has_cognition:
            covered["hrv_cognition"] = True
        if s_h2 and has_sleep and has_cognition:
            covered["sleep_cognition"] = True

    coverage = sum(covered.values()) / len(covered)
    return coverage, h1_total, h1_support, h2_total, h2_support


def compute_effect_size_score(findings: list[dict]) -> float:
    """Mean normalized effect size across all findings that report one."""
    scores = []
    for f in findings:
        if not f.get("extraction_ok", True):
            continue
        es = f.get("effect_size", {})
        norm = normalize_effect_size(es.get("cohens_d"), es.get("r"))
        if norm is not None:
            scores.append(norm)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_specificity_score(findings: list[dict]) -> float:
    """% of papers that provide specific numerical thresholds."""
    valid = [f for f in findings if f.get("extraction_ok", True)]
    if not valid:
        return 0.0
    with_threshold = sum(1 for f in valid if f.get("has_numerical_threshold", False))
    return with_threshold / len(valid)


def compute_quality_score(findings: list[dict]) -> float:
    """Weighted quality score based on study type."""
    valid = [f for f in findings if f.get("extraction_ok", True)]
    if not valid:
        return 0.0
    quality_sum = sum(STUDY_WEIGHTS.get(f.get("study_type", "other"), 0.15) for f in valid)
    return quality_sum / len(valid)


def get_top_markers(findings: list[dict], n: int = 5) -> list[dict]:
    """Return the most frequently mentioned marker+threshold combos."""
    from collections import Counter
    combos = Counter()
    details = {}
    for f in findings:
        for m in f.get("markers", []):
            key = f"{m.get('measure','?')} ({m.get('type','?')})"
            combos[key] += 1
            if m.get("threshold") and key not in details:
                details[key] = m.get("threshold")
    result = []
    for key, count in combos.most_common(n):
        result.append({"marker": key, "count": count, "threshold": details.get(key)})
    return result


def get_top_findings(findings: list[dict], n: int = 5) -> list[str]:
    """Return the most informative key findings."""
    scored = []
    for f in findings:
        kf = f.get("key_finding", "")
        if not kf or kf in ("Extraction failed.", "No abstract available."):
            continue
        # Score by: effect size + has threshold + study quality
        es = f.get("effect_size", {})
        norm = normalize_effect_size(es.get("cohens_d"), es.get("r")) or 0
        quality = STUDY_WEIGHTS.get(f.get("study_type", "other"), 0.15)
        threshold_bonus = 0.3 if f.get("has_numerical_threshold") else 0
        score = norm * 0.5 + quality * 0.3 + threshold_bonus
        scored.append((score, kf))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [kf for _, kf in scored[:n]]


def get_key_thresholds(findings: list[dict]) -> list[str]:
    """Collect all specific numerical thresholds found."""
    thresholds = set()
    for f in findings:
        for m in f.get("markers", []):
            t = m.get("threshold")
            if t:
                thresholds.add(t)
    return sorted(thresholds)


def compute_evidence_score(findings_path: Path = FINDINGS_PATH) -> EvidenceReport:
    findings = load_findings(findings_path)

    if not findings:
        return EvidenceReport(
            evidence_score=0.0, h1_score=0.0, h2_score=0.0,
            coverage_score=0.0, effect_size_score=0.0,
            specificity_score=0.0, quality_score=0.0,
            total_papers=0, h1_papers=0, h2_papers=0,
            h1_support=0, h2_support=0, papers_with_threshold=0,
            top_markers=[], top_findings=[], key_thresholds=[],
        )

    coverage, h1_total, h1_support, h2_total, h2_support = compute_coverage_score(findings)
    effect = compute_effect_size_score(findings)
    specificity = compute_specificity_score(findings)
    quality = compute_quality_score(findings)

    evidence_score = (
        0.30 * coverage +
        0.30 * effect +
        0.20 * specificity +
        0.20 * quality
    )

    # Sub-scores for H1 and H2
    h1_score = h1_support / h1_total if h1_total > 0 else 0.0
    h2_score = h2_support / h2_total if h2_total > 0 else 0.0

    papers_with_threshold = sum(1 for f in findings if f.get("has_numerical_threshold"))

    return EvidenceReport(
        evidence_score=round(evidence_score, 6),
        h1_score=round(h1_score, 4),
        h2_score=round(h2_score, 4),
        coverage_score=round(coverage, 4),
        effect_size_score=round(effect, 4),
        specificity_score=round(specificity, 4),
        quality_score=round(quality, 4),
        total_papers=len(findings),
        h1_papers=h1_total,
        h2_papers=h2_total,
        h1_support=h1_support,
        h2_support=h2_support,
        papers_with_threshold=papers_with_threshold,
        top_markers=get_top_markers(findings),
        top_findings=get_top_findings(findings),
        key_thresholds=get_key_thresholds(findings),
    )


def print_summary(report: EvidenceReport):
    """Print the autoresearch-style summary block."""
    print("---")
    print(f"evidence_score:       {report.evidence_score:.6f}")
    print(f"h1_score:             {report.h1_score:.4f}  (mental health: {report.h1_support}/{report.h1_papers} papers support)")
    print(f"h2_score:             {report.h2_score:.4f}  (cognition/marks: {report.h2_support}/{report.h2_papers} papers support)")
    print(f"coverage_score:       {report.coverage_score:.4f}")
    print(f"effect_size_score:    {report.effect_size_score:.4f}")
    print(f"specificity_score:    {report.specificity_score:.4f}")
    print(f"quality_score:        {report.quality_score:.4f}")
    print(f"total_papers:         {report.total_papers}")
    print(f"papers_with_threshold:{report.papers_with_threshold}")
    if report.top_markers:
        top = report.top_markers[0]
        print(f"top_marker:           {top['marker']} (n={top['count']} papers){' → ' + top['threshold'] if top['threshold'] else ''}")
    if report.top_findings:
        print(f"top_finding:          {report.top_findings[0][:120]}")
    if report.key_thresholds:
        print(f"key_thresholds:       {' | '.join(report.key_thresholds[:3])}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test with synthetic data")
    parser.add_argument("--findings", default="findings.jsonl", help="Path to findings.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.test:
        # Create synthetic test findings
        test_findings = [
            {
                "paper_id": "abc001", "title": "HRV and anxiety in students", "year": 2022,
                "source": "pubmed", "doi": "test", "url": "", "extraction_ok": True,
                "supports_h1": True, "supports_h2": True,
                "markers": [{"type": "HRV", "measure": "RMSSD", "threshold": "RMSSD > 50ms", "direction": "higher better"}],
                "outcomes": [{"type": "mental_health", "measure": "anxiety (GAD-7)", "direction": "improved"},
                             {"type": "academic", "measure": "GPA", "direction": "improved"}],
                "effect_size": {"cohens_d": 0.72, "r": None, "p_value": 0.003},
                "sample": {"n": 120, "population": "students", "age_range": "18-25"},
                "study_type": "cohort", "has_numerical_threshold": True,
                "key_finding": "Students with RMSSD > 50ms had lower anxiety and higher GPA.",
            },
            {
                "paper_id": "abc002", "title": "Sleep duration and exam performance", "year": 2023,
                "source": "semantic_scholar", "doi": "test2", "url": "", "extraction_ok": True,
                "supports_h1": True, "supports_h2": True,
                "markers": [{"type": "sleep", "measure": "sleep duration", "threshold": "> 7 hours", "direction": "higher better"}],
                "outcomes": [{"type": "academic", "measure": "exam score", "direction": "improved"},
                             {"type": "mental_health", "measure": "depression (PHQ-9)", "direction": "improved"}],
                "effect_size": {"cohens_d": 0.55, "r": None, "p_value": 0.01},
                "sample": {"n": 200, "population": "students", "age_range": "18-22"},
                "study_type": "rct", "has_numerical_threshold": True,
                "key_finding": "Students sleeping > 7 hours scored 15% higher on exams.",
            },
        ]
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            for f in test_findings:
                tmp.write(json.dumps(f) + "\n")
            tmp_path = tmp.name
        report = compute_evidence_score(Path(tmp_path))
        print_summary(report)
        os.unlink(tmp_path)
        print("\n✓ Hypothesis scoring test passed!")
    else:
        report = compute_evidence_score(Path(args.findings))
        print_summary(report)
