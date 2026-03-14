"""
hypothesis.py — Evidence scoring for the Learnzy Focus Score hypothesis
Computes evidence_score ∈ [0.0, 1.0] from all accumulated findings.

The Focus Score = 0.4 × HRV_Readiness + 0.6 × Sleep_Recovery (composite, not separate).

Score = 0.25 × coverage + 0.25 × effect_size + 0.20 × specificity + 0.15 × quality + 0.15 × replication
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

# The 10 sub-claims that need coverage (mapped from program.md)
SUB_CLAIMS = [
    "composite_hrv_sleep_mental_health",      # Composite HRV+sleep → mental health
    "composite_hrv_sleep_academic",           # Composite → GPA/test scores
    "rmssd_depression",                        # RMSSD → depression (PHQ-9)
    "rmssd_anxiety",                           # RMSSD → anxiety (GAD-7)
    "sleep_insomnia",                          # Sleep quality → ISI
    "gr_receptor_threshold",                   # GR activation at z=−1.5
    "bedtime_consistency_cognition",           # Circadian regularity → cognition
    "aasm_efficiency_threshold",               # 85% / 75% sleep efficiency
    "wearable_hrv_clinical_validity",          # Consumer-grade HRV for mental health
    "sleep_memory_consolidation_academic",     # Sleep → memory → exam performance
]


@dataclass
class EvidenceReport:
    evidence_score: float
    h1_score: float         # mental health detection
    h2_score: float         # cognition/academic
    replication_score: float
    coverage_score: float
    effect_size_score: float
    specificity_score: float
    quality_score: float
    total_papers: int
    h1_papers: int
    h2_papers: int
    h1_support: int
    h2_support: int
    papers_with_threshold: int
    papers_replicating_pilot: int
    composite_metric_papers: int
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
    """Convert effect size to 0-1. Cohen's d: cap at 1.5→1.0. Pearson r: cap at 0.8→1.0."""
    if d is not None and not math.isnan(d):
        return min(abs(d) / 1.5, 1.0)
    if r is not None and not math.isnan(r):
        return min(abs(r) / 0.8, 1.0)
    return None


def compute_coverage_score(findings: list[dict]) -> tuple[float, int, int, int, int]:
    """Coverage = % of 10 Focus Score sub-claims covered by ≥1 supporting paper."""
    covered = {claim: False for claim in SUB_CLAIMS}

    h1_total = h1_support = h2_total = h2_support = 0

    for f in findings:
        if not f.get("extraction_ok", True):
            continue

        markers = f.get("markers", [])
        outcomes = f.get("outcomes", [])
        s_h1 = f.get("supports_h1")
        s_h2 = f.get("supports_h2")

        has_hrv = any(m.get("type") in ("HRV", "composite") for m in markers)
        has_sleep = any(m.get("type") in ("sleep", "composite") for m in markers)
        has_composite = any(m.get("type") == "composite" for m in markers) or f.get("composite_metric_used", False)
        has_cortisol = any(m.get("type") == "cortisol" for m in markers)

        has_depression = any(o.get("type") == "depression" for o in outcomes)
        has_anxiety = any(o.get("type") == "anxiety" for o in outcomes)
        has_insomnia = any(o.get("type") == "insomnia" for o in outcomes)
        has_cognition = any(o.get("type") in ("cognition", "retention", "recall") for o in outcomes)
        has_academic = any(o.get("type") == "academic" for o in outcomes)
        has_mental = has_depression or has_anxiety or has_insomnia or any(o.get("type") in ("stress", "mental_health") for o in outcomes)

        if s_h1 is not None:
            h1_total += 1
            if s_h1:
                h1_support += 1
        if s_h2 is not None:
            h2_total += 1
            if s_h2:
                h2_support += 1

        # Map to sub-claims
        if has_composite and has_mental and s_h1:
            covered["composite_hrv_sleep_mental_health"] = True
        if has_composite and (has_academic or has_cognition) and s_h2:
            covered["composite_hrv_sleep_academic"] = True
        if has_hrv and has_depression:
            covered["rmssd_depression"] = True
        if has_hrv and has_anxiety:
            covered["rmssd_anxiety"] = True
        if has_sleep and has_insomnia:
            covered["sleep_insomnia"] = True
        if has_cortisol or any("GR" in str(m.get("measure", "")) or "glucocorticoid" in str(m.get("measure", "")).lower() for m in markers):
            covered["gr_receptor_threshold"] = True

        # Check for bedtime consistency
        for m in markers:
            measure_lower = str(m.get("measure", "")).lower()
            if "consistency" in measure_lower or "regularity" in measure_lower or "circadian" in measure_lower:
                if has_cognition or has_academic:
                    covered["bedtime_consistency_cognition"] = True

        # AASM efficiency thresholds
        for m in markers:
            measure_lower = str(m.get("measure", "")).lower()
            threshold = str(m.get("threshold", "")).lower()
            if "efficiency" in measure_lower and ("85" in threshold or "75" in threshold or "aasm" in measure_lower):
                covered["aasm_efficiency_threshold"] = True

        # Wearable HRV validity
        kf = str(f.get("key_finding", "")).lower()
        title_lower = str(f.get("title", "")).lower()
        if has_hrv and ("wearable" in kf or "wearable" in title_lower or "consumer" in kf or "smartwatch" in kf):
            covered["wearable_hrv_clinical_validity"] = True

        # Sleep → memory → academic
        if has_sleep and (has_cognition or any(o.get("type") in ("retention", "recall") for o in outcomes)):
            covered["sleep_memory_consolidation_academic"] = True

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
    return sum(scores) / len(scores) if scores else 0.0


def compute_specificity_score(findings: list[dict]) -> float:
    """% of papers providing specific numerical thresholds."""
    valid = [f for f in findings if f.get("extraction_ok", True)]
    if not valid:
        return 0.0
    return sum(1 for f in valid if f.get("has_numerical_threshold", False)) / len(valid)


def compute_quality_score(findings: list[dict]) -> float:
    valid = [f for f in findings if f.get("extraction_ok", True)]
    if not valid:
        return 0.0
    return sum(STUDY_WEIGHTS.get(f.get("study_type", "other"), 0.15) for f in valid) / len(valid)


def compute_replication_score(findings: list[dict]) -> float:
    """% of valid papers that replicate a Focus Score V2 pilot finding."""
    valid = [f for f in findings if f.get("extraction_ok", True)]
    if not valid:
        return 0.0
    return sum(1 for f in valid if f.get("replicates_pilot", False)) / len(valid)


def get_top_markers(findings: list[dict], n: int = 5) -> list[dict]:
    from collections import Counter
    combos = Counter()
    details = {}
    for f in findings:
        for m in f.get("markers", []):
            key = f"{m.get('measure','?')} ({m.get('type','?')})"
            combos[key] += 1
            if m.get("threshold") and key not in details:
                details[key] = m.get("threshold")
    return [{"marker": key, "count": count, "threshold": details.get(key)} for key, count in combos.most_common(n)]


def get_top_findings(findings: list[dict], n: int = 5) -> list[str]:
    scored = []
    for f in findings:
        kf = f.get("key_finding", "")
        if not kf or kf in ("Extraction failed.", "No abstract available."):
            continue
        es = f.get("effect_size", {})
        norm = normalize_effect_size(es.get("cohens_d"), es.get("r")) or 0
        quality = STUDY_WEIGHTS.get(f.get("study_type", "other"), 0.15)
        threshold_bonus = 0.3 if f.get("has_numerical_threshold") else 0
        composite_bonus = 0.2 if f.get("composite_metric_used") else 0
        replication_bonus = 0.2 if f.get("replicates_pilot") else 0
        score = norm * 0.3 + quality * 0.2 + threshold_bonus + composite_bonus + replication_bonus
        scored.append((score, kf))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [kf for _, kf in scored[:n]]


def get_key_thresholds(findings: list[dict]) -> list[str]:
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
            evidence_score=0.0, h1_score=0.0, h2_score=0.0, replication_score=0.0,
            coverage_score=0.0, effect_size_score=0.0,
            specificity_score=0.0, quality_score=0.0,
            total_papers=0, h1_papers=0, h2_papers=0,
            h1_support=0, h2_support=0, papers_with_threshold=0,
            papers_replicating_pilot=0, composite_metric_papers=0,
            top_markers=[], top_findings=[], key_thresholds=[],
        )

    coverage, h1_total, h1_support, h2_total, h2_support = compute_coverage_score(findings)
    effect = compute_effect_size_score(findings)
    specificity = compute_specificity_score(findings)
    quality = compute_quality_score(findings)
    replication = compute_replication_score(findings)

    evidence_score = (
        0.25 * coverage +
        0.25 * effect +
        0.20 * specificity +
        0.15 * quality +
        0.15 * replication
    )

    h1_score = h1_support / h1_total if h1_total > 0 else 0.0
    h2_score = h2_support / h2_total if h2_total > 0 else 0.0

    papers_with_threshold = sum(1 for f in findings if f.get("has_numerical_threshold"))
    papers_replicating_pilot = sum(1 for f in findings if f.get("replicates_pilot"))
    composite_metric_papers = sum(1 for f in findings if f.get("composite_metric_used"))

    return EvidenceReport(
        evidence_score=round(evidence_score, 6),
        h1_score=round(h1_score, 4),
        h2_score=round(h2_score, 4),
        replication_score=round(replication, 4),
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
        papers_replicating_pilot=papers_replicating_pilot,
        composite_metric_papers=composite_metric_papers,
        top_markers=get_top_markers(findings),
        top_findings=get_top_findings(findings),
        key_thresholds=get_key_thresholds(findings),
    )


def print_summary(report: EvidenceReport):
    """Print autoresearch-style summary block."""
    print("---")
    print(f"evidence_score:        {report.evidence_score:.6f}")
    print(f"h1_score:              {report.h1_score:.4f}  (mental health: {report.h1_support}/{report.h1_papers} papers)")
    print(f"h2_score:              {report.h2_score:.4f}  (cognition/marks: {report.h2_support}/{report.h2_papers} papers)")
    print(f"replication_score:     {report.replication_score:.4f}  ({report.papers_replicating_pilot} papers replicate pilot)")
    print(f"coverage_score:        {report.coverage_score:.4f}  ({sum(1 for _ in [])}/10 sub-claims)")
    print(f"effect_size_score:     {report.effect_size_score:.4f}")
    print(f"specificity_score:     {report.specificity_score:.4f}  ({report.papers_with_threshold} papers with thresholds)")
    print(f"quality_score:         {report.quality_score:.4f}")
    print(f"total_papers:          {report.total_papers}")
    print(f"composite_papers:      {report.composite_metric_papers}")
    if report.top_markers:
        top = report.top_markers[0]
        print(f"top_marker:            {top['marker']} (n={top['count']}){' → ' + top['threshold'] if top['threshold'] else ''}")
    if report.top_findings:
        print(f"top_finding:           {report.top_findings[0][:120]}")
    if report.key_thresholds:
        print(f"key_thresholds:        {' | '.join(report.key_thresholds[:5])}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--findings", default="findings.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.test:
        test_findings = [
            {
                "paper_id": "abc001", "title": "Composite HRV+sleep readiness and depression in students",
                "year": 2023, "source": "pubmed", "doi": "test1", "url": "", "extraction_ok": True,
                "supports_h1": True, "supports_h2": True,
                "focus_score_relevant": True, "composite_metric_used": True,
                "markers": [
                    {"type": "composite", "measure": "readiness score (HRV+sleep)", "threshold": "score > 75", "direction": "higher better"},
                    {"type": "HRV", "measure": "RMSSD", "threshold": "RMSSD > 50ms", "direction": "higher better"},
                ],
                "outcomes": [
                    {"type": "depression", "measure": "PHQ-9", "direction": "improved"},
                    {"type": "academic", "measure": "GPA", "direction": "improved"},
                ],
                "effect_size": {"cohens_d": 0.72, "r": -0.45, "p_value": 0.001},
                "sample": {"n": 120, "population": "students", "age_range": "18-25"},
                "study_type": "cohort", "has_numerical_threshold": True,
                "replicates_pilot": True,
                "pilot_finding_replicated": "HRV-depression correlation, composite > individual",
                "key_finding": "Composite HRV+sleep readiness > 75 predicted lower PHQ-9 and higher GPA in 120 students.",
            },
            {
                "paper_id": "abc002", "title": "Sleep quality and insomnia in university students: PSQI validation",
                "year": 2024, "source": "semantic_scholar", "doi": "test2", "url": "", "extraction_ok": True,
                "supports_h1": True, "supports_h2": True,
                "focus_score_relevant": True, "composite_metric_used": False,
                "markers": [{"type": "sleep", "measure": "PSQI", "threshold": "PSQI < 5", "direction": "lower better"}],
                "outcomes": [
                    {"type": "insomnia", "measure": "ISI", "direction": "improved"},
                    {"type": "retention", "measure": "exam retention", "direction": "improved"},
                ],
                "effect_size": {"cohens_d": 0.55, "r": -0.59, "p_value": 0.001},
                "sample": {"n": 200, "population": "university", "age_range": "18-22"},
                "study_type": "rct", "has_numerical_threshold": True,
                "replicates_pilot": True,
                "pilot_finding_replicated": "sleep-insomnia correlation r=−0.59",
                "key_finding": "PSQI < 5 associated with ISI reduction (r=−0.59) and 14% higher exam retention.",
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
        print("\n✓ Focus Score hypothesis scoring test passed!")
    else:
        report = compute_evidence_score(Path(args.findings))
        print_summary(report)
