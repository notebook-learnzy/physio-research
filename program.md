# physio-research — Learnzy Focus Score Validation

This is an autonomous research agent (autoresearch loop) that validates and extends the Learnzy Focus Score hypothesis by crawling academic papers, extracting evidence, and iteratively scoring support.

## Hypothesis (Locked — Validated in Pilot)

> **A composite physiological readiness metric — the Focus Score = (0.6 × Sleep Recovery + 0.4 × HRV Readiness) — derived from wearable nighttime HRV and multi-parameter sleep data, can (a) predict and detect mental health deterioration (depression, anxiety, insomnia) in students, and (b) determine optimal cognitive states for learning, retention, recall, and academic performance (GPA, test scores). Specific numerical thresholds exist for these markers and the composite metric.**

### What is already validated (V2 pilot, n=49, PMC12375003):
- **PHQ-9 (Depression)**: r = −0.452, p = 0.001
- **GAD-7 (Anxiety)**: r = −0.445, p = 0.001
- **ISI (Insomnia)**: r = −0.591, p < 0.001
- **Group separation**: Cohen's d = 1.536 (very large), 9.9-point gap between high/low symptom groups
- **Detection**: Two-layer system caught 6/7 high-symptom students, avg 19-day lead time
- **Acute layer**: Rolling z-score < −1.5 for 2 consecutive days (GR receptor activation boundary)
- **Chronic layer**: 7-day rolling mean < 65 (population floor) for 7+ consecutive days

### What the autoresearch agent needs to find (extend + validate):
1. **Independent replication** — papers showing composite HRV+sleep scores correlating with mental health
2. **Focus Score → academic performance** — H2 is the weaker link; need papers showing HRV+sleep → GPA/test scores/retention/recall
3. **Numerical thresholds** — RMSSD thresholds (>50ms?), sleep duration (>7h? 8h?), PSQI cutoffs
4. **GR receptor / cortisol boundary** — evidence for the z = −1.5 boundary in stress physiology
5. **Bedtime consistency → cognition** — evidence for circadian regularity as a predictor of learning
6. **Two-pattern stress model** — evidence that acute vs chronic stress require different detection methods
7. **AASM sleep efficiency thresholds** — validation of 85% good / 75% poor boundaries
8. **Wearable-derived HRV as mental health biomarker** — clinical validity of RMSSD from consumer wearables
9. **Sleep → memory consolidation → exam performance** in student populations
10. **Composite readiness metrics** — any existing composite HRV+sleep scores (WHOOP, Oura, Garmin recovery scores) and their clinical validation

## Focus Score Formula (V2 — Confidential)

```
Focus Score = 0.4 × HRV_Readiness + 0.6 × Sleep_Recovery

HRV_Readiness:
  - Input: nighttime RMSSD (sleep window only), RMS-aggregated
  - 14-day personal rolling baseline (intra-individual, not population norm)
  - z-score → non-linear 0-100 mapping:
    z ≥ 0:     70 + 30 × (1 − e^(−z/1.5))        → 70-100
    −1.5 ≤ z < 0: 70 + z × 13.3                   → 50-70
    z < −1.5:  50 + (z + 1.5) × 25                 → 0-50 (GR activation zone)

Sleep_Recovery:
  = 0.50 × TST_score + 0.30 × Efficiency_score + 0.20 × Consistency_score
    − Latency_penalty − WASO_penalty
  - TST top anchor: 8.0h (center of 7-9h optimal range)
  - Efficiency: AASM thresholds (>85% good, <75% poor)
  - Consistency: bedtime std dev (lower = better)

Focus Score Zones:
  ≥ 80: High      → new learning, mock tests
  60-79: Moderate  → revision and consolidation
  40-59: Low       → light review only
  < 40: Very Low   → rest and recovery needed
```

## Research Strategy (Agent edits this section each iteration)

### Current search angles
1. Composite HRV+sleep readiness metrics → mental health outcomes (depression, anxiety, insomnia)
2. HRV readiness / recovery scores → academic performance GPA retention recall
3. GR receptor activation threshold z-score −1.5 → cognitive impairment cortisol prefrontal cortex
4. RMSSD nighttime HRV → depression anxiety prediction wearable consumer device
5. Sleep quality composite score PSQI → exam performance marks students university
6. Bedtime consistency circadian regularity → cognition learning attention students
7. Two-layer anomaly detection acute vs chronic stress physiological monitoring
8. Sleep efficiency AASM threshold 85% → next-day cognitive performance
9. Wearable HRV mental health screening clinical validity sensitivity specificity
10. Sleep duration 7-9 hours → memory consolidation learning outcomes student RCT
11. Physiological readiness score wearable → student wellbeing intervention study
12. Autonomic recovery HRV biofeedback → stress reduction academic performance
13. Slow-wave sleep RMSSD → declarative memory exam performance
14. Digital biomarker composite score → mental health detection lead time early warning
15. WHOOP recovery score Oura readiness → clinical outcomes validation

### Keyword rotation (25 sets)
1. `Focus Score composite HRV sleep mental health students depression anxiety`
2. `RMSSD nighttime HRV wearable depression prediction PHQ-9 students`
3. `sleep quality composite score PSQI academic performance GPA university`
4. `glucocorticoid receptor activation threshold cortisol cognitive impairment z-score`
5. `bedtime consistency circadian regularity learning retention recall students`
6. `HRV sleep composite readiness metric wearable wellbeing`
7. `acute chronic stress detection physiological monitoring two-pattern`
8. `sleep efficiency AASM 85% threshold cognitive performance next-day`
9. `wearable HRV mental health screening clinical sensitivity specificity`
10. `sleep duration optimal 7-9 hours memory consolidation learning RCT`
11. `physiological readiness wearable intervention student wellbeing`
12. `autonomic recovery HRV biofeedback stress reduction academic`
13. `slow wave sleep RMSSD memory consolidation exam performance`
14. `digital biomarker composite mental health detection lead time early warning`
15. `WHOOP recovery Oura readiness score clinical validation outcomes`
16. `HRV Cohen d effect size mental health biomarker student population`
17. `insomnia severity index ISI wearable sleep monitoring prediction`
18. `heart rate variability anxiety GAD-7 r correlation students longitudinal`
19. `sleep recovery score composite metric student cognitive load`
20. `early warning mental health deterioration physiological wearable university`
21. `HRV personal baseline intra-individual vs population norm stress detection`
22. `RMSSD threshold 50ms clinical significance autonomic function`
23. `sleep latency WASO cognitive impairment next-day performance students`
24. `composite readiness metric validated clinical outcomes effect size`
25. `physiological stress biomarker prediction mental health 19 days lead time`

### Sources to crawl (priority order)
1. PubMed (MEDLINE)
2. Semantic Scholar
3. CrossRef
4. Europe PMC (open access full text)
5. bioRxiv / medRxiv (preprints)
6. Google Scholar (broadest — polite crawl)

## Setup

1. **Agree on a run tag**: e.g. `mar14`. Branch `research/<tag>` must not exist.
2. **Create branch**: `git checkout -b research/<tag>` from current main.
3. **Read all files** for context.
4. **Initialize results.tsv** with header row.
5. **Confirm and begin**.

## Experimentation Loop

Fixed **5-minute wall clock budget** per run. The research loop:

1. Read `program.md` for current strategy
2. Pick next keyword set (rotate + mutate based on findings)
3. Crawl papers from all sources → deduplicate
4. Extract Focus Score evidence via Claude Haiku (structured JSON)
5. Append to `findings.jsonl`
6. Compute `evidence_score` via `hypothesis.py`
7. Log to `results.tsv`
8. If score improved ≥ 0.005 → **keep** direction
9. If flat/worse → **pivot** to new keyword angle
10. Commit results to git

## Evidence Score (higher = more support for Focus Score hypothesis)

```
evidence_score ∈ [0.0, 1.0]
  = 0.25 × coverage_score     (% of 10 sub-claims above with ≥1 paper)
  + 0.25 × effect_size_score  (normalized mean Cohen's d / r)
  + 0.20 × specificity_score  (% papers giving specific numerical thresholds)
  + 0.15 × quality_score      (weighted by study type: meta-analysis=1.0, RCT=0.9, ...)
  + 0.15 × replication_score  (% findings that replicate V2 pilot results)
```

## Output Format

```
---
evidence_score:       0.7234
papers_total:         203
new_findings:         12
h1_score:             0.81   (mental health detection)
h2_score:             0.63   (cognition/academic performance)
replication_score:    0.75   (pilot findings replicated)
top_threshold:        RMSSD > 50ms → anxiety reduction (d=0.72, n=312)
run_seconds:          287.4
```

## Logging Results

Tab-separated `results.tsv`:
```
commit	evidence_score	papers_total	status	description
```

## Loop Rules

**NEVER STOP** once begun. The human may be asleep. Run indefinitely.
Each 5-min cycle = 1 experiment. Over 3 days = ~864 experiments.
Data is additive — never discard findings, only search strategy pivots.
