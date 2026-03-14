# physio-research

This is an autonomous research agent that validates a scientific hypothesis by crawling papers, extracting biomarker evidence, and iteratively scoring hypothesis support — mirroring the autoresearch loop but for science instead of ML training.

## Hypothesis

> **Physiological markers — specifically Heart Rate Variability (HRV) and sleep quality/duration — can (a) predict and reduce mental health problems (stress, anxiety, depression) in students, and (b) causally improve academic cognition and marks (GPA, test scores) in students. Specific, actionable numerical thresholds for these markers exist and can be identified.**

The two sub-hypotheses:
- **H1 (Mental Health)**: Higher HRV / better sleep → lower anxiety, stress, and depression scores in students
- **H2 (Cognition/Marks)**: Higher HRV / better sleep → higher academic performance (GPA, marks, test scores)

## Research Strategy (Agent edits this section each iteration)

### Current search angles
1. HRV (RMSSD, SDNN, LF/HF ratio) + anxiety/depression/stress in students
2. Sleep duration/quality (PSQI score, actigraphy) + academic performance GPA
3. HRV biofeedback intervention → cognitive improvement
4. Sleep deprivation → attention, working memory, executive function in students
5. Wearable HRV monitoring + mental health outcomes (prospective studies)
6. Slow-wave sleep + memory consolidation + learning outcomes
7. HRV threshold values (RMSSD 50ms, SDNN 100ms) clinical significance
8. Mindfulness + HRV + stress + student performance (RCT)

### Keyword sets
- Primary: `HRV heart rate variability mental health students academic performance`
- Secondary: `RMSSD SDNN anxiety depression university students`
- Tertiary: `sleep quality PSQI cognition GPA marks examination`
- Quaternary: `autonomic nervous system stress cortisol student performance`
- Quinary: `HRV biofeedback intervention cognition randomized controlled trial`

### Sources to crawl (in priority order)
1. PubMed (pubmed.ncbi.nlm.nih.gov) — MEDLINE database
2. Semantic Scholar (api.semanticscholar.org) — broad coverage
3. CrossRef (api.crossref.org) — DOI metadata + abstract
4. bioRxiv / medRxiv (biorxiv.org) — preprints
5. Europe PMC (europepmc.org) — open access full text
6. Google Scholar (scholar.google.com) — broadest coverage via crawl

## Setup

To set up a new research run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**: Read these for context:
   - `program.md` — this file (research direction)
   - `search.py` — crawler + search logic
   - `extract.py` — LLM extraction layer
   - `hypothesis.py` — evidence scoring (the "val_bpb" equivalent)
   - `research.py` — main orchestrator loop
4. **Initialize results.tsv**: Create with just header row; baseline recorded after first run.
5. **Confirm and begin**.

## Experimentation Loop

Each run has a **fixed 5-minute wall clock budget**. The research loop:

1. Read `program.md` for current search angles and keyword sets
2. Pick the next keyword set (rotate + mutate based on previous findings)
3. Crawl papers from all sources → deduplicate by DOI/title
4. Extract biomarker evidence via Claude Haiku (structured JSON)
5. Append to `findings.jsonl`
6. Compute `evidence_score` via `hypothesis.py`
7. Log to `results.tsv`
8. If score improved → **keep** direction (mutate keywords slightly)
9. If score flat/worse → **pivot** (try new keyword angle)
10. Commit all results to git

## Evidence Score (the metric — higher is better)

Unlike `val_bpb` where lower is better, here **higher score = more evidence for hypothesis**.

```
evidence_score ∈ [0.0, 1.0]
  = 0.30 × coverage_score    (% of H1+H2 sub-claims with ≥1 paper)
  + 0.30 × effect_size_score (normalized mean Cohen's d or r across studies)
  + 0.20 × specificity_score (% papers giving specific numerical thresholds)
  + 0.20 × quality_score     (weighted by study type: RCT=1.0, cohort=0.7, cross-sect=0.4)
```

## Output Format

Each run prints a summary:

```
---
evidence_score:    0.7234
papers_found:      47
papers_extracted:  31
new_findings:      12
h1_support:        0.81
h2_support:        0.63
top_marker:        RMSSD > 50ms → anxiety reduction (d=0.72, n=312)
run_seconds:       287.4
total_papers:      203
```

Extract key metric: `grep "^evidence_score:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit  evidence_score  papers_total  status  description
```

- `commit`: 7-char git hash
- `evidence_score`: float (e.g. 0.723400) — use 0.000000 for crashes
- `papers_total`: total papers in findings.jsonl
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what search angle was tried

## The Loop Rules

**The first run**: Always run baseline (current keyword set as-is).

**NEVER STOP**: Once begun, do NOT pause for human input. Run until manually stopped.
Each 5-min cycle = 1 experiment. Over 3 days (~72h) = ~864 experiments.

**Improvement threshold**: An evidence_score increase of ≥0.005 is "meaningful". Keep direction.
Below that: log as "no_gain" but still keep all findings (they are additive — never discard data).

**Crashes**: If crawl fails, retry once, then skip that source and log "crash".

**Rate limiting**: Respect source rate limits — 100ms delay between requests, exponential backoff on 429.
