# physio-research — Learnzy Focus Score Validation

*Autonomous scientific hypothesis validation — inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)*

![Runs every 5 min](https://img.shields.io/badge/runs%20every-5%20minutes-blueviolet) ![Claude Haiku](https://img.shields.io/badge/LLM-Claude%20Haiku-orange) ![Focus Score](https://img.shields.io/badge/metric-Focus%20Score-green)

## Hypothesis (Validated in Pilot — n=49, PMC12375003)

> **The Focus Score — a composite physiological readiness metric (0.4 × HRV Readiness + 0.6 × Sleep Recovery, scaled 0-100) derived from wearable nighttime HRV and multi-parameter sleep data — can (a) predict and detect mental health deterioration (depression, anxiety, insomnia) in students, and (b) determine optimal cognitive states for learning, retention, recall, and academic performance.**

### Pilot validation results (Cohen's d = 1.536):
| Outcome | Pearson r | p-value |
|---------|-----------|---------|
| PHQ-9 (Depression) | −0.452 | 0.001 |
| GAD-7 (Anxiety) | −0.445 | 0.001 |
| ISI (Insomnia) | −0.591 | < 0.001 |

Two-layer detection system caught 6/7 high-symptom students with 19-day avg lead time.

## How it works

| autoresearch | physio-research |
|---|---|
| Modifies `train.py` | Refines search strategy |
| Trains for 5 minutes | Crawls papers for 5 minutes |
| Minimizes `val_bpb` | Maximizes `evidence_score` |
| Keeps if loss improves | Keeps if evidence grows |

### The Loop (every 5 minutes on GitHub Actions)

```
1. Pick next keyword set (rotate + mutate based on past findings)
2. Crawl papers: PubMed, Semantic Scholar, CrossRef, Europe PMC, bioRxiv, Google Scholar
3. Extract Focus Score evidence via Claude Haiku (structured JSON per paper)
4. Score hypothesis: evidence_score ∈ [0.0, 1.0]
5. Log to results.tsv, commit to git
```

### Evidence Score

```
evidence_score
  = 0.25 × coverage       (% of 10 Focus Score sub-claims covered)
  + 0.25 × effect_size    (normalized mean Cohen's d / Pearson r)
  + 0.20 × specificity    (% papers with numerical thresholds)
  + 0.15 × quality        (weighted by study type: RCT=0.9, cohort=0.7, ...)
  + 0.15 × replication    (% findings replicating pilot results)
```

## Project Structure

```
physio-research/
├── program.md              ← human edits research direction (editable)
├── research.py             ← main loop orchestrator
├── search.py               ← multi-source paper crawler (6 sources, no API keys)
├── extract.py              ← Claude Haiku Focus Score evidence extractor
├── hypothesis.py           ← evidence scoring (10 sub-claims)
├── findings.jsonl          ← accumulated structured evidence
├── results.tsv             ← experiment log
└── .github/workflows/
    └── research.yml        ← GitHub Actions (every 5 min, 3-day autopilot)
```

## Setup

### 1. Add GitHub Secret

| Secret | Required |
|--------|----------|
| `ANTHROPIC_API_KEY` | Yes — for Claude Haiku extractions |

### 2. Push to GitHub

```bash
cd "/Users/hg/karpathy demo/physio-research"
git remote add origin https://github.com/YOUR_USERNAME/physio-research
git push -u origin research/mar14
```

### 3. Enable Actions

Push triggers the workflow. Runs every 5 minutes for 3 days.

### 4. Local testing

```bash
# Dry run (no LLM calls)
python research.py --dry-run --iterations 1

# Full run (1 iteration)
ANTHROPIC_API_KEY=your_key python research.py --iterations 1
```

## Token Budget

| Metric | Value |
|--------|-------|
| Rate limit | 2,000 tokens/min |
| Per run | ~10K tokens |
| Per day (288 runs) | ~2.88M tokens |
| 3-day total | ~8.64M tokens |
| Cost (Claude Haiku) | ~$0.86 |

## Learnzy

Built by Himanshu Gupta — [learnzy.in](https://learnzy.in)

## License

MIT
