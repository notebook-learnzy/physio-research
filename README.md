# physio-research

*Autonomous scientific hypothesis validation — inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)*

![Concept: autoresearch for science](https://img.shields.io/badge/runs%20every-5%20minutes-blueviolet) ![powered by Claude Haiku](https://img.shields.io/badge/LLM-Claude%20Haiku-orange)

## Hypothesis

> **Physiological markers — specifically Heart Rate Variability (HRV) and sleep quality/duration — can (a) predict and reduce mental health problems (stress, anxiety, depression) in students, and (b) causally improve academic cognition and marks in students. Specific, actionable numerical thresholds exist for these markers.**

Sub-hypotheses:
- **H1 (Mental Health)**: Higher HRV / better sleep → lower anxiety, stress, depression in students
- **H2 (Cognition/Marks)**: Higher HRV / better sleep → higher GPA, exam scores, cognitive performance

## How it works

Mirrors autoresearch's loop but for **scientific validation** instead of ML training:

| autoresearch | physio-research |
|---|---|
| Modifies `train.py` | Refines search strategy |
| Trains for 5 minutes | Crawls papers for 5 minutes |
| Minimizes `val_bpb` | Maximizes `evidence_score` |
| Keeps if validation improves | Keeps direction if score improves |

### The Loop (every 5 minutes on GitHub Actions)

```
1. Pick next keyword set (rotate + mutate based on past findings)
2. Crawl papers: PubMed, Semantic Scholar, CrossRef, Europe PMC, bioRxiv, Google Scholar
3. Extract biomarker evidence via Claude Haiku (structured JSON per paper)
4. Score hypothesis: evidence_score ∈ [0.0, 1.0]
5. Log to results.tsv, commit to git
6. If score ↑ ≥ 0.005 → keep direction | else pivot to new keywords
```

### Evidence Score (higher = more support for hypothesis)

```
evidence_score
  = 0.30 × coverage_score     (% of H1+H2 sub-claims covered by ≥1 paper)
  + 0.30 × effect_size_score  (normalized mean Cohen's d / Pearson r)
  + 0.20 × specificity_score  (% papers giving specific numerical thresholds)
  + 0.20 × quality_score      (weighted by study type: RCT=0.9, cohort=0.7, ...)
```

## Project Structure

```
physio-research/
├── program.md              ← human edits research direction + keyword sets
├── research.py             ← main loop orchestrator (do not modify)
├── search.py               ← multi-source paper crawler (6 sources)
├── extract.py              ← Claude Haiku biomarker extractor
├── hypothesis.py           ← evidence scoring engine
├── findings.jsonl          ← accumulated paper evidence (grows over time)
├── results.tsv             ← experiment log
├── requirements.txt
└── .github/
    └── workflows/
        └── research.yml    ← GitHub Actions (every 5 min, 3-day run)
```

## Setup

### 1. Fork / clone this repo

```bash
git clone <your-repo-url>
cd physio-research
```

### 2. Add GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (for Claude Haiku extractions) |

No other API keys needed — all paper sources are crawled via free public endpoints.

### 3. Initialize the branch

```bash
git checkout -b research/mar14
```

### 4. Install dependencies (for local testing)

```bash
pip install -r requirements.txt
```

### 5. Run a dry-run test (no LLM calls)

```bash
python research.py --dry-run --iterations 1
```

### 6. Run a single full experiment locally

```bash
ANTHROPIC_API_KEY=your_key python research.py --iterations 1
```

### 7. Enable GitHub Actions

Push to GitHub — the workflow triggers automatically every 5 minutes.

```bash
git add .
git commit -m "init: physio-research"
git push origin research/mar14
```

For manual trigger: Actions tab → "Physio-Research Autonomous Loop" → Run workflow.

## Token Budget

- **Rate limit**: 2,000 tokens/minute (enforced in `extract.py`)
- **Per run**: ~5 min × 2,000 tok/min = ~10,000 tokens/run
- **Per day**: ~288 runs × 10,000 tokens = ~2.88M tokens/day
- **3-day total**: ~8.64M tokens → ~$0.86 at Claude Haiku pricing ($0.10/1M input, $0.30/1M output)

## Output

After each run, `results.tsv` is updated:

```
commit	evidence_score	papers_total	status	description
a1b2c3d	0.000000	0	keep	[iter1] baseline | +12 findings | delta=+0.0000
b2c3d4e	0.423100	47	keep	[iter2] HRV RMSSD anxiety students | +8 findings | delta=+0.4231
c3d4e5f	0.531200	89	keep	[iter3] sleep PSQI GPA university | +15 findings | delta=+0.1081
```

And `findings.jsonl` accumulates structured evidence:

```json
{"paper_id": "abc123", "title": "...", "supports_h1": true, "supports_h2": true,
 "markers": [{"type": "HRV", "measure": "RMSSD", "threshold": "RMSSD > 50ms", ...}],
 "effect_size": {"cohens_d": 0.72, "r": null, "p_value": 0.003}, ...}
```

## Tuning

Edit `program.md` to change the research direction — add new keyword sets, prioritize specific sources, or focus the hypothesis. This is the human-in-the-loop part, just like editing `program.md` in autoresearch.

## License

MIT
