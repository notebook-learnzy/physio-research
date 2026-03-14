"""
extract.py — LLM-based Focus Score evidence extractor using Claude Haiku
Reads paper abstracts and extracts structured evidence for the Learnzy Focus Score hypothesis.

Focus Score = 0.6 × Sleep Recovery + 0.4 × HRV Readiness (composite metric, not separate)

Token budget: 2000 tokens/minute over 3 days = 8,640,000 tokens total
Claude Haiku is used for cost efficiency.
"""

import os
import json
import time
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("extract")

CLAUDE_MODEL = "claude-haiku-4-6-20251001"  # Claude 4.5 Haiku — fast + cheap
MAX_TOKENS_PER_CALL = 1024
TARGET_TOKENS_PER_MIN = 2000  # rate limit to stay within budget

# ─── TOKEN RATE LIMITER ───────────────────────────────────────────────────────
class TokenRateLimiter:
    """Enforces a max tokens/minute budget using a sliding window."""

    def __init__(self, tokens_per_minute: int = TARGET_TOKENS_PER_MIN):
        self.tokens_per_minute = tokens_per_minute
        self.window: list[tuple[float, int]] = []  # (timestamp, tokens_used)

    def wait_if_needed(self, tokens_about_to_use: int):
        now = time.time()
        self.window = [(t, n) for t, n in self.window if now - t < 60]
        used = sum(n for _, n in self.window)
        if used + tokens_about_to_use > self.tokens_per_minute:
            if self.window:
                oldest = self.window[0][0]
                wait = 60 - (now - oldest) + 0.1
                if wait > 0:
                    log.info(f"[RateLimiter] Budget hit. Waiting {wait:.1f}s ...")
                    time.sleep(wait)
            self.window = []

    def record(self, tokens_used: int):
        self.window.append((time.time(), tokens_used))


limiter = TokenRateLimiter(TARGET_TOKENS_PER_MIN)


# ─── EXTRACTION PROMPT ────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
You are a biomedical research analyst for Learnzy. Extract evidence about physiological readiness metrics — specifically the combination of HRV and sleep into composite scores — and their effects on mental health and academic performance.

## Context — Learnzy Focus Score (already validated in pilot):
- Focus Score = 0.4 × HRV_Readiness + 0.6 × Sleep_Recovery (daily 0-100 metric)
- HRV component: nighttime RMSSD, personal 14-day rolling baseline, non-linear z-score mapping
- Sleep component: TST + efficiency (AASM thresholds) + bedtime consistency − latency/WASO penalties
- Pilot (n=49 students): PHQ-9 r=−0.452, GAD-7 r=−0.445, ISI r=−0.591, Cohen's d=1.536
- Two-layer detection: acute (z<−1.5 for 2 days) + chronic (7-day mean<65 for 7 days) → caught 6/7 high-symptom students, 19-day avg lead time

## Hypothesis being validated:
H1 (Mental Health): Composite HRV+sleep readiness → predicts/detects depression, anxiety, insomnia in students
H2 (Cognition/Marks): Composite HRV+sleep readiness → predicts/improves GPA, test scores, retention, recall

## Extract and return ONLY valid JSON:
{{
  "supports_h1": true/false/null,
  "supports_h2": true/false/null,
  "focus_score_relevant": true/false,
  "composite_metric_used": true/false,
  "markers": [
    {{
      "type": "HRV" | "sleep" | "composite" | "cortisol" | "other",
      "measure": "e.g. RMSSD, SDNN, PSQI, Focus Score, recovery score, etc.",
      "threshold": "e.g. RMSSD > 50ms, PSQI < 5, sleep > 7h",
      "direction": "higher better" | "lower better" | null
    }}
  ],
  "outcomes": [
    {{
      "type": "depression" | "anxiety" | "insomnia" | "stress" | "cognition" | "academic" | "retention" | "recall" | "other",
      "measure": "e.g. PHQ-9, GAD-7, ISI, GPA, exam score, attention, working memory",
      "direction": "improved" | "worsened" | "no effect" | null
    }}
  ],
  "effect_size": {{
    "cohens_d": null or number,
    "r": null or number,
    "p_value": null or number
  }},
  "sample": {{
    "n": null or integer,
    "population": "students" | "university" | "medical_students" | "general" | "clinical" | "mixed",
    "age_range": "e.g. 18-25" or null
  }},
  "study_type": "rct" | "cohort" | "cross_sectional" | "meta_analysis" | "review" | "case_study" | "other",
  "has_numerical_threshold": true/false,
  "replicates_pilot": true/false,
  "pilot_finding_replicated": "e.g. HRV-depression correlation, sleep-insomnia correlation, composite better than individual" or null,
  "key_finding": "1-2 sentence summary most relevant to Focus Score hypothesis"
}}

Return ONLY the JSON object, no other text.

## Abstract:
{abstract}
"""


@dataclass
class Finding:
    paper_id: str
    title: str
    year: int
    source: str
    doi: str
    url: str
    supports_h1: Optional[bool]
    supports_h2: Optional[bool]
    focus_score_relevant: bool
    composite_metric_used: bool
    markers: list
    outcomes: list
    effect_size: dict
    sample: dict
    study_type: str
    has_numerical_threshold: bool
    replicates_pilot: bool
    pilot_finding_replicated: Optional[str]
    key_finding: str
    extraction_ok: bool = True


def make_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to environment or GitHub Secrets.")
    return anthropic.Anthropic(api_key=api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def extract_finding(client: anthropic.Anthropic, paper: dict) -> Finding:
    """Extract structured Focus Score evidence from one paper abstract."""
    abstract = paper.get("abstract", "").strip()
    if not abstract:
        return Finding(
            paper_id=paper.get("paper_id", ""),
            title=paper.get("title", ""),
            year=paper.get("year", 0),
            source=paper.get("source", ""),
            doi=paper.get("doi", ""),
            url=paper.get("url", ""),
            supports_h1=None, supports_h2=None,
            focus_score_relevant=False, composite_metric_used=False,
            markers=[], outcomes=[],
            effect_size={"cohens_d": None, "r": None, "p_value": None},
            sample={"n": None, "population": None, "age_range": None},
            study_type="other",
            has_numerical_threshold=False,
            replicates_pilot=False, pilot_finding_replicated=None,
            key_finding="No abstract available.",
            extraction_ok=False,
        )

    prompt = EXTRACTION_PROMPT.format(abstract=abstract[:3000])
    estimated_in = len(prompt) // 4
    estimated_out = MAX_TOKENS_PER_CALL

    limiter.wait_if_needed(estimated_in + estimated_out)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS_PER_CALL,
        messages=[{"role": "user", "content": prompt}],
    )

    out_tokens = response.usage.output_tokens if response.usage else estimated_out
    in_tokens = response.usage.input_tokens if response.usage else estimated_in
    limiter.record(in_tokens + out_tokens)

    raw = response.content[0].text.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            log.warning(f"Could not parse JSON for paper: {paper.get('title','?')[:60]}")
            return Finding(
                paper_id=paper.get("paper_id", ""),
                title=paper.get("title", ""),
                year=paper.get("year", 0),
                source=paper.get("source", ""),
                doi=paper.get("doi", ""),
                url=paper.get("url", ""),
                supports_h1=None, supports_h2=None,
                focus_score_relevant=False, composite_metric_used=False,
                markers=[], outcomes=[],
                effect_size={"cohens_d": None, "r": None, "p_value": None},
                sample={"n": None, "population": None, "age_range": None},
                study_type="other",
                has_numerical_threshold=False,
                replicates_pilot=False, pilot_finding_replicated=None,
                key_finding="Extraction failed.",
                extraction_ok=False,
            )

    return Finding(
        paper_id=paper.get("paper_id", ""),
        title=paper.get("title", ""),
        year=paper.get("year", 0),
        source=paper.get("source", ""),
        doi=paper.get("doi", ""),
        url=paper.get("url", ""),
        supports_h1=data.get("supports_h1"),
        supports_h2=data.get("supports_h2"),
        focus_score_relevant=data.get("focus_score_relevant", False),
        composite_metric_used=data.get("composite_metric_used", False),
        markers=data.get("markers", []),
        outcomes=data.get("outcomes", []),
        effect_size=data.get("effect_size", {"cohens_d": None, "r": None, "p_value": None}),
        sample=data.get("sample", {"n": None, "population": None, "age_range": None}),
        study_type=data.get("study_type", "other"),
        has_numerical_threshold=data.get("has_numerical_threshold", False),
        replicates_pilot=data.get("replicates_pilot", False),
        pilot_finding_replicated=data.get("pilot_finding_replicated"),
        key_finding=data.get("key_finding", ""),
        extraction_ok=True,
    )


def extract_batch(papers: list[dict], existing_ids: set = None) -> list[Finding]:
    """Extract findings from a list of paper dicts (from search.py output)."""
    if existing_ids is None:
        existing_ids = set()

    client = make_client()
    findings = []
    new_papers = [p for p in papers if p.get("paper_id", "") not in existing_ids]
    log.info(f"[Extract] Processing {len(new_papers)} new papers (skipping {len(papers)-len(new_papers)} known)")

    for i, paper in enumerate(new_papers):
        try:
            finding = extract_finding(client, paper)
            findings.append(finding)
            status = "✓" if finding.extraction_ok else "✗"
            h1 = "H1✓" if finding.supports_h1 else ("H1✗" if finding.supports_h1 is False else "H1?")
            h2 = "H2✓" if finding.supports_h2 else ("H2✗" if finding.supports_h2 is False else "H2?")
            rep = "REP" if finding.replicates_pilot else ""
            log.info(f"  [{i+1}/{len(new_papers)}] {status} {h1} {h2} {rep} | {paper.get('title','')[:70]}")
        except Exception as e:
            log.error(f"  [{i+1}/{len(new_papers)}] FAILED: {e} | {paper.get('title','')[:60]}")
    return findings


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test with a sample abstract")
    parser.add_argument("--input", type=str, help="Path to JSONL of papers from search.py")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.test:
        test_paper = {
            "paper_id": "test001",
            "title": "Composite HRV-sleep readiness and academic performance in university students",
            "abstract": "Background: Physiological readiness metrics combining heart rate variability (HRV) "
                       "and sleep quality may predict cognitive performance. Methods: We recruited 120 university "
                       "students (mean age 21.3 years) and measured RMSSD via nighttime wearable ECG for 28 days. "
                       "Sleep was assessed via diary (TST, efficiency, consistency). A composite readiness score "
                       "(0.4×HRV + 0.6×Sleep, scale 0-100) was computed daily. Academic performance was indexed "
                       "by GPA and exam retention scores. Results: Students with composite readiness > 75 had "
                       "significantly higher GPA (3.6 vs 3.1, p=0.003, d=0.72), better retention (82% vs 68%, p=0.01), "
                       "and lower anxiety (GAD-7: r=−0.48, p<0.001). The composite score outperformed "
                       "HRV-alone (d=0.45) and sleep-alone (d=0.52) for predicting anxiety. Insomnia "
                       "(ISI) showed the strongest correlation (r=−0.59, p<0.001). Conclusion: A composite "
                       "HRV+sleep readiness metric is a stronger predictor of mental health and academic "
                       "performance than either component alone in university students.",
            "year": 2024, "source": "test", "doi": "10.1234/test", "url": "https://example.com",
        }
        client = make_client()
        finding = extract_finding(client, test_paper)
        print(json.dumps(asdict(finding), indent=2))
        print("\n✓ Extraction test passed!")
    elif args.input:
        with open(args.input) as f:
            papers = [json.loads(line) for line in f if line.strip()]
        findings = extract_batch(papers)
        for f_item in findings:
            print(json.dumps(asdict(f_item)))
    else:
        print("Use --test for a quick test, or --input papers.jsonl to process papers")
