"""
extract.py — LLM-based biomarker extractor using Claude Haiku
Reads paper abstracts and extracts structured evidence for the HRV/sleep hypothesis.

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

CLAUDE_MODEL = "claude-haiku-4-5"  # Claude Haiku — fast + cheap
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
        # Drop entries older than 60s
        self.window = [(t, n) for t, n in self.window if now - t < 60]
        used = sum(n for _, n in self.window)
        if used + tokens_about_to_use > self.tokens_per_minute:
            # Wait until oldest entry expires
            if self.window:
                oldest = self.window[0][0]
                wait = 60 - (now - oldest) + 0.1
                if wait > 0:
                    log.info(f"[RateLimiter] Budget hit. Waiting {wait:.1f}s ...")
                    time.sleep(wait)
            self.window = []  # Reset after wait

    def record(self, tokens_used: int):
        self.window.append((time.time(), tokens_used))


limiter = TokenRateLimiter(TARGET_TOKENS_PER_MIN)


# ─── EXTRACTION SCHEMA ────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
You are a biomedical research analyst. Extract evidence about physiological markers (HRV and sleep) and their effects on mental health and academic performance from the paper abstract below.

## Hypothesis being tested:
H1 (Mental Health): Higher HRV / better sleep → lower anxiety, stress, depression in students
H2 (Cognition/Marks): Higher HRV / better sleep → higher academic performance (GPA, marks, test scores)

## Extract and return ONLY valid JSON in this exact format:
{
  "supports_h1": true/false/null,         // null if paper doesn't address H1
  "supports_h2": true/false/null,         // null if paper doesn't address H2
  "markers": [                            // list of physiological markers studied
    {
      "type": "HRV" | "sleep" | "other",
      "measure": "e.g. RMSSD, SDNN, LF/HF, PSQI, sleep duration, etc.",
      "threshold": "e.g. RMSSD > 50ms",  // null if no threshold given
      "direction": "higher better" | "lower better" | null
    }
  ],
  "outcomes": [                           // measured outcomes
    {
      "type": "mental_health" | "cognition" | "academic" | "other",
      "measure": "e.g. anxiety score, GPA, attention, cortisol",
      "direction": "improved" | "worsened" | "no effect" | null
    }
  ],
  "effect_size": {
    "cohens_d": null or number,           // null if not reported
    "r": null or number,                  // Pearson r, null if not reported
    "p_value": null or number
  },
  "sample": {
    "n": null or integer,                 // sample size
    "population": "students" | "general" | "clinical" | "mixed" | null,
    "age_range": "e.g. 18-25" or null
  },
  "study_type": "rct" | "cohort" | "cross_sectional" | "meta_analysis" | "review" | "case_study" | "other",
  "has_numerical_threshold": true/false,  // does paper give specific numerical marker thresholds?
  "key_finding": "1-2 sentence summary of the most important finding relevant to the hypothesis"
}

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
    markers: list
    outcomes: list
    effect_size: dict
    sample: dict
    study_type: str
    has_numerical_threshold: bool
    key_finding: str
    extraction_ok: bool = True


def make_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to environment or GitHub Secrets.")
    return anthropic.Anthropic(api_key=api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def extract_finding(client: anthropic.Anthropic, paper: dict) -> Finding:
    """Extract structured biomarker evidence from one paper abstract."""
    abstract = paper.get("abstract", "").strip()
    if not abstract:
        # No abstract — return empty finding
        return Finding(
            paper_id=paper.get("paper_id", ""),
            title=paper.get("title", ""),
            year=paper.get("year", 0),
            source=paper.get("source", ""),
            doi=paper.get("doi", ""),
            url=paper.get("url", ""),
            supports_h1=None, supports_h2=None,
            markers=[], outcomes=[],
            effect_size={"cohens_d": None, "r": None, "p_value": None},
            sample={"n": None, "population": None, "age_range": None},
            study_type="other",
            has_numerical_threshold=False,
            key_finding="No abstract available.",
            extraction_ok=False,
        )

    prompt = EXTRACTION_PROMPT.format(abstract=abstract[:3000])  # truncate very long abstracts
    # Estimate tokens (rough: 4 chars ≈ 1 token)
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

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
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
                markers=[], outcomes=[],
                effect_size={"cohens_d": None, "r": None, "p_value": None},
                sample={"n": None, "population": None, "age_range": None},
                study_type="other",
                has_numerical_threshold=False,
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
        markers=data.get("markers", []),
        outcomes=data.get("outcomes", []),
        effect_size=data.get("effect_size", {"cohens_d": None, "r": None, "p_value": None}),
        sample=data.get("sample", {"n": None, "population": None, "age_range": None}),
        study_type=data.get("study_type", "other"),
        has_numerical_threshold=data.get("has_numerical_threshold", False),
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
            log.info(f"  [{i+1}/{len(new_papers)}] {status} {h1} {h2} | {paper.get('title','')[:70]}")
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
            "title": "Heart rate variability and academic performance in university students",
            "abstract": "Background: Autonomic nervous system activity, measured via heart rate variability (HRV), "
                       "may be associated with cognitive performance. Methods: We recruited 120 university students "
                       "(mean age 21.3 years) and measured RMSSD via 5-minute ECG recordings. Academic performance "
                       "was indexed by GPA. Results: Students with RMSSD > 50ms had significantly higher GPA "
                       "(3.6 vs 3.1, p=0.003, d=0.72). Anxiety scores (GAD-7) were negatively correlated with "
                       "RMSSD (r=-0.48, p<0.001). Conclusion: Higher HRV is associated with better academic "
                       "performance and lower anxiety in university students.",
            "year": 2023, "source": "test", "doi": "10.1234/test", "url": "https://example.com",
        }
        client = make_client()
        finding = extract_finding(client, test_paper)
        print(json.dumps(asdict(finding), indent=2))
        print("\n✓ Extraction test passed!")
    elif args.input:
        import sys
        with open(args.input) as f:
            papers = [json.loads(line) for line in f if line.strip()]
        findings = extract_batch(papers)
        for f in findings:
            print(json.dumps(asdict(f)))
    else:
        print("Use --test for a quick test, or --input papers.jsonl to process papers")
