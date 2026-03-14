"""
search.py — Multi-source academic paper crawler
Crawls PubMed, Semantic Scholar, CrossRef, bioRxiv, Europe PMC, Google Scholar
No API keys required (uses public endpoints + web crawling).
"""

import time
import random
import hashlib
import logging
import argparse
import json
import re
from typing import Optional
from urllib.parse import urlencode, quote_plus
from dataclasses import dataclass, asdict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("search")

# ─── USER AGENT ROTATION ──────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

REQUEST_DELAY = 0.15   # seconds between requests
PAGE_LIMIT    = 3      # pages per source per query
RESULTS_PER_PAGE = 20


@dataclass
class Paper:
    title: str
    abstract: str
    doi: str
    year: int
    authors: list
    source: str
    url: str
    paper_id: str = ""  # content hash dedup key

    def __post_init__(self):
        raw = (self.doi or self.title or "").lower().strip()
        self.paper_id = hashlib.md5(raw.encode()).hexdigest()[:12]


def get_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }


def safe_sleep(base: float = REQUEST_DELAY):
    time.sleep(base + random.uniform(0, 0.1))


# ─── RETRY DECORATOR ──────────────────────────────────────────────────────────
def make_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: PubMed (NCBI E-utilities — free, no key)
# ═══════════════════════════════════════════════════════════════════════════════
class PubMedCrawler:
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 60) -> list[Paper]:
        log.info(f"[PubMed] Searching: {query!r}")
        ids = self._search_ids(query, max_results)
        if not ids:
            return []
        papers = self._fetch_details(ids)
        log.info(f"[PubMed] Found {len(papers)} papers")
        return papers

    @make_retry()
    def _search_ids(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        with httpx.Client(timeout=20) as client:
            resp = client.get(f"{self.BASE}/esearch.fcgi", params=params, headers=get_headers())
            resp.raise_for_status()
            safe_sleep()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])

    @make_retry()
    def _fetch_details(self, ids: list[str]) -> list[Paper]:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
            "rettype": "abstract",
        }
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{self.BASE}/efetch.fcgi", params=params, headers=get_headers())
            # PubMed efetch in JSON is via summary
            params2 = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            }
            resp2 = client.get(f"{self.BASE}/esummary.fcgi", params=params2, headers=get_headers())
            resp2.raise_for_status()
            safe_sleep()
            data = resp2.json()
            papers = []
            result = data.get("result", {})
            for pmid in result.get("uids", []):
                art = result.get(pmid, {})
                title = art.get("title", "").strip()
                if not title:
                    continue
                authors = [a.get("name", "") for a in art.get("authors", [])]
                doi = ""
                for artid in art.get("articleids", []):
                    if artid.get("idtype") == "doi":
                        doi = artid.get("value", "")
                year_raw = art.get("pubdate", "2000")
                year = int(year_raw[:4]) if year_raw[:4].isdigit() else 2000
                # Fetch abstract separately
                abstract = self._fetch_abstract(pmid)
                papers.append(Paper(
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    year=year,
                    authors=authors,
                    source="pubmed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                ))
            return papers

    @make_retry()
    def _fetch_abstract(self, pmid: str) -> str:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(url, headers=get_headers())
            safe_sleep(0.1)
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            # Try structured abstract
            abstract_div = soup.find("div", {"class": "abstract-content"})
            if abstract_div:
                return abstract_div.get_text(separator=" ", strip=True)
            # Fallback
            meta = soup.find("meta", {"name": "citation_abstract"})
            if meta:
                return meta.get("content", "")
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: Semantic Scholar (public API — 100 req/5min without key)
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticScholarCrawler:
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,year,authors,externalIds,url"

    def search(self, query: str, max_results: int = 60) -> list[Paper]:
        log.info(f"[SemanticScholar] Searching: {query!r}")
        papers = []
        offset = 0
        limit = min(100, max_results)
        while len(papers) < max_results:
            batch = self._search_batch(query, offset, limit)
            if not batch:
                break
            papers.extend(batch)
            offset += limit
            if offset >= max_results:
                break
            safe_sleep(0.5)
        log.info(f"[SemanticScholar] Found {len(papers)} papers")
        return papers[:max_results]

    @make_retry()
    def _search_batch(self, query: str, offset: int, limit: int) -> list[Paper]:
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": self.FIELDS,
        }
        with httpx.Client(timeout=20) as client:
            resp = client.get(f"{self.BASE}/paper/search", params=params, headers=get_headers())
            if resp.status_code == 429:
                log.warning("[SemanticScholar] Rate limited, backing off 60s")
                time.sleep(60)
                return []
            resp.raise_for_status()
            safe_sleep()
            data = resp.json()
            papers = []
            for item in data.get("data", []):
                title = (item.get("title") or "").strip()
                abstract = (item.get("abstract") or "").strip()
                if not title or not abstract:
                    continue
                ext_ids = item.get("externalIds") or {}
                doi = ext_ids.get("DOI", "")
                year = item.get("year") or 2000
                authors = [a.get("name", "") for a in (item.get("authors") or [])]
                url = item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId','')}"
                papers.append(Paper(
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    year=year,
                    authors=authors,
                    source="semantic_scholar",
                    url=url,
                ))
            return papers


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: CrossRef (free REST API — no key)
# ═══════════════════════════════════════════════════════════════════════════════
class CrossRefCrawler:
    BASE = "https://api.crossref.org/works"

    def search(self, query: str, max_results: int = 40) -> list[Paper]:
        log.info(f"[CrossRef] Searching: {query!r}")
        papers = []
        offset = 0
        rows = min(100, max_results)
        while len(papers) < max_results:
            batch = self._search_batch(query, offset, rows)
            if not batch:
                break
            papers.extend(batch)
            offset += rows
            safe_sleep(0.2)
        log.info(f"[CrossRef] Found {len(papers)} papers")
        return papers[:max_results]

    @make_retry()
    def _search_batch(self, query: str, offset: int, rows: int) -> list[Paper]:
        params = {
            "query": query,
            "rows": rows,
            "offset": offset,
            "select": "title,abstract,DOI,published,author,URL",
            "mailto": "research@physio-research.ai",
        }
        with httpx.Client(timeout=20) as client:
            resp = client.get(self.BASE, params=params, headers=get_headers())
            resp.raise_for_status()
            safe_sleep()
            data = resp.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                title_list = item.get("title", [])
                title = title_list[0] if title_list else ""
                abstract = item.get("abstract", "")
                # Strip JATS XML tags from abstract
                abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                if not title or not abstract:
                    continue
                doi = item.get("DOI", "")
                pub = item.get("published", {}).get("date-parts", [[2000]])
                year = pub[0][0] if pub and pub[0] else 2000
                authors = []
                for a in item.get("author", []):
                    name = f"{a.get('given','')} {a.get('family','')}".strip()
                    authors.append(name)
                url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
                papers.append(Paper(
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    year=year,
                    authors=authors,
                    source="crossref",
                    url=url,
                ))
            return papers


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: Europe PMC (open access full text!)
# ═══════════════════════════════════════════════════════════════════════════════
class EuropePMCCrawler:
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, max_results: int = 40) -> list[Paper]:
        log.info(f"[EuropePMC] Searching: {query!r}")
        papers = []
        cursor_mark = "*"
        page_size = min(100, max_results)
        while len(papers) < max_results:
            batch, next_cursor = self._search_batch(query, cursor_mark, page_size)
            if not batch:
                break
            papers.extend(batch)
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor
            safe_sleep(0.2)
        log.info(f"[EuropePMC] Found {len(papers)} papers")
        return papers[:max_results]

    @make_retry()
    def _search_batch(self, query: str, cursor_mark: str, page_size: int):
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "sort": "RELEVANCE",
        }
        with httpx.Client(timeout=20) as client:
            resp = client.get(self.BASE, params=params, headers=get_headers())
            resp.raise_for_status()
            safe_sleep()
            data = resp.json()
            papers = []
            for item in data.get("resultList", {}).get("result", []):
                title = (item.get("title") or "").strip().rstrip(".")
                abstract = (item.get("abstractText") or "").strip()
                if not title or not abstract:
                    continue
                doi = item.get("doi", "")
                year = int(item.get("pubYear") or 2000)
                authors_raw = item.get("authorList", {}).get("author", [])
                authors = [f"{a.get('firstName','')} {a.get('lastName','')}".strip() for a in authors_raw]
                url = item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "") or f"https://europepmc.org/article/pmc/{item.get('pmcid','')}"
                papers.append(Paper(
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    year=year,
                    authors=authors,
                    source="europepmc",
                    url=url,
                ))
            next_cursor = data.get("nextCursorMark", "")
            return papers, next_cursor


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: bioRxiv / medRxiv (preprints — crawl API)
# ═══════════════════════════════════════════════════════════════════════════════
class BioRxivCrawler:
    BASE = "https://api.biorxiv.org/details"

    def search(self, query: str, max_results: int = 30) -> list[Paper]:
        """bioRxiv API doesn't support keyword search directly. We search via
        the Rxivist API which indexes biorxiv by keyword."""
        log.info(f"[bioRxiv] Searching: {query!r}")
        papers = []
        for server in ["biorxiv", "medrxiv"]:
            batch = self._rxivist_search(query, server, max_results // 2)
            papers.extend(batch)
            safe_sleep(0.3)
        log.info(f"[bioRxiv] Found {len(papers)} papers")
        return papers[:max_results]

    @make_retry()
    def _rxivist_search(self, query: str, server: str, max_results: int) -> list[Paper]:
        # Use Rxivist (indexes both bioRxiv and medRxiv)
        params = {
            "q": query,
            "page_size": min(max_results, 100),
            "page": 0,
            "server": server,
            "sort": "relevance",
            "category_filter": "",
        }
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.get("https://api.rxivist.org/v1/papers", params=params, headers=get_headers())
                resp.raise_for_status()
                safe_sleep()
                data = resp.json()
                papers = []
                for item in data.get("results", []):
                    title = (item.get("title") or "").strip()
                    abstract = (item.get("abstract") or "").strip()
                    if not title or not abstract:
                        continue
                    doi = item.get("doi", "")
                    year = 2020  # rxivist doesn't always provide year easily
                    biorxiv_url = item.get("biorxiv_url", "")
                    authors = [a.get("name", "") for a in item.get("authors", [])]
                    papers.append(Paper(
                        title=title,
                        abstract=abstract,
                        doi=doi,
                        year=year,
                        authors=authors,
                        source=server,
                        url=biorxiv_url,
                    ))
                return papers
        except Exception as e:
            log.warning(f"[bioRxiv/{server}] Rxivist failed: {e}. Trying direct crawl.")
            return self._direct_crawl(query, server, max_results)

    def _direct_crawl(self, query: str, server: str, max_results: int) -> list[Paper]:
        """Fallback: scrape biorxiv search page directly."""
        encoded = quote_plus(query)
        url = f"https://www.{server}.org/search/{encoded}"
        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url, headers=get_headers())
                safe_sleep(0.3)
                if resp.status_code != 200:
                    return []
                soup = BeautifulSoup(resp.text, "html.parser")
                papers = []
                for article in soup.select("li.search-result")[:max_results]:
                    title_tag = article.find("span", class_="highwire-cite-title")
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    abstract_tag = article.find("span", class_="highwire-cite-snippet")
                    abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""
                    link_tag = article.find("a", class_="highwire-cite-linked-title")
                    link = f"https://www.{server}.org" + link_tag["href"] if link_tag and link_tag.get("href") else ""
                    if title:
                        papers.append(Paper(
                            title=title, abstract=abstract, doi="",
                            year=2022, authors=[], source=server, url=link,
                        ))
                return papers
        except Exception as e:
            log.warning(f"[bioRxiv/{server}] Direct crawl failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 6: Google Scholar (web scrape — last resort, polite crawl)
# ═══════════════════════════════════════════════════════════════════════════════
class GoogleScholarCrawler:
    BASE = "https://scholar.google.com/scholar"

    def search(self, query: str, max_results: int = 20) -> list[Paper]:
        log.info(f"[GoogleScholar] Searching: {query!r}")
        papers = []
        for start in range(0, min(max_results, 40), 10):
            batch = self._search_page(query, start)
            papers.extend(batch)
            if not batch or len(papers) >= max_results:
                break
            safe_sleep(2.0 + random.uniform(0, 1))  # polite — scholar is sensitive
        log.info(f"[GoogleScholar] Found {len(papers)} papers")
        return papers[:max_results]

    @make_retry()
    def _search_page(self, query: str, start: int) -> list[Paper]:
        params = {"q": query, "start": start, "hl": "en", "as_sdt": "0,5"}
        headers = get_headers()
        headers["Referer"] = "https://scholar.google.com/"
        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(self.BASE, params=params, headers=headers)
                safe_sleep(0.5)
                if resp.status_code in (429, 503):
                    log.warning("[GoogleScholar] Blocked/rate limited — skipping")
                    return []
                if resp.status_code != 200:
                    return []
                soup = BeautifulSoup(resp.text, "html.parser")
                papers = []
                for div in soup.select(".gs_r.gs_or.gs_scl"):
                    title_tag = div.select_one(".gs_rt a")
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    url = title_tag["href"] if title_tag else ""
                    snippet_tag = div.select_one(".gs_rs")
                    abstract = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    meta_tag = div.select_one(".gs_a")
                    meta = meta_tag.get_text(strip=True) if meta_tag else ""
                    # Try to extract year from meta
                    year_match = re.search(r"\b(19|20)\d{2}\b", meta)
                    year = int(year_match.group()) if year_match else 2020
                    if title and abstract:
                        papers.append(Paper(
                            title=title, abstract=abstract, doi="",
                            year=year, authors=[], source="google_scholar", url=url,
                        ))
                return papers
        except Exception as e:
            log.warning(f"[GoogleScholar] Scrape failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SEARCH ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def dedup_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate by DOI first, then by paper_id (title hash)."""
    seen_dois = set()
    seen_ids = set()
    deduped = []
    for p in papers:
        key = p.doi.lower().strip() if p.doi else None
        if key and key in seen_dois:
            continue
        if p.paper_id in seen_ids:
            continue
        if key:
            seen_dois.add(key)
        seen_ids.add(p.paper_id)
        deduped.append(p)
    return deduped


def search_all(
    query: str,
    max_per_source: int = 50,
    sources: Optional[list[str]] = None,
    existing_ids: Optional[set] = None,
) -> list[Paper]:
    """
    Search all sources for papers matching query.
    Returns deduplicated list of new papers only.
    """
    if sources is None:
        sources = ["pubmed", "semantic_scholar", "crossref", "europepmc", "biorxiv", "google_scholar"]
    if existing_ids is None:
        existing_ids = set()

    crawlers = {
        "pubmed": PubMedCrawler(),
        "semantic_scholar": SemanticScholarCrawler(),
        "crossref": CrossRefCrawler(),
        "europepmc": EuropePMCCrawler(),
        "biorxiv": BioRxivCrawler(),
        "google_scholar": GoogleScholarCrawler(),
    }

    all_papers = []
    for source in sources:
        if source not in crawlers:
            log.warning(f"Unknown source: {source}")
            continue
        try:
            papers = crawlers[source].search(query, max_per_source)
            all_papers.extend(papers)
            log.info(f"  → {source}: {len(papers)} papers")
        except Exception as e:
            log.error(f"  ✗ {source} failed: {e}")
        safe_sleep(0.2)

    # Deduplicate and filter out already-seen papers
    deduped = dedup_papers(all_papers)
    new_papers = [p for p in deduped if p.paper_id not in existing_ids]
    log.info(f"Total: {len(all_papers)} raw → {len(deduped)} deduped → {len(new_papers)} new")
    return new_papers


# ═══════════════════════════════════════════════════════════════════════════════
# RELEVANCE PRE-SCORING — rank papers before LLM extraction
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword tiers for Focus Score relevance scoring
# Tier 1 (weight=3): exact Focus Score concepts
TIER1_KEYWORDS = [
    "hrv", "heart rate variability", "rmssd", "sdnn",
    "sleep quality", "sleep duration", "psqi", "insomnia",
    "focus score", "readiness score", "composite score", "composite metric",
    "physiological readiness", "recovery score",
]
# Tier 2 (weight=2): outcome measures from hypothesis
TIER2_KEYWORDS = [
    "phq-9", "phq9", "gad-7", "gad7", "isi", "insomnia severity",
    "depression", "anxiety", "mental health", "stress",
    "academic performance", "gpa", "exam", "test score", "marks",
    "cognition", "cognitive", "retention", "recall", "memory",
    "learning", "attention", "working memory",
]
# Tier 3 (weight=1): population and method
TIER3_KEYWORDS = [
    "student", "university", "college", "undergraduate",
    "wearable", "smartwatch", "oura", "whoop", "garmin",
    "biofeedback", "intervention", "rct", "randomized",
    "cortisol", "autonomic", "parasympathetic", "vagal",
    "circadian", "bedtime", "sleep efficiency", "aasm",
    "glucocorticoid", "prefrontal", "cohen",
    "effect size", "threshold", "biomarker",
]


def relevance_score(paper: 'Paper') -> float:
    """
    Compute a keyword-based relevance score for a paper (0.0 to 1.0).
    Looks at title + abstract text against Focus Score relevant keywords.
    """
    text = (paper.title + " " + paper.abstract).lower()
    score = 0.0
    max_possible = 0.0

    for kw in TIER1_KEYWORDS:
        max_possible += 3
        if kw in text:
            score += 3
    for kw in TIER2_KEYWORDS:
        max_possible += 2
        if kw in text:
            score += 2
    for kw in TIER3_KEYWORDS:
        max_possible += 1
        if kw in text:
            score += 1

    # Bonus: recency (post-2018 papers are more relevant)
    if paper.year >= 2020:
        score += 3
    elif paper.year >= 2015:
        score += 1

    # Bonus: has abstract (essential for extraction)
    if len(paper.abstract) > 200:
        score += 2

    max_possible += 5  # recency + abstract bonus ceiling
    return min(score / max_possible, 1.0) if max_possible > 0 else 0.0


def rank_papers(papers: list['Paper'], top_n: int = 10) -> list['Paper']:
    """
    Rank papers by relevance score and return the top N.
    Logs the scoring for visibility.
    """
    if not papers:
        return []

    scored = [(relevance_score(p), p) for p in papers]
    scored.sort(key=lambda x: x[0], reverse=True)

    log.info(f"[Ranking] Top {top_n} of {len(scored)} papers by relevance:")
    for i, (score, p) in enumerate(scored[:top_n]):
        log.info(f"  #{i+1} [{score:.2f}] {p.title[:80]}")

    if len(scored) > top_n:
        cutoff_score = scored[top_n - 1][0]
        dropped_score = scored[top_n][0] if len(scored) > top_n else 0
        log.info(f"  --- cutoff: {cutoff_score:.2f} | next dropped: {dropped_score:.2f} ---")

    return [p for _, p in scored[:top_n]]


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="HRV heart rate variability mental health students academic performance")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sources", nargs="+", default=None)
    args = parser.parse_args()

    results = search_all(args.query, max_per_source=args.limit, sources=args.sources)
    for p in results:
        print(json.dumps({
            "title": p.title,
            "year": p.year,
            "source": p.source,
            "doi": p.doi,
            "abstract_preview": p.abstract[:200],
        }, ensure_ascii=False))
    print(f"\nTotal new papers found: {len(results)}")
