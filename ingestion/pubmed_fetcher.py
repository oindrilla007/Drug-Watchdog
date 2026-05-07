"""
ingestion/pubmed_fetcher.py
────────────────────────────────────────────────────────────────────────────────
Fetches drug-interaction related abstracts from PubMed using the NCBI
E-utilities REST API (free, no login required — API key optional but
recommended for higher rate limits).

What it fetches:
  - Abstracts for drug-drug interaction queries
  - Abstracts for CYP450 enzyme-drug relationships
  - Abstracts for specific high-severity drug pairs from DrugBank

NCBI E-utilities docs:
  https://www.ncbi.nlm.nih.gov/books/NBK25497/

API key (optional — raises limit from 3 to 10 req/sec):
  https://www.ncbi.nlm.nih.gov/account/

Set your key in .env:
  NCBI_API_KEY=your_key_here

Output:
  data/raw/pubmed/          ← raw JSON responses (one file per query batch)
  data/processed/pubmed_abstracts.parquet  ← clean abstract table
  data/processed/chunks/pubmed_chunks.parquet  ← RAG-ready chunks

Usage:
  from ingestion.pubmed_fetcher import PubMedFetcher

  fetcher = PubMedFetcher(api_key="your_key")   # api_key optional
  fetcher.fetch_for_drug_pairs(drug_pairs)       # list of (drug_a, drug_b) tuples
  fetcher.fetch_cyp450_literature()              # CYP450 enzyme papers
  fetcher.save_all("data/processed/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── E-utilities base URLs ─────────────────────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Without API key: max 3 requests/sec. With key: max 10 requests/sec.
DELAY_NO_KEY  = 0.34   # seconds between requests
DELAY_WITH_KEY = 0.11

# ── Fetch settings ────────────────────────────────────────────────────────────
MAX_RESULTS_PER_QUERY = 50     # abstracts per drug pair query
MAX_CHUNK_CHARS       = 1_200  # RAG chunk size limit
RETRIES               = 3      # retry count on network errors

# ── CYP450 query templates ────────────────────────────────────────────────────
CYP_QUERIES = [
    "CYP3A4 drug interaction inhibitor substrate",
    "CYP2D6 drug interaction inhibitor substrate",
    "CYP2C9 drug interaction inhibitor substrate",
    "CYP2C19 drug interaction inhibitor substrate",
    "CYP1A2 drug interaction inhibitor substrate",
    "CYP2B6 drug interaction inhibitor substrate",
    "cytochrome P450 polypharmacy adverse drug reaction",
    "drug drug interaction pharmacokinetic mechanism review",
    "drug interaction elderly polypharmacy hospital",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PubMedAbstract:
    pmid:       str
    title:      str
    abstract:   str
    authors:    list[str]   = field(default_factory=list)
    journal:    str         = ""
    pub_year:   str         = ""
    mesh_terms: list[str]   = field(default_factory=list)
    query:      str         = ""     # the search query that found this paper
    source:     str         = "pubmed"


@dataclass
class TextChunk:
    chunk_id:     str
    source:       str
    pmid:         str
    drug_query:   str
    section:      str    # "title_abstract" or "abstract"
    text:         str
    char_count:   int


# ─────────────────────────────────────────────────────────────────────────────
# Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class PubMedFetcher:
    """
    Queries PubMed E-utilities API for drug interaction literature
    and saves abstracts as structured data + RAG-ready chunks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        raw_dir: str | Path = "data/raw/pubmed/",
    ) -> None:
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._delay  = DELAY_WITH_KEY if self.api_key else DELAY_NO_KEY

        self._abstracts: list[PubMedAbstract] = []
        self._seen_pmids: set[str] = set()    # deduplication across queries

        if self.api_key:
            logger.info("PubMedFetcher initialised with API key (10 req/sec)")
        else:
            logger.info("PubMedFetcher initialised without API key (3 req/sec) — "
                        "set NCBI_API_KEY in .env for faster fetching")

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_for_drug_pairs(
        self,
        drug_pairs: list[tuple[str, str]],
        max_pairs: Optional[int] = None,
    ) -> "PubMedFetcher":
        """
        For each (drug_a, drug_b) pair, search PubMed and fetch abstracts.
        Focuses on high-severity pairs — pass your DrugBank interaction pairs
        filtered to severity >= 2.

        drug_pairs: list of (drug_a_name, drug_b_name) tuples
        max_pairs:  cap for dev/test (None = all pairs)
        """
        pairs = drug_pairs[:max_pairs] if max_pairs else drug_pairs
        logger.info(f"Fetching PubMed abstracts for {len(pairs):,} drug pairs …")

        for i, (drug_a, drug_b) in enumerate(pairs, 1):
            query = f'"{drug_a}"[Title/Abstract] AND "{drug_b}"[Title/Abstract] AND (interaction OR toxicity OR adverse)'
            self._fetch_query(query, label=f"{drug_a}+{drug_b}")

            if i % 50 == 0:
                logger.info(f"  … {i}/{len(pairs)} pairs done | "
                            f"{len(self._abstracts):,} abstracts collected")

        logger.info(f"✅ Drug pair fetch complete — {len(self._abstracts):,} unique abstracts")
        return self

    def fetch_cyp450_literature(self) -> "PubMedFetcher":
        """Fetch general CYP450 and polypharmacy review papers."""
        logger.info(f"Fetching CYP450 / polypharmacy literature ({len(CYP_QUERIES)} queries) …")
        for query in CYP_QUERIES:
            self._fetch_query(query, label="cyp450_general")
        logger.info(f"✅ CYP450 fetch complete — total abstracts: {len(self._abstracts):,}")
        return self

    def fetch_custom(self, query: str, label: str = "custom") -> "PubMedFetcher":
        """Fetch abstracts for any custom PubMed query string."""
        self._fetch_query(query, label=label)
        return self

    def get_abstracts(self) -> list[PubMedAbstract]:
        return self._abstracts

    def get_abstracts_df(self) -> pd.DataFrame:
        rows = [
            {
                "pmid":       a.pmid,
                "title":      a.title,
                "abstract":   a.abstract,
                "authors":    "|".join(a.authors),
                "journal":    a.journal,
                "pub_year":   a.pub_year,
                "mesh_terms": "|".join(a.mesh_terms),
                "query":      a.query,
                "source":     a.source,
                "text_len":   len(a.abstract),
            }
            for a in self._abstracts
        ]
        return pd.DataFrame(rows)

    def get_chunks_df(self) -> pd.DataFrame:
        chunks = self._make_all_chunks()
        rows = [
            {
                "chunk_id":   c.chunk_id,
                "source":     c.source,
                "pmid":       c.pmid,
                "drug_query": c.drug_query,
                "section":    c.section,
                "text":       c.text,
                "char_count": c.char_count,
            }
            for c in chunks
        ]
        return pd.DataFrame(rows)

    def save_all(self, output_dir: str | Path = "data/processed/") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        abstracts_df = self.get_abstracts_df()
        abstracts_df.to_parquet(out / "pubmed_abstracts.parquet", index=False)
        logger.info(f"💾 Saved {len(abstracts_df):,} abstracts → {out / 'pubmed_abstracts.parquet'}")

        chunks_dir = out / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        chunks_df = self.get_chunks_df()
        chunks_df.to_parquet(chunks_dir / "pubmed_chunks.parquet", index=False)
        logger.info(f"💾 Saved {len(chunks_df):,} chunks → {chunks_dir / 'pubmed_chunks.parquet'}")

    # ── Private: query pipeline ───────────────────────────────────────────────

    def _fetch_query(self, query: str, label: str) -> None:
        """Run esearch → get PMIDs → efetch abstracts → parse → store."""
        pmids = self._esearch(query)
        if not pmids:
            return

        # Only fetch PMIDs we haven't seen yet
        new_pmids = [p for p in pmids if p not in self._seen_pmids]
        if not new_pmids:
            return

        abstracts = self._efetch(new_pmids, query=query)
        for ab in abstracts:
            if ab.pmid not in self._seen_pmids:
                self._abstracts.append(ab)
                self._seen_pmids.add(ab.pmid)

        # Save raw JSON to disk for reproducibility
        self._save_raw(label, pmids, abstracts)

    def _esearch(self, query: str) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        params: dict[str, str] = {
            "db":       "pubmed",
            "term":     query,
            "retmax":   str(MAX_RESULTS_PER_QUERY),
            "retmode":  "json",
            "sort":     "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{ESEARCH_URL}?{urlencode(params)}"
        data = self._get_json(url)
        if not data:
            return []

        pmids = data.get("esearchresult", {}).get("idlist", [])
        time.sleep(self._delay)
        return pmids

    def _efetch(self, pmids: list[str], query: str = "") -> list[PubMedAbstract]:
        """Fetch full records for a list of PMIDs and parse them."""
        params: dict[str, str] = {
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{EFETCH_URL}?{urlencode(params)}"
        xml_bytes = self._get_bytes(url)
        time.sleep(self._delay)

        if not xml_bytes:
            return []

        return self._parse_pubmed_xml(xml_bytes, query=query)

    # ── Private: XML parsing ──────────────────────────────────────────────────

    @staticmethod
    def _parse_pubmed_xml(xml_bytes: bytes, query: str = "") -> list[PubMedAbstract]:
        """
        Parse PubMed efetch XML response into PubMedAbstract objects.
        Uses ElementTree — no extra dependencies.
        """
        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return []

        abstracts = []

        for article in root.findall(".//PubmedArticle"):
            try:
                # PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text.strip() if pmid_elem is not None else ""
                if not pmid:
                    continue

                # Title
                title_elem = article.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

                # Abstract — may have multiple <AbstractText> with Label attributes
                abstract_parts = []
                for ab_elem in article.findall(".//AbstractText"):
                    label = ab_elem.get("Label", "")
                    text  = "".join(ab_elem.itertext()).strip()
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)

                if not abstract:
                    continue   # skip records with no abstract text

                # Authors
                authors = []
                for author in article.findall(".//Author"):
                    last  = author.findtext("LastName", "")
                    first = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {first}".strip())

                # Journal
                journal = article.findtext(".//Journal/Title", "")

                # Publication year
                pub_year = (
                    article.findtext(".//PubDate/Year") or
                    article.findtext(".//PubDate/MedlineDate", "")[:4]
                )

                # MeSH terms
                mesh_terms = [
                    d.findtext("DescriptorName", "")
                    for d in article.findall(".//MeshHeading")
                    if d.findtext("DescriptorName", "")
                ]

                abstracts.append(PubMedAbstract(
                    pmid       = pmid,
                    title      = title,
                    abstract   = abstract,
                    authors    = authors,
                    journal    = journal,
                    pub_year   = pub_year,
                    mesh_terms = mesh_terms,
                    query      = query,
                ))

            except Exception as exc:
                logger.debug(f"Skipped article due to parse error: {exc}")
                continue

        return abstracts

    # ── Private: chunking ─────────────────────────────────────────────────────

    def _make_all_chunks(self) -> list[TextChunk]:
        chunks = []
        for ab in self._abstracts:
            # Combine title + abstract as one text block for RAG
            full_text = f"{ab.title}. {ab.abstract}".strip()
            sub_texts = self._split_text(full_text)
            for sub in sub_texts:
                chunks.append(TextChunk(
                    chunk_id   = str(uuid.uuid4()),
                    source     = "pubmed",
                    pmid       = ab.pmid,
                    drug_query = ab.query,
                    section    = "title_abstract",
                    text       = sub,
                    char_count = len(sub),
                ))
        return chunks

    @staticmethod
    def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
        """Split at sentence boundaries with one-sentence overlap."""
        if len(text) <= max_chars:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > max_chars and current:
                chunks.append(" ".join(current))
                current = [current[-1], sent]
                current_len = len(current[-2]) + len(sent)
            else:
                current.append(sent)
                current_len += len(sent)

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ── Private: HTTP helpers ─────────────────────────────────────────────────

    def _get_json(self, url: str) -> Optional[dict]:
        raw = self._get_bytes(url)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return None

    def _get_bytes(self, url: str) -> Optional[bytes]:
        """GET request with retry logic."""
        for attempt in range(1, RETRIES + 1):
            try:
                req = Request(url, headers={"User-Agent": "DrugWatchdog/1.0 (research)"})
                with urlopen(req, timeout=30) as resp:
                    return resp.read()
            except HTTPError as e:
                if e.code == 429:   # rate limited
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt}/{RETRIES})")
                    time.sleep(wait)
                else:
                    logger.warning(f"HTTP {e.code} for URL: {url}")
                    return None
            except URLError as e:
                logger.warning(f"Network error (attempt {attempt}/{RETRIES}): {e}")
                time.sleep(1)
        return None

    def _save_raw(
        self,
        label: str,
        pmids: list[str],
        abstracts: list[PubMedAbstract],
    ) -> None:
        """Persist raw results to disk for reproducibility / offline re-runs."""
        safe_label = re.sub(r"[^\w\-]", "_", label)[:60]
        out_path = self.raw_dir / f"{safe_label}.json"
        payload = {
            "query_label": label,
            "pmids":       pmids,
            "abstracts": [
                {
                    "pmid":    a.pmid,
                    "title":   a.title,
                    "abstract": a.abstract,
                    "journal": a.journal,
                    "pub_year": a.pub_year,
                }
                for a in abstracts
            ],
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.debug(f"Could not save raw JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch PubMed abstracts for drug interactions")
    ap.add_argument("--api-key",  default="",              help="NCBI API key (optional)")
    ap.add_argument("--pairs",    default=None,            help="Parquet file with drug_pairs (drug_a_name, drug_b_name, severity)")
    ap.add_argument("--min-sev",  type=int, default=2,     help="Min severity to include a pair (default 2)")
    ap.add_argument("--max-pairs",type=int, default=200,   help="Max drug pairs to query (default 200)")
    ap.add_argument("--out",      default="data/processed/", help="Output directory")
    ap.add_argument("--raw-dir",  default="data/raw/pubmed/", help="Raw JSON output directory")
    ap.add_argument("--cyp450",   action="store_true",     help="Also fetch CYP450 literature")
    args = ap.parse_args()

    fetcher = PubMedFetcher(
        api_key = args.api_key or os.getenv("NCBI_API_KEY", ""),
        raw_dir = args.raw_dir,
    )

    # Load drug pairs from DrugBank output if provided
    if args.pairs:
        pairs_df = pd.read_parquet(args.pairs)
        # Filter to high-severity pairs only
        high_sev = pairs_df[pairs_df["severity"] >= args.min_sev]
        pair_list = list(zip(high_sev["drug_a_name"], high_sev["drug_b_name"]))
        logger.info(f"Loaded {len(pair_list):,} pairs with severity >= {args.min_sev}")
        fetcher.fetch_for_drug_pairs(pair_list, max_pairs=args.max_pairs)
    else:
        # Demo mode — fetch for a handful of well-known pairs
        demo_pairs = [
            ("warfarin", "aspirin"),
            ("warfarin", "ibuprofen"),
            ("simvastatin", "amiodarone"),
            ("clopidogrel", "omeprazole"),
            ("digoxin", "amiodarone"),
            ("metformin", "contrast media"),
            ("lithium", "ibuprofen"),
            ("ssri", "tramadol"),
        ]
        logger.info("No --pairs file provided — running demo pairs")
        fetcher.fetch_for_drug_pairs(demo_pairs)

    if args.cyp450:
        fetcher.fetch_cyp450_literature()

    fetcher.save_all(args.out)

    # Sanity report
    abstracts_df = fetcher.get_abstracts_df()
    chunks_df    = fetcher.get_chunks_df()

    print(f"\n── Abstracts collected: {len(abstracts_df):,} ──────────────────")
    print(abstracts_df[["pmid", "title", "pub_year", "journal"]].head(8).to_string(index=False))

    print(f"\n── Chunks generated: {len(chunks_df):,} ──────────────────────")
    print(f"Average chunk length: {chunks_df['char_count'].mean():.0f} chars")

    print(f"\n── Year distribution ──────────────────────────────────")
    print(abstracts_df["pub_year"].value_counts().head(10))