"""
ingestion/rxnorm_mapper.py
────────────────────────────────────────────────────────────────────────────────
Maps all drug names across every data source (DrugBank, DailyMed, FAERS,
PubMed) to canonical RxNorm CUI identifiers using the NIH NLM RxNorm REST API.

Why this matters:
  - DrugBank calls it "Acetaminophen"
  - FAERS calls it "TYLENOL" or "PARACETAMOL"
  - DailyMed calls it "acetaminophen 500 MG Oral Tablet"
  Without normalisation, the same drug looks like 3 different drugs to the ML
  model and RAG pipeline. RxNorm is the FDA-standard canonical drug vocabulary.

RxNorm REST API (free, no key required):
  https://rxnav.nlm.nih.gov/REST/

What it produces:
  data/processed/rxnorm_mapping.json   ← {raw_name: {cui, name, synonyms}}
  data/processed/rxnorm_mapping.parquet

Performance modes:
  Default        : 12 concurrent workers, all fallback strategies enabled
  --exact-only   : Skip spelling/approximate fallbacks (2x fewer API calls on misses)
  --no-synonyms  : Skip /allrelated fetch (halves API calls for resolved names)
  --limit N      : Only process the first N unique normalised names (for testing)
  --source X     : Only load one dataset — faers | drugbank | dailymed

Usage:
  from ingestion.rxnorm_mapper import RxNormMapper

  mapper = RxNormMapper()
  mapper.map_names(["warfarin", "TYLENOL", "paracetamol", "aspirin 81mg"])
  mapping = mapper.get_mapping()        # dict: raw_name → RxNormEntry
  mapper.save("data/processed/")

  # After mapping, normalise a DataFrame column:
  df["canonical_name"] = mapper.normalise_series(df["drugname"])

Approximate throughput (68k unique normalised names):
  Sequential (original)    :  ~10–14 hours
  12 workers, all fallbacks:  ~1–1.5 hours
  12 workers, exact-only   :  ~30–45 min
  Warm cache rerun          :  ~5–10 min
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── RxNorm API base ───────────────────────────────────────────────────────────
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# ── Concurrency ───────────────────────────────────────────────────────────────
# RxNorm asks for max 20 req/sec total. With 12 workers each sleeping 0.06s
# between their own calls that stays safely under the limit.
DEFAULT_WORKERS = 12
REQUEST_DELAY   = 0.06    # per-thread delay in seconds
RETRIES         = 3

# ── Approximate search fallback settings ─────────────────────────────────────
APPROX_MIN_SCORE = 80

# ── Dosage / form noise patterns to strip before lookup ──────────────────────
DOSE_PATTERN = re.compile(
    r"""
    \b(
        \d+\s*(?:mg|mcg|g|ml|%|units?|iu|meq|mmol)   # numeric dose
        | oral | tablet | capsule | injection | solution
        | extended.release | er | xr | sr | cr | dr
        | hcl | hydrochloride | sodium | potassium
        | film.coated | delayed.release | modified.release
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RxNormEntry:
    raw_name:         str
    cui:              Optional[str]
    canonical_name:   Optional[str]
    synonyms:         list[str]      = field(default_factory=list)
    tty:              Optional[str]  = None
    match_method:     str            = ""
    normalised_input: str            = ""


# ─────────────────────────────────────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────────────────────────────────────

class RxNormMapper:
    """
    Resolves arbitrary drug name strings to RxNorm CUIs.

    Key speed improvements over v1:
    ─────────────────────────────
    1. ThreadPoolExecutor — concurrent HTTP requests instead of sequential.
    2. exact_only mode    — skips spelling + approximate fallback API calls.
    3. fetch_synonyms     — can be disabled to halve calls for resolved names.
    4. Thread-safe cache  — lock around all cache reads/writes.
    5. Periodic checkpointing — progress saved every 1 000 resolved names so
       interrupted runs resume from where they left off.
    """

    def __init__(
        self,
        cache_path:    str | Path = "data/processed/rxnorm_mapping.json",
        workers:       int        = DEFAULT_WORKERS,
        exact_only:    bool       = False,
        fetch_synonyms: bool      = True,
    ) -> None:
        self.cache_path     = Path(cache_path)
        self.workers        = workers
        self.exact_only     = exact_only
        self.fetch_synonyms = fetch_synonyms

        self._cache: dict[str, dict]         = self._load_cache()
        self._entries: dict[str, RxNormEntry] = {}
        self._cui_details_cache: dict[str, tuple] = {}

        # Locks for shared state accessed from worker threads
        self._cache_lock      = threading.Lock()
        self._entries_lock    = threading.Lock()
        self._cui_lock        = threading.Lock()
        self._counter_lock    = threading.Lock()
        self._counters        = {"hit": 0, "miss": 0, "fail": 0, "done": 0}
        # Dedicated checkpoint tracker — prevents multiple threads all seeing
        # the same modulo==0 value and racing to write the cache simultaneously.
        self._last_checkpoint = 0
        self._checkpoint_lock = threading.Lock()

        logger.info(
            f"RxNormMapper initialised — {len(self._cache):,} cached entries | "
            f"workers={workers} | exact_only={exact_only} | fetch_synonyms={fetch_synonyms}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def map_names(
        self,
        names:          list[str],
        limit:          Optional[int] = None,
        batch_log_every: int          = 500,
    ) -> "RxNormMapper":
        """
        Resolve a list of raw drug name strings to RxNorm CUIs.

        Args:
            names:           Raw drug name strings (may contain duplicates).
            limit:           If set, only process the first N unique normalised
                             keys — useful for smoke-testing a large dataset.
            batch_log_every: Log progress every N completions.
        """
        unique_names = list(dict.fromkeys(names))

        # Group raw names by their normalised key so we do one API call per
        # unique normalised form, then fan the result out to all raw variants.
        normalised_groups: dict[str, list[str]] = {}
        representative_raw: dict[str, str]      = {}
        for name in unique_names:
            key = self._preprocess(name).lower()
            normalised_groups.setdefault(key, []).append(name)
            representative_raw.setdefault(key, name)

        all_keys = list(normalised_groups.keys())

        # ── Pre-run deduplication report ─────────────────────────────────────
        logger.info(
            f"Input: {len(names):,} total | {len(unique_names):,} unique raw | "
            f"{len(all_keys):,} unique normalised"
        )

        # ── Apply limit ───────────────────────────────────────────────────────
        if limit and limit < len(all_keys):
            logger.info(f"--limit {limit:,} applied — truncating from {len(all_keys):,} keys")
            all_keys = all_keys[:limit]

        # ── Separate cache hits from API work ─────────────────────────────────
        to_fetch: list[tuple[str, str]] = []   # (cache_key, representative_raw_name)
        for key in all_keys:
            if key in self._cache:
                raw_names   = normalised_groups[key]
                cached_dict = self._cache[key]
                for raw in raw_names:
                    entry            = RxNormEntry(**{k: v for k, v in cached_dict.items() if k != "raw_name"}, raw_name=raw)
                    entry.match_method = "cache"
                    with self._entries_lock:
                        self._entries[raw] = entry
                with self._counter_lock:
                    self._counters["hit"] += 1
                    self._counters["done"] += 1
            else:
                to_fetch.append((key, representative_raw[key]))

        logger.info(
            f"Cache hits: {self._counters['hit']:,} | "
            f"API lookups needed: {len(to_fetch):,}"
        )

        if not to_fetch:
            logger.info("✅ All names served from cache — no API calls needed")
            return self

        total_api = len(to_fetch)

        # ── Concurrent API resolution ─────────────────────────────────────────
        def _worker(args: tuple[str, str]) -> None:
            cache_key, raw_name = args
            normalised = self._preprocess(raw_name)
            entry      = self._resolve(raw_name, normalised)

            # Fan result out to all raw variants with this normalised key
            raw_names = normalised_groups[cache_key]
            with self._entries_lock:
                for raw in raw_names:
                    e          = deepcopy(entry)
                    e.raw_name = raw
                    self._entries[raw] = e

            # Write to cache
            cache_record = {
                "raw_name":         entry.raw_name,
                "cui":              entry.cui,
                "canonical_name":   entry.canonical_name,
                "synonyms":         entry.synonyms,
                "tty":              entry.tty,
                "match_method":     entry.match_method,
                "normalised_input": entry.normalised_input,
            }
            with self._cache_lock:
                self._cache[cache_key] = cache_record

            # Update counters
            with self._counter_lock:
                if entry.cui:
                    self._counters["miss"] += 1
                else:
                    self._counters["fail"] += 1
                self._counters["done"] += 1
                done = self._counters["done"]

            # Progress log
            if done % batch_log_every == 0:
                with self._counter_lock:
                    h, m, f, d = (
                        self._counters["hit"],
                        self._counters["miss"],
                        self._counters["fail"],
                        self._counters["done"],
                    )
                api_done = d - h
                pct      = api_done / total_api * 100 if total_api else 100
                logger.info(
                    f"  … {api_done}/{total_api} API ({pct:.1f}%) | "
                    f"total done: {d:,} | resolved: {m:,} | failed: {f:,}"
                )

            # Periodic checkpoint every 1 000 new API results.
            # _checkpoint_lock ensures exactly one thread fires the save even
            # if multiple threads cross the same modulo boundary simultaneously.
            with self._checkpoint_lock:
                api_done = self._counters["miss"] + self._counters["fail"]
                due      = (api_done // 1000) * 1000
                if due > self._last_checkpoint and due > 0:
                    self._last_checkpoint = due
                    do_checkpoint = True
                else:
                    do_checkpoint = False
            if do_checkpoint:
                self._save_cache()

        logger.info(f"Starting {self.workers}-worker concurrent resolution …")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_worker, item): item for item in to_fetch}
            # Drain futures — exceptions are re-raised here if any worker crashed
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    logger.warning(f"Worker error: {exc}")

        elapsed = time.time() - start
        self._save_cache()

        with self._counter_lock:
            h, m, f, d = (
                self._counters["hit"],
                self._counters["miss"],
                self._counters["fail"],
                self._counters["done"],
            )

        resolved_total = h + m
        coverage = resolved_total / max(len(unique_names), 1) * 100
        rate     = total_api / elapsed if elapsed > 0 else 0

        logger.info(
            f"✅ RxNorm mapping complete in {elapsed:.1f}s ({rate:.1f} lookups/sec) — "
            f"cache hits: {h:,} | newly resolved: {m:,} | unresolved: {f:,} | "
            f"coverage: {coverage:.1f}%"
        )
        return self

    def map_from_parquets(
        self,
        drugbank_path:  Optional[str | Path] = None,
        faers_path:     Optional[str | Path] = None,
        dailymed_path:  Optional[str | Path] = None,
        source_filter:  Optional[str]        = None,   # "faers" | "drugbank" | "dailymed"
        limit:          Optional[int]        = None,
        include_drugbank_synonyms: bool      = False,
    ) -> "RxNormMapper":
        """
        Load drug names from processed Parquet files and map them.

        Source loading order is intentional:
          FAERS → DailyMed → DrugBank primary names → DrugBank synonyms (optional)

        FAERS and DailyMed names are clinical drug names with high RxNorm exact-match
        rates (~90%+). DrugBank synonyms are often IUPAC/chemical strings that mostly
        fail exact lookup — loading them last means the cache is already warm for the
        names that matter, and you can kill the run early without losing coverage on
        the important sources.

        Args:
            source_filter:             If set, only load the named source.
            limit:                     Cap unique normalised lookups (smoke-test mode).
            include_drugbank_synonyms: Whether to explode DrugBank synonym strings
                                       into individual lookups (default False —
                                       chemical synonyms have poor RxNorm coverage
                                       and inflate the queue by ~60k names).
        """
        all_names: list[str] = []

        def _should_load(src: str) -> bool:
            return source_filter is None or source_filter.lower() == src

        # ── 1. FAERS first — highest clinical name match rate ─────────────────
        if _should_load("faers") and faers_path and Path(faers_path).exists():
            df    = pd.read_parquet(faers_path)
            before = len(all_names)
            if "drugname" in df.columns:
                all_names.extend(df["drugname"].dropna().unique().tolist())
            logger.info(f"  FAERS: {len(all_names) - before:,} unique drug names loaded")

        # ── 2. DailyMed second — structured label names, also high match rate ─
        if _should_load("dailymed") and dailymed_path and Path(dailymed_path).exists():
            df    = pd.read_parquet(dailymed_path)
            before = len(all_names)
            if "drug_name" in df.columns:
                all_names.extend(df["drug_name"].dropna().unique().tolist())
            logger.info(f"  DailyMed: {len(all_names) - before:,} drug names loaded")

        # ── 3. DrugBank primary names third ───────────────────────────────────
        if _should_load("drugbank") and drugbank_path and Path(drugbank_path).exists():
            df = pd.read_parquet(drugbank_path)
            before = len(all_names)
            if "name" in df.columns:
                all_names.extend(df["name"].dropna().tolist())
            logger.info(f"  DrugBank primary names: {len(all_names) - before:,} loaded")

            # ── 4. DrugBank synonyms last (opt-in) — often IUPAC/chemical names
            #       with ~30-40% RxNorm exact-match rate vs ~90% for clinical names
            if include_drugbank_synonyms and "synonyms" in df.columns:
                before = len(all_names)
                for syn_str in df["synonyms"].dropna():
                    all_names.extend(syn_str.split("|"))
                logger.info(f"  DrugBank synonyms: {len(all_names) - before:,} loaded (use --include-db-synonyms to skip this warning)")

        return self.map_names(all_names, limit=limit)

    def get_mapping(self) -> dict[str, RxNormEntry]:
        return self._entries

    def get_mapping_df(self) -> pd.DataFrame:
        rows = [
            {
                "raw_name":         e.raw_name,
                "cui":              e.cui,
                "canonical_name":   e.canonical_name,
                "synonyms":         "|".join(e.synonyms),
                "tty":              e.tty,
                "match_method":     e.match_method,
                "normalised_input": e.normalised_input,
                "resolved":         e.cui is not None,
            }
            for e in self._entries.values()
        ]
        return pd.DataFrame(rows)

    def normalise_series(self, series: pd.Series) -> pd.Series:
        """Map a Series of raw drug names → canonical RxNorm names."""
        index = {k.lower(): v for k, v in {
            e.raw_name: e.canonical_name
            for e in self._entries.values()
            if e.canonical_name
        }.items()}

        def _lookup(name: str) -> str:
            return index.get(str(name).lower(), str(name).upper())

        return series.apply(_lookup)

    def cui_for(self, name: str) -> Optional[str]:
        entry = self._entries.get(name)
        return entry.cui if entry else None

    def preview_dedup(self, names: list[str]) -> None:
        """
        Print deduplication stats without running any API calls.
        Useful for estimating actual lookup count before a long run.
        """
        unique_raw  = len(set(names))
        unique_norm = len({self._preprocess(n).lower() for n in names})
        cached      = sum(1 for n in {self._preprocess(n).lower() for n in names} if n in self._cache)
        needed      = unique_norm - cached
        print(f"── Dedup preview ─────────────────────────────────────")
        print(f"Total input names   : {len(names):>10,}")
        print(f"Unique raw names    : {unique_raw:>10,}")
        print(f"Unique normalised   : {unique_norm:>10,}")
        print(f"Already cached      : {cached:>10,}")
        print(f"API calls needed    : {needed:>10,}")
        est_secs = needed / (DEFAULT_WORKERS * (1 / REQUEST_DELAY) * 0.6)
        print(f"Est. time (approx)  : {est_secs/60:.0f}–{est_secs/60*1.4:.0f} min")

    def save(self, output_dir: str | Path = "data/processed/") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "rxnorm_mapping.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved {len(self._cache):,} entries → {json_path}")

        df          = self.get_mapping_df()
        parquet_path = out / "rxnorm_mapping.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"💾 Saved {len(df):,} rows → {parquet_path}")

    # ── Private: resolution logic ─────────────────────────────────────────────

    def _resolve(self, raw_name: str, normalised: str) -> RxNormEntry:
        entry = RxNormEntry(
            raw_name         = raw_name,
            cui              = None,
            canonical_name   = None,
            normalised_input = normalised,
        )

        # Strategy 1: exact match
        cui = self._exact_lookup(normalised)
        if cui:
            entry.cui          = cui
            entry.match_method = "exact"
            entry.canonical_name, entry.synonyms, entry.tty = self._fetch_details(cui)
            return entry

        if self.exact_only:
            entry.match_method = "failed"
            return entry

        # Strategy 2: spelling suggestions
        suggestion = self._spelling_suggestion(normalised)
        if suggestion and suggestion.lower() != normalised.lower():
            cui = self._exact_lookup(suggestion)
            if cui:
                entry.cui          = cui
                entry.match_method = "spelling"
                entry.canonical_name, entry.synonyms, entry.tty = self._fetch_details(cui)
                return entry

        # Strategy 3: approximate match
        cui, score = self._approximate_lookup(normalised)
        if cui and score >= APPROX_MIN_SCORE:
            entry.cui          = cui
            entry.match_method = f"approx(score={score})"
            entry.canonical_name, entry.synonyms, entry.tty = self._fetch_details(cui)
            return entry

        entry.match_method = "failed"
        logger.debug(f"  ✗ Could not resolve: '{raw_name}'")
        return entry

    def _exact_lookup(self, name: str) -> Optional[str]:
        url  = f"{RXNORM_BASE}/rxcui.json?name={quote(name)}&search=1"
        data = self._get_json(url)
        if not data:
            return None
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    def _spelling_suggestion(self, name: str) -> Optional[str]:
        url  = f"{RXNORM_BASE}/spellingsuggestions.json?name={quote(name)}"
        data = self._get_json(url)
        if not data:
            return None
        suggestions = (
            data.get("suggestionGroup", {})
                .get("suggestionList", {})
                .get("suggestion", [])
        )
        return suggestions[0] if suggestions else None

    def _approximate_lookup(self, name: str) -> tuple[Optional[str], float]:
        params = urlencode({"term": name, "maxEntries": 1})
        url    = f"{RXNORM_BASE}/approximateTerm.json?{params}"
        data   = self._get_json(url)
        if not data:
            return None, 0.0
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None, 0.0
        top = candidates[0]
        try:
            score = float(top.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        return top.get("rxcui"), score

    def _fetch_details(
        self, cui: str
    ) -> tuple[Optional[str], list[str], Optional[str]]:
        with self._cui_lock:
            cached = self._cui_details_cache.get(cui)
        if cached is not None:
            return cached

        # Properties endpoint (always fetched — gives us canonical name + tty)
        url   = f"{RXNORM_BASE}/rxcui/{cui}/properties.json"
        data  = self._get_json(url)
        props = data.get("properties", {}) if data else {}
        canonical = props.get("name")
        tty       = props.get("tty")

        # Synonyms endpoint (optional — skip if fetch_synonyms=False)
        synonyms: list[str] = []
        if self.fetch_synonyms:
            url2  = f"{RXNORM_BASE}/rxcui/{cui}/allrelated.json"
            data2 = self._get_json(url2)
            if data2:
                for cg in data2.get("allRelatedGroup", {}).get("conceptGroup", []):
                    for prop in cg.get("conceptProperties", []):
                        syn = prop.get("name", "")
                        if syn and syn != canonical:
                            synonyms.append(syn)

        result = (canonical, list(set(synonyms))[:20], tty)
        with self._cui_lock:
            self._cui_details_cache[cui] = result
        return result

    # ── Private: HTTP ─────────────────────────────────────────────────────────

    def _get_json(self, url: str) -> Optional[dict]:
        for attempt in range(1, RETRIES + 1):
            try:
                req = Request(
                    url,
                    headers={
                        "User-Agent": "DrugWatchdog/1.0 (research)",
                        "Accept":     "application/json",
                    },
                )
                with urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read())
                time.sleep(REQUEST_DELAY)
                return data
            except HTTPError as e:
                if e.code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"RxNorm rate limit — waiting {wait}s (thread {threading.current_thread().name})")
                    time.sleep(wait)
                else:
                    logger.debug(f"HTTP {e.code}: {url}")
                    return None
            except (URLError, json.JSONDecodeError, Exception) as exc:
                logger.debug(f"Request error (attempt {attempt}): {exc}")
                time.sleep(1)
        return None

    # ── Private: preprocessing ────────────────────────────────────────────────

    @staticmethod
    def _preprocess(name: str) -> str:
        text = str(name).strip()
        text = re.sub(r"\(.*?\)", "", text)
        text = DOSE_PATTERN.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text if len(text) >= 2 else str(name).strip().lower()

    # ── Private: cache I/O ────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Could not load cache from {self.cache_path} — starting fresh")
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_lock:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Map drug names to RxNorm CUIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--drugbank",      default="data/processed/drugbank_drugs.parquet")
    ap.add_argument("--faers",         default="data/processed/faers_drug_events.parquet")
    ap.add_argument("--dailymed",      default="data/processed/dailymed_labels.parquet")
    ap.add_argument("--out",           default="data/processed/")
    ap.add_argument("--cache",         default="data/processed/rxnorm_mapping.json")
    ap.add_argument("--workers",       type=int,  default=DEFAULT_WORKERS,
                    help="Concurrent HTTP workers (default 12; raise to 20 if you see no 429s)")
    ap.add_argument("--exact-only",    action="store_true",
                    help="Skip spelling/approximate fallbacks — ~2x faster, lower coverage")
    ap.add_argument("--no-synonyms",   action="store_true",
                    help="Skip /allrelated synonym fetch — halves API calls for resolved names")
    ap.add_argument("--limit",         type=int,  default=None,
                    help="Only process first N unique normalised names (smoke-test mode)")
    ap.add_argument("--source",              choices=["faers", "drugbank", "dailymed"], default=None,
                    help="Only load one data source")
    ap.add_argument("--include-db-synonyms", action="store_true",
                    help="Also resolve DrugBank synonym strings (IUPAC/chemical names, ~30%% match rate — slow)")
    ap.add_argument("--preview",       action="store_true",
                    help="Print dedup stats and estimated runtime without making API calls")
    args = ap.parse_args()

    mapper = RxNormMapper(
        cache_path    = args.cache,
        workers       = args.workers,
        exact_only    = args.exact_only,
        fetch_synonyms= not args.no_synonyms,
    )

    if args.preview:
        # Load names but don't resolve — just print stats
        tmp = RxNormMapper.__new__(RxNormMapper)
        tmp.cache_path = Path(args.cache)
        tmp._cache     = mapper._load_cache()
        tmp._entries   = {}
        all_names: list[str] = []
        for src, path, col in [
            ("drugbank",  args.drugbank,  "name"),
            ("faers",     args.faers,     "drugname"),
            ("dailymed",  args.dailymed,  "drug_name"),
        ]:
            if args.source and args.source != src:
                continue
            p = Path(path)
            if p.exists():
                df = pd.read_parquet(p)
                if col in df.columns:
                    all_names.extend(df[col].dropna().unique().tolist())
        mapper.preview_dedup(all_names)
    else:
        mapper.map_from_parquets(
            drugbank_path             = args.drugbank  if Path(args.drugbank).exists()  else None,
            faers_path                = args.faers     if Path(args.faers).exists()     else None,
            dailymed_path             = args.dailymed  if Path(args.dailymed).exists()  else None,
            source_filter             = args.source,
            limit                     = args.limit,
            include_drugbank_synonyms = args.include_db_synonyms,
        )
        mapper.save(args.out)

        df = mapper.get_mapping_df()
        print(f"\n── Mapping summary ─────────────────────────────────────")
        print(f"Total names   : {len(df):,}")
        print(f"Resolved      : {df['resolved'].sum():,} ({df['resolved'].mean()*100:.1f}%)")
        print(f"Match methods :\n{df['match_method'].value_counts().to_string()}")

        print(f"\n── Sample resolved ─────────────────────────────────────")
        resolved = df[df["resolved"]].head(10)
        print(resolved[["raw_name", "canonical_name", "cui", "tty", "match_method"]].to_string(index=False))

        print(f"\n── Sample unresolved ───────────────────────────────────")
        unresolved = df[~df["resolved"]].head(10)
        print(unresolved[["raw_name", "normalised_input"]].to_string(index=False))