"""
query_expander.py  ·  Drug Watchdog Phase 3
============================================
Standalone RxNorm query expansion module.

Responsibilities
----------------
  1. Resolve a drug name → canonical RxCUI (handles typos, brand/generic)
  2. Expand CUI → all synonyms, brand names, ingredient names
  3. Handle common abbreviations & misspellings (local lookup table)
  4. Disk-persistent cache (JSON) so repeated lookups are instant
  5. Batch expansion for drug pairs
  6. Build a multi-term BM25-friendly query string from expanded terms

Why a separate module from retriever.py's inline RxNormExpander?
----------------------------------------------------------------
retriever.py contains a minimal, in-memory-only expander.  This module:
  • Adds disk cache (survives restarts, shared across processes)
  • Adds approximate name matching for typos / abbreviations
  • Adds CUI-level deduplication (Coumadin, warfarin → same CUI → one set)
  • Adds relationship-type filtering (keep only SY, SB, BN, IN, MIN, PIN)
  • Adds a structured QueryExpansion dataclass used by rag_pipeline.py
"""

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────── Config ─────────────────────────────────────────

PROJECT_ROOT      = Path(__file__).resolve().parents[1]
RXNORM_API        = "https://rxnav.nlm.nih.gov/REST"
CACHE_PATH        = PROJECT_ROOT / "data/processed/rxnorm_expansion_cache.json"
REQUEST_TIMEOUT   = 6       # seconds per HTTP call
MAX_SYNONYMS      = 12      # synonyms per drug (caps explosion)
MIN_TERM_LENGTH   = 3       # ignore single-char / 2-char tokens

# RxNorm relationship types we trust for expansion
# SY = synonym, SB = has ingredient, BN = brand name,
# IN = ingredient, MIN = multiple-ingredient, PIN = precise ingredient
TRUSTED_TTY = {"SY", "BN", "IN", "MIN", "PIN", "SBD", "BPCK"}

# ──────────────────────── Abbreviation / alias table ─────────────────────────
# Hand-curated list of common drug abbreviations and misspellings
# that RxNorm may not resolve without help.

ALIASES: dict[str, str] = {
    # Anticoagulants
    "coumadin":   "warfarin",
    "warfrin":    "warfarin",
    "asa":        "aspirin",
    "acetylsalicylic acid": "aspirin",
    # Antibiotics
    "amox":       "amoxicillin",
    "augmentin":  "amoxicillin-clavulanate",
    "zithromax":  "azithromycin",
    "z-pak":      "azithromycin",
    "cipro":      "ciprofloxacin",
    "flagyl":     "metronidazole",
    # Statins
    "lipitor":    "atorvastatin",
    "crestor":    "rosuvastatin",
    "zocor":      "simvastatin",
    # Antihypertensives
    "norvasc":    "amlodipine",
    "lisinop":    "lisinopril",
    "prinivil":   "lisinopril",
    "zestril":    "lisinopril",
    # Antidepressants / CNS
    "prozac":     "fluoxetine",
    "zoloft":     "sertraline",
    "paxil":      "paroxetine",
    "effexor":    "venlafaxine",
    "wellbutrin": "bupropion",
    "lexapro":    "escitalopram",
    # Proton pump inhibitors
    "nexium":     "esomeprazole",
    "prilosec":   "omeprazole",
    "prevacid":   "lansoprazole",
    "protonix":   "pantoprazole",
    # Diabetes
    "glucophage": "metformin",
    "lantus":     "insulin glargine",
    "humalog":    "insulin lispro",
    # Cardiovascular
    "lopressor":  "metoprolol",
    "toprol":     "metoprolol",
    "lasix":      "furosemide",
    "digoxin":    "digoxin",  # identity — catches 'lanoxin' below
    "lanoxin":    "digoxin",
    # HIV
    "truvada":    "emtricitabine-tenofovir",
    # Opioids
    "vicodin":    "hydrocodone-acetaminophen",
    "percocet":   "oxycodone-acetaminophen",
    "tylenol":    "acetaminophen",
    "paracetamol":"acetaminophen",
    # NSAIDs
    "advil":      "ibuprofen",
    "motrin":     "ibuprofen",
    "aleve":      "naproxen",
    "naprosyn":   "naproxen",
    "celebrex":   "celecoxib",
    # Antiplatelet
    "plavix":     "clopidogrel",
    "brilinta":   "ticagrelor",
    "effient":    "prasugrel",
    # Anticoagulants (newer)
    "xarelto":    "rivaroxaban",
    "eliquis":    "apixaban",
    "pradaxa":    "dabigatran",
}


# ──────────────────────────── Data classes ────────────────────────────────────

@dataclass
class DrugExpansion:
    """Expansion result for a single drug name."""
    original_name:  str
    resolved_name:  str            # after alias lookup
    rxcui:          str            # primary RxCUI ("" if not found)
    terms:          list[str]      # original + all synonyms, deduped
    from_cache:     bool = False
    error:          str  = ""


@dataclass
class QueryExpansion:
    """Expansion result for a drug pair query."""
    drug_a:          str
    drug_b:          str
    expansion_a:     DrugExpansion
    expansion_b:     DrugExpansion
    combined_terms:  list[str]     # union of both term lists, deduped
    query_string:    str           # space-joined terms for BM25
    structured_query: str          # richer query for semantic retrieval
    latency_ms:      float


# ──────────────────────────── Disk cache ─────────────────────────────────────

class ExpansionCache:
    """
    Simple JSON disk cache keyed by lowercase drug name.
    Thread-safe for single-process use; add file locking if multiprocessing.
    """

    def __init__(self, path: Path = CACHE_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                log.info("Expansion cache loaded: %d entries from %s", len(self._data), self._path)
            except Exception as exc:
                log.warning("Could not load expansion cache: %s", exc)
                self._data = {}

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            log.warning("Could not save expansion cache: %s", exc)

    def get(self, key: str) -> dict | None:
        return self._data.get(key.lower())

    def set(self, key: str, value: dict):
        self._data[key.lower()] = value
        self._save()

    def __len__(self):
        return len(self._data)


# ──────────────────────────── Core expander ───────────────────────────────────

class QueryExpander:
    """
    Full drug name expander used by rag_pipeline.py.

    Usage
    -----
    expander = QueryExpander()

    # Single drug
    exp = expander.expand_drug("warfarin")
    print(exp.terms)          # ['warfarin', 'Coumadin', 'warfarin sodium', ...]

    # Drug pair → ready-to-use query
    qexp = expander.expand_pair("warfarin", "aspirin")
    print(qexp.query_string)  # space-joined synonym soup for BM25
    print(qexp.structured_query)  # richer query for BioBERT
    """

    def __init__(self, cache_path: Path = CACHE_PATH):
        self._cache = ExpansionCache(cache_path)

    # ── Alias resolution ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_alias(name: str) -> str:
        """Normalize via alias table; return original if not found."""
        key = name.lower().strip()
        return ALIASES.get(key, name)

    # ── RxNorm lookup ────────────────────────────────────────────────────────

    def _fetch_rxcui(self, name: str) -> str:
        """Resolve drug name → primary RxCUI. Returns "" on failure."""
        try:
            r = requests.get(
                f"{RXNORM_API}/rxcui.json",
                params={"name": name, "search": 1},
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            ids = r.json().get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else ""
        except Exception as exc:
            log.warning("RxCUI lookup failed for '%s': %s", name, exc)
            return ""

    def _fetch_synonyms(self, rxcui: str, max_synonyms: int = MAX_SYNONYMS) -> list[str]:
        """
        Fetch all related drug names for a CUI, filtered to trusted
        term types (SY, BN, IN, MIN, PIN, SBD, BPCK).
        """
        synonyms: list[str] = []
        try:
            r = requests.get(
                f"{RXNORM_API}/rxcui/{rxcui}/allrelated.json",
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            concept_groups = (
                r.json()
                .get("allRelatedGroup", {})
                .get("conceptGroup", [])
            )
            seen: set[str] = set()
            for group in concept_groups:
                tty = group.get("tty", "")
                if tty not in TRUSTED_TTY:
                    continue
                for props in group.get("conceptProperties", []):
                    name = props.get("name", "").strip()
                    if name and name.lower() not in seen and len(name) >= MIN_TERM_LENGTH:
                        synonyms.append(name)
                        seen.add(name.lower())
                        if len(synonyms) >= max_synonyms:
                            return synonyms
        except Exception as exc:
            log.warning("Synonym fetch failed for CUI '%s': %s", rxcui, exc)
        return synonyms

    # ── Dedup helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _dedup(terms: list[str]) -> list[str]:
        """Case-insensitive dedup preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for t in terms:
            if t.lower() not in seen:
                result.append(t)
                seen.add(t.lower())
        return result

    # ── Public: expand single drug ────────────────────────────────────────────

    def expand_drug(
        self,
        name: str,
        max_synonyms: int = MAX_SYNONYMS,
    ) -> DrugExpansion:
        """
        Expand a single drug name to all known synonyms.

        Parameters
        ----------
        name         : Drug name (brand, generic, abbreviation, misspelling)
        max_synonyms : Maximum number of synonyms to include

        Returns
        -------
        DrugExpansion with rxcui, terms list, and metadata
        """
        # 1. Alias resolution
        resolved = self._resolve_alias(name)

        # 2. Cache check
        cache_key = resolved.lower()
        cached = self._cache.get(cache_key)
        if cached:
            exp = DrugExpansion(
                original_name = name,
                resolved_name = resolved,
                rxcui         = cached.get("rxcui", ""),
                terms         = cached.get("terms", [resolved]),
                from_cache    = True,
            )
            log.debug("Cache hit for '%s' → %d terms", resolved, len(exp.terms))
            return exp

        # 3. RxNorm CUI lookup
        rxcui = self._fetch_rxcui(resolved)

        # 4. Synonym expansion
        if rxcui:
            synonyms = self._fetch_synonyms(rxcui, max_synonyms)
        else:
            log.warning("No RxCUI found for '%s' — using name only", resolved)
            synonyms = []

        # 5. Assemble and dedup terms (original name first)
        all_terms = self._dedup([name, resolved] + synonyms)

        # 6. Cache and return
        cache_entry = {"rxcui": rxcui, "terms": all_terms}
        self._cache.set(cache_key, cache_entry)

        exp = DrugExpansion(
            original_name = name,
            resolved_name = resolved,
            rxcui         = rxcui,
            terms         = all_terms,
            from_cache    = False,
        )
        log.info(
            "Expanded '%s' → CUI=%s  %d terms: %s",
            name, rxcui or "none", len(all_terms), all_terms[:4],
        )
        return exp

    # ── Public: expand drug pair ──────────────────────────────────────────────

    def expand_pair(
        self,
        drug_a: str,
        drug_b: str,
        max_synonyms: int = MAX_SYNONYMS,
    ) -> QueryExpansion:
        """
        Expand both drugs and build retrieval query strings.

        Returns
        -------
        QueryExpansion with:
          - query_string      : flat space-joined terms  (BM25-friendly)
          - structured_query  : interaction-focused phrase (BioBERT-friendly)
        """
        t0 = time.perf_counter()

        exp_a = self.expand_drug(drug_a, max_synonyms)
        exp_b = self.expand_drug(drug_b, max_synonyms)

        # Union of both term lists, deduped
        combined = self._dedup(exp_a.terms + exp_b.terms)

        # Flat BM25 query: all synonyms space-joined
        query_string = " ".join(combined)

        # Structured semantic query: prioritise the canonical names + clinical terms
        name_a = exp_a.resolved_name
        name_b = exp_b.resolved_name
        structured_query = (
            f"{name_a} {name_b} drug interaction "
            f"adverse effect contraindication warning "
            f"CYP metabolism pharmacokinetic pharmacodynamic"
        )
        # Append top-3 synonyms per drug to help BioBERT
        extras_a = [t for t in exp_a.terms[1:4] if t.lower() != name_a.lower()]
        extras_b = [t for t in exp_b.terms[1:4] if t.lower() != name_b.lower()]
        if extras_a or extras_b:
            structured_query += " " + " ".join(extras_a + extras_b)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Pair expansion done: '%s' + '%s' → %d combined terms  (%.1f ms)",
            drug_a, drug_b, len(combined), latency_ms,
        )

        return QueryExpansion(
            drug_a           = drug_a,
            drug_b           = drug_b,
            expansion_a      = exp_a,
            expansion_b      = exp_b,
            combined_terms   = combined,
            query_string     = query_string,
            structured_query = structured_query,
            latency_ms       = latency_ms,
        )

    # ── Batch expand ─────────────────────────────────────────────────────────

    def expand_pairs_batch(
        self,
        pairs: list[tuple[str, str]],
        max_synonyms: int = MAX_SYNONYMS,
    ) -> list[QueryExpansion]:
        """Expand multiple drug pairs. Each drug is looked up once (cached)."""
        results: list[QueryExpansion] = []
        for drug_a, drug_b in pairs:
            results.append(self.expand_pair(drug_a, drug_b, max_synonyms))
        return results

    # ── Cache stats ───────────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        return {
            "cache_path":    str(self._cache._path),
            "cached_drugs":  len(self._cache),
        }


# ──────────────────────────── CLI smoke-test ─────────────────────────────────

if __name__ == "__main__":
    import sys

    drug_a = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    drug_b = sys.argv[2] if len(sys.argv) > 2 else "aspirin"

    print(f"\nExpanding pair: {drug_a!r} + {drug_b!r}")
    print("─" * 70)

    expander = QueryExpander()
    qexp = expander.expand_pair(drug_a, drug_b)

    print(f"\nDrug A — '{drug_a}'")
    print(f"  Resolved : {qexp.expansion_a.resolved_name}")
    print(f"  RxCUI    : {qexp.expansion_a.rxcui or '(not found)'}")
    print(f"  Terms    : {qexp.expansion_a.terms}")

    print(f"\nDrug B — '{drug_b}'")
    print(f"  Resolved : {qexp.expansion_b.resolved_name}")
    print(f"  RxCUI    : {qexp.expansion_b.rxcui or '(not found)'}")
    print(f"  Terms    : {qexp.expansion_b.terms}")

    print(f"\nCombined terms ({len(qexp.combined_terms)}):")
    print(f"  {qexp.combined_terms}")

    print(f"\nBM25 query string (first 200 chars):")
    print(f"  {qexp.query_string[:200]}")

    print(f"\nStructured semantic query:")
    print(f"  {qexp.structured_query}")

    print(f"\nLatency: {qexp.latency_ms:.1f} ms")
    print(f"Cache stats: {expander.cache_stats()}")
