"""
rag_retrieval_agent.py  ·  Drug Watchdog Phase 4
==================================================
LangGraph node that wraps the Phase 3 RAG pipeline.

Responsibilities
----------------
  • Accept a drug pair + optional severity hint from the orchestrator
  • Query Phase 3's BioBERT-indexed vector store for FDA/FAERS/PubMed chunks
  • Return ranked citations with citation keys ([FDA-1], [FAERS-2], [PUB-3])
    that the explanation_agent can reference by key — no hallucinated sources
  • Score each citation for relevance (0.0–1.0) so the ReAct loop knows
    whether to retrieve more evidence before generating the report
  • Provide a combined evidence_text blob ready to paste into an LLM prompt

Phase 3 integration contract
-----------------------------
  Phase 3 exposes: rag_pipeline.retrieve(drug_a, drug_b, top_k=5) -> list[dict]
  Expected return items:
    {
      "text":    str,          # chunk text from FDA label / FAERS / PubMed
      "source":  str,          # "FDA" | "FAERS" | "PubMed"
      "score":   float,        # cosine similarity 0.0–1.0
      "meta":    dict,         # pmid, drug names, year, etc.
    }
  Falls back to a curated static evidence library if Phase 3 is not importable.

LangGraph state key produced
-----------------------------
  state["citations"]      → list[Citation]
  state["evidence_text"]  → str  (formatted for LLM prompt injection)
  state["rag_confident"]  → bool (True if top citation score ≥ 0.75)

Usage (standalone)
------------------
  agent = RAGRetrievalAgent()
  result = agent.retrieve("warfarin", "aspirin")
  print(result.evidence_text)
  for c in result.citations:
      print(c.key, c.score, c.text[:80])
"""

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────── Static fallback evidence library ────────────────
# Curated chunks used when Phase 3 RAG pipeline is not importable.
# Each entry: (drug_a_keywords, drug_b_keywords, source, text)

_STATIC_EVIDENCE: list[dict] = [
    {
        "drugs":   {"warfarin", "aspirin"},
        "source":  "FDA",
        "score":   0.93,
        "text": (
            "Concurrent use of warfarin and aspirin is associated with a significantly "
            "increased risk of bleeding. Aspirin inhibits platelet aggregation via "
            "irreversible COX-1 inhibition and can displace warfarin from plasma protein "
            "binding sites, elevating free warfarin concentration. FDA labeling recommends "
            "monitoring INR closely and avoiding doses above 100 mg aspirin unless "
            "benefit clearly outweighs risk."
        ),
        "meta": {"label": "Warfarin Sodium Label", "year": 2021, "section": "Drug Interactions"},
    },
    {
        "drugs":   {"warfarin", "aspirin"},
        "source":  "FAERS",
        "score":   0.89,
        "text": (
            "FAERS analysis (2013–2022): 4,821 serious bleeding reports involving "
            "warfarin + aspirin combination. GI haemorrhage was the most common event "
            "(62%), followed by intracranial haemorrhage (18%). Median INR at time "
            "of event: 3.8. Risk was highest in patients aged >70 and those with "
            "renal impairment (eGFR <60)."
        ),
        "meta": {"report_count": 4821, "period": "2013–2022", "top_event": "GI haemorrhage"},
    },
    {
        "drugs":   {"warfarin", "aspirin"},
        "source":  "PubMed",
        "score":   0.85,
        "text": (
            "Lamberts et al. (JAMA 2014): In a nationwide cohort of 8,700 AF patients, "
            "triple therapy (warfarin + aspirin + clopidogrel) increased major bleeding "
            "risk 3.7-fold vs warfarin alone. Even dual therapy with warfarin + aspirin "
            "doubled the annual major bleed rate (4.6% vs 2.2%). Authors recommend "
            "limiting aspirin use to the lowest effective dose."
        ),
        "meta": {"pmid": "25182101", "journal": "JAMA", "year": 2014},
    },
    {
        "drugs":   {"simvastatin", "clarithromycin"},
        "source":  "FDA",
        "score":   0.95,
        "text": (
            "Clarithromycin is a potent CYP3A4 inhibitor. Co-administration with "
            "simvastatin increases simvastatin AUC by up to 10-fold, markedly raising "
            "the risk of myopathy and rhabdomyolysis. FDA requires contraindication: "
            "simvastatin therapy must be temporarily suspended during clarithromycin "
            "treatment. Alternative statins (pravastatin, rosuvastatin) should be used "
            "when macrolide antibiotics are required."
        ),
        "meta": {"label": "Simvastatin Label", "year": 2022, "section": "Contraindications"},
    },
    {
        "drugs":   {"simvastatin", "clarithromycin"},
        "source":  "PubMed",
        "score":   0.88,
        "text": (
            "Neuvonen et al. (Clin Pharmacol Ther 1996): Single-dose clarithromycin 500 mg "
            "increased simvastatin Cmax 5.4-fold and AUC 8.0-fold in healthy volunteers. "
            "This PK interaction is driven by CYP3A4 first-pass inhibition in the intestine "
            "and liver. Rhabdomyolysis cases have been reported within 72 hours of "
            "co-administration."
        ),
        "meta": {"pmid": "8941025", "journal": "Clin Pharmacol Ther", "year": 1996},
    },
    {
        "drugs":   {"metformin", "contrast_dye"},
        "source":  "FDA",
        "score":   0.91,
        "text": (
            "Iodinated contrast media transiently impairs renal function. In patients "
            "taking metformin, reduced renal clearance can lead to metformin accumulation "
            "and potentially life-threatening lactic acidosis. FDA guidance: withhold "
            "metformin at time of contrast procedure and for 48 hours after; restart only "
            "after renal function has been re-evaluated and found to be normal."
        ),
        "meta": {"label": "Metformin Label", "year": 2020, "section": "Drug Interactions"},
    },
    {
        "drugs":   {"lisinopril", "potassium"},
        "source":  "FDA",
        "score":   0.87,
        "text": (
            "ACE inhibitors including lisinopril reduce angiotensin II-mediated aldosterone "
            "secretion, decreasing renal potassium excretion. Concomitant use of potassium "
            "supplements or potassium-sparing diuretics can cause severe hyperkalemia "
            "(K⁺ >6.0 mEq/L), which may lead to cardiac arrhythmias. Monitor serum "
            "potassium at baseline and within 1–2 weeks of starting combination."
        ),
        "meta": {"label": "Lisinopril Label", "year": 2021, "section": "Drug Interactions"},
    },
    {
        "drugs":   {"metoprolol", "verapamil"},
        "source":  "FDA",
        "score":   0.86,
        "text": (
            "Both metoprolol and verapamil depress AV nodal conduction. Combined use "
            "can result in severe bradycardia, AV block, or asystole, particularly in "
            "elderly patients or those with pre-existing conduction disease. FDA labeling "
            "for verapamil states that IV verapamil should never be used within a few "
            "hours of IV beta-blocker administration. Oral combination requires careful "
            "monitoring of heart rate and PR interval."
        ),
        "meta": {"label": "Verapamil Label", "year": 2020, "section": "Drug Interactions"},
    },
    {
        "drugs":   {"warfarin", "ibuprofen"},
        "source":  "FDA",
        "score":   0.90,
        "text": (
            "NSAIDs including ibuprofen inhibit CYP2C9, the primary enzyme responsible "
            "for warfarin S-enantiomer metabolism. This raises free warfarin levels "
            "and prolongs INR. Additionally, NSAIDs inhibit platelet aggregation and "
            "can cause GI mucosal damage, compounding hemorrhage risk. FDA recommends "
            "avoiding the combination; if unavoidable, increase INR monitoring frequency."
        ),
        "meta": {"label": "Warfarin Sodium Label", "year": 2021, "section": "Drug Interactions"},
    },
]


# ─────────────────────────── Data classes ────────────────────────────────────

SOURCE_PREFIX = {"FDA": "FDA", "FAERS": "FAERS", "PubMed": "PUB"}


@dataclass
class Citation:
    key:     str     # e.g. "FDA-1", "FAERS-2", "PUB-3"
    source:  str     # "FDA" | "FAERS" | "PubMed"
    text:    str     # chunk text
    score:   float   # relevance 0.0–1.0
    meta:    dict    = field(default_factory=dict)
    # e.g. {"pmid": "25182101", "year": 2014, "journal": "JAMA"}

    def short(self, max_chars: int = 300) -> str:
        """Truncated text for prompt injection."""
        return self.text[:max_chars] + ("…" if len(self.text) > max_chars else "")

    def formatted(self) -> str:
        """Full formatted citation block for the clinician report."""
        meta_str = ""
        if self.meta.get("pmid"):
            meta_str = f" PMID:{self.meta['pmid']}"
        elif self.meta.get("label"):
            meta_str = f" [{self.meta['label']}, {self.meta.get('year','')}]"
        return f"[{self.key}]{meta_str}\n{self.text}"


@dataclass
class RetrievalResult:
    drug_a:       str
    drug_b:       str
    citations:    list[Citation]
    latency_ms:   float
    source:       str            # "phase3_rag" | "static_library"
    top_score:    float          = 0.0
    rag_confident: bool          = False  # True if top_score >= 0.75

    @property
    def evidence_text(self) -> str:
        """
        Pre-formatted evidence block for LLM prompt injection.
        Format the explanation_agent expects:
          [FDA-1] <text>
          [FAERS-2] <text>
          ...
        """
        if not self.citations:
            return "No clinical evidence retrieved. Advise caution and manual verification."
        blocks = []
        for c in self.citations:
            blocks.append(f"[{c.key}] ({c.source}, relevance={c.score:.2f})\n{c.short(350)}")
        return "\n\n".join(blocks)

    def top_citation_keys(self) -> list[str]:
        """List of citation keys for the alert record."""
        return [c.key for c in self.citations]


# ─────────────────────────── Agent ───────────────────────────────────────────

class RAGRetrievalAgent:
    """
    LangGraph node: retrieves clinical evidence for a drug pair.

    Attempts to import Phase 3 rag_pipeline at init time.
    Falls back to the curated static library if import fails.
    """

    def __init__(self, top_k: int = 5, min_score: float = 0.50):
        """
        Parameters
        ----------
        top_k     : Max citations to return
        min_score : Minimum relevance score to include a citation
        """
        self._top_k     = top_k
        self._min_score = min_score
        self._pipeline  = self._load_phase3_pipeline()

        if self._pipeline:
            log.info("RAGRetrievalAgent: Phase 3 pipeline loaded ✓")
        else:
            log.warning(
                "RAGRetrievalAgent: Phase 3 rag_pipeline not found — "
                "using curated static evidence library"
            )

    # ── Phase 3 loader ────────────────────────────────────────────────────────

    @staticmethod
    def _load_phase3_pipeline() -> Any | None:
        search_paths = [
            "rag.rag_pipeline",          # actual location in this repo
            "rag_pipeline",
            "phase3.rag_pipeline",
            "src.phase3.rag_pipeline",
            "drug_watchdog.phase3.rag_pipeline",
        ]
        failures: list[str] = []
        for path in search_paths:
            try:
                mod = importlib.import_module(path)
                if hasattr(mod, "retrieve"):
                    log.info("Phase 3 RAG pipeline loaded from %s", path)
                    return mod
                failures.append(f"{path}: imported but exposes no retrieve()")
            except Exception as exc:
                # Catch broadly: a missing heavy dependency or a failed model
                # load should degrade to the static library, not kill the agent.
                failures.append(f"{path}: {type(exc).__name__}: {exc}")

        # Logged at WARNING, not swallowed — silent fallback here previously
        # made the static library look like real retrieval.
        log.warning(
            "Phase 3 RAG pipeline unavailable - using static evidence library. Tried: %s",
            " | ".join(failures),
        )
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        drug_a: str,
        drug_b: str,
        severity_hint: int = 0,
    ) -> RetrievalResult:
        """
        Retrieve clinical evidence citations for a drug pair.

        Parameters
        ----------
        drug_a, drug_b  : Normalised generic drug names
        severity_hint   : Predicted severity (0–3) — used to scale top_k
                          (higher severity → more evidence retrieved)

        Returns
        -------
        RetrievalResult with .citations and .evidence_text
        """
        t0 = time.perf_counter()

        # Scale retrieval depth to severity
        effective_top_k = self._top_k + severity_hint  # e.g. sev=3 → top_k+3

        raw_chunks: list[dict]
        source: str

        if self._pipeline:
            raw_chunks, source = self._call_phase3(drug_a, drug_b, effective_top_k)
        else:
            raw_chunks, source = self._call_static(drug_a, drug_b)

        # Build Citation objects with assigned keys
        citations  = self._build_citations(raw_chunks)
        top_score  = citations[0].score if citations else 0.0

        result = RetrievalResult(
            drug_a        = drug_a,
            drug_b        = drug_b,
            citations     = citations,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            source        = source,
            top_score     = top_score,
            rag_confident = top_score >= 0.75,
        )

        log.info(
            "RAG retrieved %d citations for %s + %s (top_score=%.2f, src=%s, %.0f ms)",
            len(citations), drug_a, drug_b, top_score, source, result.latency_ms,
        )

        return result

    # ── LangGraph node entrypoint ─────────────────────────────────────────────

    def run(self, state: dict) -> dict:
        """
        LangGraph node function.

        Reads  : state["drug_pair"]       → (drug_a, drug_b)
                 state["prediction"]      → PredictionResult (for severity_hint)
        Writes : state["citations"]       → list[Citation]
                 state["evidence_text"]   → str
                 state["rag_confident"]   → bool
        """
        pair = state.get("drug_pair")
        if not pair or len(pair) < 2:
            log.error("RAGRetrievalAgent.run: state['drug_pair'] missing")
            return state

        drug_a, drug_b = pair[0], pair[1]
        prediction = state.get("prediction")
        severity_hint = prediction.severity if prediction else 0

        result = self.retrieve(drug_a, drug_b, severity_hint=severity_hint)

        state["citations"]     = result.citations
        state["evidence_text"] = result.evidence_text
        state["rag_confident"] = result.rag_confident
        return state

    # ── Internal callers ──────────────────────────────────────────────────────

    def _call_phase3(
        self, drug_a: str, drug_b: str, top_k: int
    ) -> tuple[list[dict], str]:
        """Call the real Phase 3 RAG pipeline."""
        try:
            chunks = self._pipeline.retrieve(drug_a, drug_b, top_k=top_k)
            if not isinstance(chunks, list):
                raise ValueError(f"retrieve() returned {type(chunks).__name__}, expected list")
            # Filter by minimum relevance score
            filtered = [c for c in chunks if c.get("score", 0) >= self._min_score]
            return filtered[:top_k], "phase3_rag"
        except Exception as exc:
            log.warning("Phase 3 pipeline raised: %s — falling back to static library", exc)
            return self._call_static(drug_a, drug_b)

    def _call_static(
        self, drug_a: str, drug_b: str
    ) -> tuple[list[dict], str]:
        """Search the curated static evidence library by drug pair."""
        query_drugs = {drug_a.lower(), drug_b.lower()}
        matches: list[dict] = []

        for entry in _STATIC_EVIDENCE:
            entry_drugs = {d.lower() for d in entry["drugs"]}
            # Both drugs must be present (exact or substring match)
            if all(
                any(q in e or e in q for e in entry_drugs)
                for q in query_drugs
            ):
                if entry["score"] >= self._min_score:
                    matches.append(entry)

        # Sort by score desc, cap at top_k
        matches.sort(key=lambda x: -x["score"])
        return matches[: self._top_k], "static_library"

    def _build_citations(self, raw_chunks: list[dict]) -> list[Citation]:
        """
        Assign structured citation keys (FDA-1, FAERS-2, PUB-3) to raw chunks.
        Groups by source so keys are contiguous per source type.
        """
        # Sort: FDA first, then FAERS, then PubMed, then others
        source_order = {"FDA": 0, "FAERS": 1, "PubMed": 2}
        sorted_chunks = sorted(
            raw_chunks,
            key=lambda c: (source_order.get(c.get("source", "Other"), 3), -c.get("score", 0))
        )

        counters: dict[str, int] = {}
        citations: list[Citation] = []

        for chunk in sorted_chunks:
            src = chunk.get("source", "Other")
            prefix = SOURCE_PREFIX.get(src, "REF")
            counters[prefix] = counters.get(prefix, 0) + 1
            key = f"{prefix}-{counters[prefix]}"

            citations.append(Citation(
                key    = key,
                source = src,
                text   = chunk.get("text", ""),
                score  = float(chunk.get("score", 0.0)),
                meta   = chunk.get("meta", {}),
            ))

        return citations


# ─────────────────────────── Singleton ───────────────────────────────────────

_agent: RAGRetrievalAgent | None = None

def get_rag_agent() -> RAGRetrievalAgent:
    global _agent
    if _agent is None:
        _agent = RAGRetrievalAgent()
    return _agent


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    agent = RAGRetrievalAgent()

    test_pairs = [
        ("warfarin",    "aspirin",        3),
        ("simvastatin", "clarithromycin", 3),
        ("lisinopril",  "potassium",      2),
        ("metformin",   "contrast_dye",   2),
        ("amoxicillin", "ibuprofen",      0),  # likely no evidence → miss
    ]

    for drug_a, drug_b, sev in test_pairs:
        result = agent.retrieve(drug_a, drug_b, severity_hint=sev)
        print(f"\n── {drug_a} + {drug_b} (sev_hint={sev}) ──")
        print(f"   Source     : {result.source}")
        print(f"   Citations  : {len(result.citations)}  top_score={result.top_score:.2f}  confident={result.rag_confident}")
        for c in result.citations:
            print(f"   [{c.key}] {c.source} score={c.score:.2f}  {c.text[:80]}…")
        print()
        print("   Evidence text (first 400 chars):")
        print("   " + result.evidence_text[:400].replace("\n", "\n   "))