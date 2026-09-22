"""
context_assembler.py  ·  Drug Watchdog Phase 3
================================================
Assembles the final structured context block that gets passed to the
LLM Explanation Agent (Phase 4).

Inputs
------
  • ML prediction result  (from predictor.py — severity 0–3, CYP pathway)
  • RerankedResult         (from reranker.py  — top-5 evidence chunks)
  • Drug pair metadata     (names, patient ID if available)

Output
------
  AssembledContext dataclass containing:
    • Structured dict ready to format into an LLM prompt
    • CitationIndex — maps every evidence claim to its source URL
    • Severity summary (plain English)
    • Token-count estimate (to avoid LLM context overflow)

Citation tracking
-----------------
Every evidence chunk that enters the assembled context gets assigned a
citation key like [FDA-1], [FAERS-2], [PUB-3].  The LLM is instructed
to use these keys inline so the explanation agent can auto-link them.
"""

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from citation_tracker import CitationIndex as DedicatedCitationIndex
from reranker import RankedChunk, RerankedResult
from retriever import RetrievedChunk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Severity helpers ────────────────────────────────

class Severity(IntEnum):
    NONE     = 0
    MINOR    = 1
    MODERATE = 2
    MAJOR    = 3

SEVERITY_LABELS = {
    Severity.NONE:     "No known interaction",
    Severity.MINOR:    "Minor interaction — monitor",
    Severity.MODERATE: "Moderate interaction — caution advised",
    Severity.MAJOR:    "Major interaction — avoid combination",
}

SEVERITY_COLOURS = {
    Severity.NONE:     "green",
    Severity.MINOR:    "yellow",
    Severity.MODERATE: "orange",
    Severity.MAJOR:    "red",
}

CYP_DESCRIPTIONS = {
    "CYP1A2":  "metabolises caffeine, theophylline, some antidepressants",
    "CYP2C9":  "metabolises warfarin, NSAIDs, some antidiabetics",
    "CYP2C19": "metabolises clopidogrel, PPIs, some SSRIs",
    "CYP2D6":  "metabolises codeine, tamoxifen, many antidepressants/antipsychotics",
    "CYP3A4":  "metabolises >50% of all drugs — major interaction gateway",
}

# ─────────────────────────── Citation index ───────────────────────────────────

class CitationIndex:
    """Assigns citation keys and tracks all sources used in one analysis."""

    def __init__(self):
        self._citations: list[Citation] = []
        self._seen_ids:  set[str]       = set()
        self._counters:  dict[str, int] = {}

    def add(self, chunk: RetrievedChunk) -> str:
        """Register a chunk and return its citation key."""
        uid = chunk.chunk_id
        if uid in self._seen_ids:
            # Return existing key
            for c in self._citations:
                if c.doc_id == chunk.doc_id and c.section == chunk.section:
                    return c.key
        prefix  = SOURCE_PREFIXES.get(chunk.source, "SRC")
        n       = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        key     = f"[{prefix}-{n}]"
        citation = Citation(
            key       = key,
            source    = chunk.source,
            doc_id    = chunk.doc_id,
            doc_title = chunk.doc_title,
            section   = chunk.section,
            doc_url   = chunk.doc_url,
            snippet   = chunk.text[:200],
        )
        self._citations.append(citation)
        self._seen_ids.add(uid)
        return key

    def to_list(self) -> list[dict]:
        return [
            {
                "key":       c.key,
                "source":    c.source,
                "doc_id":    c.doc_id,
                "doc_title": c.doc_title,
                "doc_url":   c.doc_url,
                "section":   c.section,
                "snippet":   c.snippet,
            }
            for c in self._citations
        ]

    def format_references(self) -> str:
        """Formatted reference block for the LLM prompt."""
        lines = ["REFERENCES:"]
        for c in self._citations:
            lines.append(
                f"  {c.key}  {c.source.upper()} · {c.doc_title or c.doc_id}"
                f" · {c.section}  →  {c.doc_url or 'no URL'}"
            )
        return "\n".join(lines)


CitationIndex = DedicatedCitationIndex


# ─────────────────────────── ML prediction stub ───────────────────────────────
# In production this comes from predictor.py.  We define the expected schema here.

@dataclass
class MLPrediction:
    drug_a:           str
    drug_b:           str
    severity:         int           # 0–3
    confidence:       float         # 0–1
    cyp_pathway:      str           # e.g. "CYP2C9"
    model_used:       str           # "xgboost" | "gnn"
    shap_top_features: list[dict]   # [{"feature": "logp_a", "value": 0.34, "importance": 0.18}]


# ─────────────────────────── Assembled context ───────────────────────────────

@dataclass
class AssembledContext:
    # Identifiers
    drug_a:          str
    drug_b:          str
    patient_id:      str | None

    # ML signal
    severity:        int
    severity_label:  str
    severity_colour: str
    confidence:      float
    cyp_pathway:     str
    cyp_description: str
    model_used:      str
    shap_features:   list[dict]

    # Evidence
    evidence_blocks: list[dict]     # ordered evidence chunks with citation keys
    citation_index:  CitationIndex

    # Prompt-ready text
    prompt_context:  str            # the full context block to inject into LLM prompt
    token_estimate:  int            # rough estimate (chars / 4)

    # Provenance hash (for logging/dedup)
    context_hash:    str


# ─────────────────────────── Assembler ───────────────────────────────────────

class ContextAssembler:
    """
    Drug Watchdog Phase 3 — Context Assembler.

    Usage
    -----
    assembler = ContextAssembler()
    ctx = assembler.assemble(
        ml_prediction  = prediction,    # from predictor.py
        reranked_result = reranked,     # from reranker.py
    )
    print(ctx.prompt_context)           # inject into LLM prompt
    print(ctx.citation_index.format_references())
    """

    # Max characters per evidence chunk in the assembled prompt
    CHUNK_MAX_CHARS = 600

    def assemble(
        self,
        ml_prediction:   MLPrediction,
        reranked_result: RerankedResult,
        patient_id:      str | None = None,
        max_chunks:      int        = 5,
    ) -> AssembledContext:
        """
        Combine ML prediction + top evidence chunks into a structured context.
        """
        drug_a = ml_prediction.drug_a
        drug_b = ml_prediction.drug_b
        sev    = Severity(ml_prediction.severity)

        # ── Build citation index ──────────────────────────────────────────────
        citations = CitationIndex()
        evidence_blocks: list[dict] = []

        for ranked_chunk in reranked_result.ranked_chunks[:max_chunks]:
            chunk  = ranked_chunk.chunk
            key    = citations.add(chunk)
            citation = citations.to_key_map().get(key, {})
            block  = {
                "citation_key":    key,
                "source":          chunk.source,
                "section":         chunk.section,
                "doc_title":       chunk.doc_title or chunk.doc_id,
                "doc_url":         citation.get("doc_url") or chunk.doc_url,
                "relevance_score": round(ranked_chunk.relevance_score, 3),
                "text":            chunk.text[:self.CHUNK_MAX_CHARS],
                "drug_names":      chunk.drug_names,
            }
            evidence_blocks.append(block)

        # ── Build structured prompt context ───────────────────────────────────
        prompt_context = self._build_prompt_context(
            drug_a         = drug_a,
            drug_b         = drug_b,
            sev            = sev,
            ml_prediction  = ml_prediction,
            evidence_blocks= evidence_blocks,
            citation_index = citations,
        )

        # ── Provenance hash ───────────────────────────────────────────────────
        context_hash = hashlib.md5(
            f"{drug_a}|{drug_b}|{ml_prediction.severity}|{reranked_result.query}".encode()
        ).hexdigest()[:12]

        return AssembledContext(
            drug_a          = drug_a,
            drug_b          = drug_b,
            patient_id      = patient_id,
            severity        = ml_prediction.severity,
            severity_label  = SEVERITY_LABELS[sev],
            severity_colour = SEVERITY_COLOURS[sev],
            confidence      = ml_prediction.confidence,
            cyp_pathway     = ml_prediction.cyp_pathway,
            cyp_description = CYP_DESCRIPTIONS.get(ml_prediction.cyp_pathway, "unknown pathway"),
            model_used      = ml_prediction.model_used,
            shap_features   = ml_prediction.shap_top_features,
            evidence_blocks = evidence_blocks,
            citation_index  = citations,
            prompt_context  = prompt_context,
            token_estimate  = len(prompt_context) // 4,
            context_hash    = context_hash,
        )

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt_context(
        self,
        drug_a:          str,
        drug_b:          str,
        sev:             Severity,
        ml_prediction:   MLPrediction,
        evidence_blocks: list[dict],
        citation_index:  CitationIndex,
    ) -> str:
        """
        Produces a structured context block like:

        ═══════════════════════════════════════════════════
        DRUG INTERACTION ANALYSIS CONTEXT
        ═══════════════════════════════════════════════════
        Drug Pair      : WARFARIN + ASPIRIN
        ML Severity    : 3 — MAJOR interaction (avoid combination)
        Confidence     : 94.2%
        CYP Pathway    : CYP2C9 (metabolises warfarin, NSAIDs ...)
        Model          : xgboost

        TOP SHAP FEATURES:
          • logp_a: 0.34 (importance 0.18)
          ...

        RETRIEVED EVIDENCE (top 5 — use citation keys inline):
        ───────────────────────────────────────────────────
        [FDA-1]  DailyMed — COUMADIN label · BOXED WARNING
          "Warfarin sodium has a narrow therapeutic range ..."

        [FDA-2]  DailyMed — ASPIRIN label · DRUG INTERACTIONS
          "Concomitant use of aspirin with anticoagulants ..."
          ...

        REFERENCES:
          [FDA-1]  FDA · COUMADIN label ...
        ═══════════════════════════════════════════════════
        """
        lines: list[str] = []
        div   = "═" * 70

        lines += [
            div,
            "DRUG INTERACTION ANALYSIS CONTEXT",
            div,
            f"Drug Pair      : {drug_a.upper()} + {drug_b.upper()}",
            f"ML Severity    : {ml_prediction.severity} — {SEVERITY_LABELS[sev].upper()}",
            f"Confidence     : {ml_prediction.confidence * 100:.1f}%",
            f"CYP Pathway    : {ml_prediction.cyp_pathway} "
            f"({CYP_DESCRIPTIONS.get(ml_prediction.cyp_pathway, 'unknown')})",
            f"Model          : {ml_prediction.model_used}",
            "",
        ]

        # SHAP features
        if ml_prediction.shap_top_features:
            lines.append("TOP SHAP FEATURES (driving the prediction):")
            for feat in ml_prediction.shap_top_features[:5]:
                lines.append(
                    f"  • {feat['feature']}: {feat['value']} "
                    f"(SHAP importance {feat['importance']:.3f})"
                )
            lines.append("")

        # Evidence blocks
        lines.append(f"RETRIEVED EVIDENCE (top {len(evidence_blocks)} — cite these keys inline):")
        lines.append("─" * 70)
        for block in evidence_blocks:
            lines += [
                f"{block['citation_key']}  "
                f"{block['source'].upper()} — {block['doc_title']}",
            ]
            if block["section"]:
                lines.append(f"  Section : {block['section']}")
            lines.append(f"  Score   : {block['relevance_score']}")
            # Truncate text for prompt
            text = block["text"].replace("\n", " ")
            lines.append(f"  Text    : \"{text[:self.CHUNK_MAX_CHARS]}\"")
            lines.append("")

        # References
        lines.append(citation_index.format_references())
        lines.append(div)

        return "\n".join(lines)

    # ── Batch assembly ────────────────────────────────────────────────────────

    def assemble_batch(
        self,
        predictions:      list[MLPrediction],
        reranked_results: list[RerankedResult],
    ) -> list[AssembledContext]:
        """Assemble contexts for multiple drug pairs at once."""
        if len(predictions) != len(reranked_results):
            raise ValueError("predictions and reranked_results must be the same length")
        contexts = []
        for pred, reranked in zip(predictions, reranked_results):
            ctx = self.assemble(pred, reranked)
            contexts.append(ctx)
        return contexts


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    # Fake ML prediction + reranked result for local testing
    from retriever import HybridRetriever, RetrievedChunk, RetrievalResult
    from reranker  import RankedChunk, RerankedResult, SourceWeightedReranker

    drug_a, drug_b = "warfarin", "aspirin"

    # --- Mock ML prediction ---
    pred = MLPrediction(
        drug_a    = drug_a,
        drug_b    = drug_b,
        severity  = 3,
        confidence= 0.942,
        cyp_pathway   = "CYP2C9",
        model_used    = "xgboost",
        shap_top_features = [
            {"feature": "logp_a",              "value": 0.34,  "importance": 0.182},
            {"feature": "molecular_weight",    "value": 308.3, "importance": 0.154},
            {"feature": "either_has_boxed_warning", "value": 1, "importance": 0.121},
            {"feature": "one_biologic",        "value": 0,    "importance": 0.089},
            {"feature": "both_small_molecule", "value": 1,    "importance": 0.072},
        ],
    )

    # --- Real retrieval + reranking ---
    retriever = HybridRetriever()
    reranker  = SourceWeightedReranker()

    retrieval = retriever.retrieve_for_pair(drug_a, drug_b, top_k=20)
    reranked  = reranker.rerank(retrieval, top_k=5)

    # --- Assemble context ---
    assembler = ContextAssembler()
    ctx       = assembler.assemble(pred, reranked)

    print(ctx.prompt_context)
    print(f"\n[token estimate: ~{ctx.token_estimate} tokens]")
    print(f"[context hash: {ctx.context_hash}]")
    print(f"\nCitation list ({len(ctx.citation_index.to_list())} citations):")
    for c in ctx.citation_index.to_list():
        print(f"  {c['key']}  {c['doc_title']}  {c['doc_url']}")
