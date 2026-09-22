"""
rag_pipeline.py  ·  Drug Watchdog Phase 3
==========================================
Master RAG pipeline orchestrator.

Wires together all Phase 3 components in the correct order:

  query_expander.py   →  expand drug names via RxNorm
        ↓
  retriever.py        →  hybrid BioBERT + BM25 + RRF retrieval
        ↓
  reranker.py         →  cross-encoder + source-weighted reranking
        ↓
  context_assembler.py →  ML prediction + evidence → cited LLM context
        ↓
  AssembledContext    →  ready for Phase 4 Explanation Agent

Architecture
------------
  RAGPipeline
    .run(drug_a, drug_b, ml_prediction)   → AssembledContext
    .run_batch([pairs], [predictions])    → list[AssembledContext]

All latency is tracked per stage and logged as a summary table.

Output
------
  AssembledContext  (see context_assembler.py for the full schema)
  Includes:
    • prompt_context  : inject directly into Phase 4 LLM agent
    • citation_index  : [FDA-1], [FAERS-2], [PUB-3] → full source details
    • severity_label  : "Major interaction — avoid combination"
    • token_estimate  : context window pre-check

Example output string the Phase 4 agent will receive
------------------------------------------------------
  "warfarin + aspirin interaction confirmed by FDA DailyMed boxed warning
   [FDA-1] + 847 FAERS death reports [FAERS-2] + 3 PubMed citations [PUB-3]."
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from context_assembler import AssembledContext, ContextAssembler, MLPrediction
from query_expander     import QueryExpander, QueryExpansion
from reranker           import RerankedResult, SourceWeightedReranker
from retriever          import HybridRetriever, RetrievalResult

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────── Config ─────────────────────────────────────────

# How many candidates to retrieve before reranking
RETRIEVAL_TOP_K  = 20
# How many chunks to keep after reranking (passed to context assembler)
RERANK_TOP_K     = 5
# Max chunks included in the assembled LLM context
CONTEXT_MAX_CHUNKS = 5

# Output directory for saved pipeline results
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data/rag_outputs"


# ──────────────────────────── Timing helper ──────────────────────────────────

@dataclass
class PipelineTimings:
    expansion_ms:  float = 0.0
    retrieval_ms:  float = 0.0
    rerank_ms:     float = 0.0
    assembly_ms:   float = 0.0

    @property
    def total_ms(self) -> float:
        return self.expansion_ms + self.retrieval_ms + self.rerank_ms + self.assembly_ms

    def summary(self) -> str:
        return (
            f"  Expansion  : {self.expansion_ms:6.1f} ms\n"
            f"  Retrieval  : {self.retrieval_ms:6.1f} ms\n"
            f"  Reranking  : {self.rerank_ms:6.1f} ms\n"
            f"  Assembly   : {self.assembly_ms:6.1f} ms\n"
            f"  ─────────────────────\n"
            f"  TOTAL      : {self.total_ms:6.1f} ms"
        )


# ──────────────────────────── RAG Pipeline ───────────────────────────────────

class RAGPipeline:
    """
    Drug Watchdog Phase 3 — end-to-end RAG pipeline.

    Initialising this class loads all models into memory once:
      • BioBERT (768-dim encoder, ~400 MB)
      • FAISS index (1.29 M vectors)
      • BM25 index (built over DB chunks at startup, ~2–3 min)
      • Cross-encoder (ms-marco-MiniLM, ~90 MB)
      • RxNorm expander (HTTP + disk cache, instant after first run)

    After init, each .run() call typically completes in 200–600 ms.

    Parameters
    ----------
    retrieval_top_k  : Candidates passed to reranker (default 20)
    rerank_top_k     : Chunks kept after reranking   (default 5)
    context_max_chunks : Chunks included in LLM context (default 5)
    save_outputs     : If True, save JSON result per query to OUTPUT_DIR
    """

    def __init__(
        self,
        retrieval_top_k:    int  = RETRIEVAL_TOP_K,
        rerank_top_k:       int  = RERANK_TOP_K,
        context_max_chunks: int  = CONTEXT_MAX_CHUNKS,
        save_outputs:       bool = False,
    ):
        self.retrieval_top_k    = retrieval_top_k
        self.rerank_top_k       = rerank_top_k
        self.context_max_chunks = context_max_chunks
        self.save_outputs       = save_outputs

        log.info("Initialising RAG Pipeline — loading models …")
        t0 = time.perf_counter()

        self._expander  = QueryExpander()
        self._retriever = HybridRetriever()
        self._reranker  = SourceWeightedReranker()
        self._assembler = ContextAssembler()

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("RAG Pipeline ready — models loaded in %.1f ms", elapsed)

        if save_outputs:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Core run ─────────────────────────────────────────────────────────────

    def run(
        self,
        drug_a:        str,
        drug_b:        str,
        ml_prediction: MLPrediction,
        patient_id:    str | None = None,
    ) -> tuple[AssembledContext, PipelineTimings]:
        """
        Run the full RAG pipeline for a drug pair.

        Parameters
        ----------
        drug_a         : First drug name (brand or generic)
        drug_b         : Second drug name (brand or generic)
        ml_prediction  : Output from predictor.py (severity 0–3, CYP, SHAP)
        patient_id     : Optional — included in context for traceability

        Returns
        -------
        (AssembledContext, PipelineTimings)
          AssembledContext.prompt_context  → inject into Phase 4 LLM prompt
          AssembledContext.citation_index  → citation key → source URL map
        """
        timings = PipelineTimings()
        log.info("RAG run: %s + %s  (severity=%d)", drug_a, drug_b, ml_prediction.severity)

        # ── Stage 1: Query Expansion ──────────────────────────────────────────
        t1 = time.perf_counter()
        qexp: QueryExpansion = self._expander.expand_pair(drug_a, drug_b)
        timings.expansion_ms = (time.perf_counter() - t1) * 1000

        log.info(
            "Stage 1 expansion: %d combined terms in %.1f ms",
            len(qexp.combined_terms), timings.expansion_ms,
        )

        # ── Stage 2: Hybrid Retrieval ─────────────────────────────────────────
        t2 = time.perf_counter()
        retrieval_result: RetrievalResult = self._retriever.retrieve(
            query      = qexp.structured_query,
            drug_names = qexp.combined_terms,   # pass all synonyms for BM25
            top_k      = self.retrieval_top_k,
        )
        timings.retrieval_ms = (time.perf_counter() - t2) * 1000

        log.info(
            "Stage 2 retrieval: %d chunks  (dense=%d sparse=%d)  %.1f ms",
            len(retrieval_result.chunks),
            retrieval_result.dense_hits,
            retrieval_result.sparse_hits,
            timings.retrieval_ms,
        )

        # ── Stage 3: Cross-Encoder Reranking ──────────────────────────────────
        t3 = time.perf_counter()
        # Use the structured query (not all synonyms) as the scoring query
        reranked_result: RerankedResult = self._reranker.rerank(
            retrieval_result,
            top_k          = self.rerank_top_k,
            query_override = qexp.structured_query,
        )
        timings.rerank_ms = (time.perf_counter() - t3) * 1000

        log.info(
            "Stage 3 reranking: top-%d from %d candidates  %.1f ms",
            len(reranked_result.ranked_chunks),
            reranked_result.n_candidates,
            timings.rerank_ms,
        )

        # ── Stage 4: Context Assembly ─────────────────────────────────────────
        t4 = time.perf_counter()
        context: AssembledContext = self._assembler.assemble(
            ml_prediction   = ml_prediction,
            reranked_result = reranked_result,
            patient_id      = patient_id,
            max_chunks      = self.context_max_chunks,
        )
        timings.assembly_ms = (time.perf_counter() - t4) * 1000

        log.info(
            "Stage 4 assembly: %d citations  ~%d tokens  hash=%s  %.1f ms",
            len(context.citation_index.to_list()),
            context.token_estimate,
            context.context_hash,
            timings.assembly_ms,
        )

        # ── Timing summary ────────────────────────────────────────────────────
        log.info(
            "Pipeline complete for %s + %s:\n%s",
            drug_a, drug_b, timings.summary(),
        )

        # ── Optional: save output ─────────────────────────────────────────────
        if self.save_outputs:
            self._save(drug_a, drug_b, context, timings)

        return context, timings

    # ── Batch run ─────────────────────────────────────────────────────────────

    def run_batch(
        self,
        pairs:          list[tuple[str, str]],
        ml_predictions: list[MLPrediction],
    ) -> list[tuple[AssembledContext, PipelineTimings]]:
        """
        Run the pipeline for multiple drug pairs.

        Parameters
        ----------
        pairs          : list of (drug_a, drug_b) tuples
        ml_predictions : list of MLPrediction objects, same length as pairs

        Returns
        -------
        list of (AssembledContext, PipelineTimings) in the same order
        """
        if len(pairs) != len(ml_predictions):
            raise ValueError("pairs and ml_predictions must be the same length")

        results = []
        for (drug_a, drug_b), pred in zip(pairs, ml_predictions):
            result = self.run(drug_a, drug_b, pred)
            results.append(result)

        total = sum(r[1].total_ms for r in results)
        log.info(
            "Batch complete: %d pairs in %.1f ms  (avg %.1f ms/pair)",
            len(pairs), total, total / len(pairs),
        )
        return results

    # ── Save helper ───────────────────────────────────────────────────────────

    def _save(
        self,
        drug_a:   str,
        drug_b:   str,
        context:  AssembledContext,
        timings:  PipelineTimings,
    ):
        """Save pipeline result as JSON to OUTPUT_DIR."""
        filename = OUTPUT_DIR / f"{drug_a.lower()}_{drug_b.lower()}_{context.context_hash}.json"
        payload = {
            "drug_a":          context.drug_a,
            "drug_b":          context.drug_b,
            "patient_id":      context.patient_id,
            "severity":        context.severity,
            "severity_label":  context.severity_label,
            "severity_colour": context.severity_colour,
            "confidence":      context.confidence,
            "cyp_pathway":     context.cyp_pathway,
            "cyp_description": context.cyp_description,
            "model_used":      context.model_used,
            "shap_features":   context.shap_features,
            "evidence_blocks": context.evidence_blocks,
            "citations":       context.citation_index.to_list(),
            "prompt_context":  context.prompt_context,
            "token_estimate":  context.token_estimate,
            "context_hash":    context.context_hash,
            "timings": {
                "expansion_ms":  timings.expansion_ms,
                "retrieval_ms":  timings.retrieval_ms,
                "rerank_ms":     timings.rerank_ms,
                "assembly_ms":   timings.assembly_ms,
                "total_ms":      timings.total_ms,
            },
        }
        try:
            with open(filename, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("Output saved → %s", filename)
        except Exception as exc:
            log.warning("Could not save output: %s", exc)


# ──────────────────────────── CLI smoke-test ─────────────────────────────────

if __name__ == "__main__":
    import sys
    from context_assembler import MLPrediction

    drug_a = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    drug_b = sys.argv[2] if len(sys.argv) > 2 else "aspirin"

    # ── Mock ML prediction (replace with real predictor.py output in Phase 4) ─
    mock_prediction = MLPrediction(
        drug_a    = drug_a,
        drug_b    = drug_b,
        severity  = 3,
        confidence= 0.942,
        cyp_pathway = "CYP2C9",
        model_used  = "xgboost",
        shap_top_features = [
            {"feature": "logp_a",                  "value": 0.34,  "importance": 0.182},
            {"feature": "molecular_weight",        "value": 308.3, "importance": 0.154},
            {"feature": "either_has_boxed_warning","value": 1,     "importance": 0.121},
            {"feature": "one_biologic",            "value": 0,     "importance": 0.089},
            {"feature": "both_small_molecule",     "value": 1,     "importance": 0.072},
        ],
    )

    pipeline = RAGPipeline(save_outputs=True)
    context, timings = pipeline.run(drug_a, drug_b, mock_prediction)

    print("\n" + "═" * 70)
    print("ASSEMBLED CONTEXT (injected into Phase 4 LLM Agent)")
    print("═" * 70)
    print(context.prompt_context)

    print("\n" + "─" * 70)
    print("PIPELINE TIMINGS")
    print("─" * 70)
    print(timings.summary())

    print("\n" + "─" * 70)
    print(f"Token estimate : ~{context.token_estimate} tokens")
    print(f"Context hash   : {context.context_hash}")
    print(f"Severity       : {context.severity_label}  [{context.severity_colour.upper()}]")
    print(f"Citations      : {len(context.citation_index.to_list())}")
    for c in context.citation_index.to_list():
        print(f"  {c['key']}  {c['source'].upper()} · {c['doc_title'] or c['doc_id']}")
        if c["doc_url"]:
            print(f"        → {c['doc_url']}")
