"""
reranker.py  ·  Drug Watchdog Phase 3
=======================================
Cross-encoder reranker that takes the top-K chunks from the hybrid
retriever and rescores them using a sentence-pair classification model.

Why this matters
----------------
FAISS cosine similarity and BM25 both operate on the query in isolation.
A cross-encoder sees (query, chunk) jointly → much higher precision at
the cost of speed.  We run it only on the top-K candidates (20–50),
which keeps latency manageable.

Model used
----------
  cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, good for passage re-ranking)
  OR
  cross-encoder/qnli-distilroberta-base  (better for yes/no relevance)

We default to ms-marco because it's trained on passage retrieval tasks.
"""

import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .retriever import RetrievedChunk, RetrievalResult
except ImportError:  # direct script execution, not package import
    from retriever import RetrievedChunk, RetrievalResult

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ─────────────────────────────────────────

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LENGTH      = 512
BATCH_SIZE          = 16      # pairs per forward pass
TOP_K_RERANKED      = 5       # how many chunks to keep after reranking

# ─────────────────────────── Data classes ────────────────────────────────────

@dataclass
class RankedChunk:
    """A RetrievedChunk with an additional cross-encoder relevance score."""
    chunk:            RetrievedChunk
    relevance_score:  float           # logit from cross-encoder (higher = more relevant)
    rerank_position:  int             # 1-indexed rank after reranking
    rrf_position:     int             # original rank from retriever

@dataclass
class RerankedResult:
    query:           str
    ranked_chunks:   list[RankedChunk]
    latency_ms:      float
    n_candidates:    int   # chunks given to reranker


# ─────────────────────────── Cross-Encoder ───────────────────────────────────

class CrossEncoderReranker:
    """
    Loads a cross-encoder model once and reranks any list of chunks.

    Usage
    -----
    reranker = CrossEncoderReranker()
    result   = reranker.rerank(retrieval_result, top_k=5)
    for rc in result.ranked_chunks:
        print(rc.rerank_position, rc.relevance_score, rc.chunk.text[:100])
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        log.info("Loading cross-encoder: %s …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self._device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)
        log.info("Cross-encoder ready on %s.", self._device)

    # ── Core scoring ─────────────────────────────────────────────────────────

    def _score_pairs(self, query: str, passages: list[str]) -> np.ndarray:
        """
        Score (query, passage) pairs in batches.
        Returns a 1-D numpy array of logits, shape (len(passages),).
        """
        all_logits: list[float] = []
        for i in range(0, len(passages), BATCH_SIZE):
            batch_passages = passages[i : i + BATCH_SIZE]
            pairs          = [(query, p) for p in batch_passages]

            enc = self.tokenizer(
                [q for q, _ in pairs],
                [p for _, p in pairs],
                padding      = True,
                truncation   = True,
                max_length   = MAX_SEQ_LENGTH,
                return_tensors = "pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits

            # ms-marco model outputs a single logit per pair (binary relevance)
            if logits.shape[-1] == 1:
                all_logits.extend(logits.squeeze(-1).cpu().tolist())
            else:
                # If multi-class, take the positive class score
                all_logits.extend(logits[:, 1].cpu().tolist())

        return np.array(all_logits, dtype=float)

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        retrieval_result: RetrievalResult,
        top_k: int = TOP_K_RERANKED,
        query_override: str | None = None,
    ) -> RerankedResult:
        """
        Parameters
        ----------
        retrieval_result : Output from HybridRetriever.retrieve()
        top_k            : Number of chunks to return after reranking
        query_override   : Use a different query string for scoring (optional)

        Returns
        -------
        RerankedResult with ranked_chunks sorted by cross-encoder relevance
        """
        t0 = time.perf_counter()

        query    = query_override or retrieval_result.query
        chunks   = retrieval_result.chunks

        if not chunks:
            return RerankedResult(
                query         = query,
                ranked_chunks = [],
                latency_ms    = 0.0,
                n_candidates  = 0,
            )

        passages = [c.text for c in chunks]
        scores   = self._score_pairs(query, passages)

        # Sort by score descending
        order    = np.argsort(scores)[::-1]

        ranked_chunks: list[RankedChunk] = []
        for new_pos, orig_pos in enumerate(order[:top_k], start=1):
            chunk = chunks[orig_pos]
            ranked_chunks.append(RankedChunk(
                chunk           = chunk,
                relevance_score = float(scores[orig_pos]),
                rerank_position = new_pos,
                rrf_position    = orig_pos + 1,   # 1-indexed original position
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Reranking done — %d → top-%d  in %.1f ms",
            len(chunks), len(ranked_chunks), latency_ms,
        )
        return RerankedResult(
            query         = query,
            ranked_chunks = ranked_chunks,
            latency_ms    = latency_ms,
            n_candidates  = len(chunks),
        )

    def rerank_chunks(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  int = TOP_K_RERANKED,
    ) -> RerankedResult:
        """
        Convenience method: rerank a raw list of RetrievedChunk objects
        without a RetrievalResult wrapper.
        """
        # Wrap in a minimal RetrievalResult-like object
        dummy = RetrievalResult(
            query          = query,
            expanded_terms = [],
            chunks         = chunks,
            latency_ms     = 0.0,
            dense_hits     = len(chunks),
            sparse_hits    = len(chunks),
        )
        return self.rerank(dummy, top_k=top_k, query_override=query)


# ─────────────────────────── Source-weighted reranker ────────────────────────

class SourceWeightedReranker(CrossEncoderReranker):
    """
    Extends the base reranker with domain-specific source weights.
    FDA DailyMed boxed warnings get a score boost; PubMed abstracts
    get a slight boost; FAERS signal counts inform an additive bonus.

    Source weights (additive logit adjustment):
      dailymed  + boxed_warning_section → +2.0
      dailymed  + drug_interactions     → +1.0
      faers                             → +0.5   (scaled by log10(report_count))
      pubmed                            → +0.3
      drugbank                          → +0.2
    """

    SOURCE_BASE = {
        "dailymed":  0.8,
        "faers":     0.5,
        "pubmed":    0.3,
        "drugbank":  0.2,
    }
    SECTION_BONUS = {
        "BOXED WARNING":       2.0,
        "WARNINGS AND PRECAUTIONS": 1.2,
        "DRUG INTERACTIONS":   1.0,
        "CONTRAINDICATIONS":   1.5,
        "ADVERSE REACTIONS":   0.6,
    }

    def rerank(
        self,
        retrieval_result: RetrievalResult,
        top_k: int        = TOP_K_RERANKED,
        query_override:   str | None = None,
    ) -> RerankedResult:
        # Run base cross-encoder on all candidates
        base_result = super().rerank(
            retrieval_result,
            top_k   = len(retrieval_result.chunks),   # score all, filter later
            query_override = query_override,
        )

        # Apply source weights
        for rc in base_result.ranked_chunks:
            src     = rc.chunk.source.lower()
            section = rc.chunk.section.upper()
            bonus   = self.SOURCE_BASE.get(src, 0.0)
            bonus  += self.SECTION_BONUS.get(section, 0.0)
            rc.relevance_score += bonus

        # Re-sort by updated score
        base_result.ranked_chunks.sort(key=lambda x: x.relevance_score, reverse=True)

        # Re-assign positions and truncate
        for i, rc in enumerate(base_result.ranked_chunks, start=1):
            rc.rerank_position = i

        base_result.ranked_chunks = base_result.ranked_chunks[:top_k]
        return base_result


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    import sys
    from retriever import HybridRetriever

    drug_a = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    drug_b = sys.argv[2] if len(sys.argv) > 2 else "aspirin"

    print(f"\nRunning full retrieve → rerank for: {drug_a} + {drug_b}")
    print("─" * 70)

    retriever = HybridRetriever()
    reranker  = SourceWeightedReranker()

    retrieval_result = retriever.retrieve_for_pair(drug_a, drug_b, top_k=20)
    reranked         = reranker.rerank(retrieval_result, top_k=5)

    print(f"Total latency: {retrieval_result.latency_ms + reranked.latency_ms:.1f} ms")
    print(f"Candidates given to reranker : {reranked.n_candidates}")
    print(f"Top-{len(reranked.ranked_chunks)} after reranking:\n")

    for rc in reranked.ranked_chunks:
        print(f"  [{rc.rerank_position}] score={rc.relevance_score:.3f}  "
              f"(was RRF #{rc.rrf_position})  source={rc.chunk.source}")
        print(f"      section  : {rc.chunk.section}")
        print(f"      doc      : {rc.chunk.doc_title or rc.chunk.doc_id}")
        print(f"      text     : {rc.chunk.text[:180]} …")
        print()
