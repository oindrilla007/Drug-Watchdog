"""
retriever.py  ·  Drug Watchdog Phase 3
========================================
Hybrid retrieval engine combining:
  1. Dense  — BioBERT embeddings queried via FAISS
  2. Sparse — BM25 over the same corpus
  3. RRF    — Reciprocal Rank Fusion to merge both ranked lists
  4. Query expansion via RxNorm synonym mapping

Returns ranked RetrievedChunk objects ready for the cross-encoder reranker.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import psycopg2
import requests
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ─────────────────────────────────────────

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
FAISS_INDEX_PATH   = PROJECT_ROOT / "data/embeddings/faiss_index.bin"
FAISS_EMB_PATH     = PROJECT_ROOT / "data/embeddings/faiss_embeddings.npy"
FAISS_META_JSON    = PROJECT_ROOT / "data/embeddings/faiss_metadata.json"
FAISS_META_PARQUET = PROJECT_ROOT / "data/embeddings/faiss_metadata.parquet"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data/embeddings/chroma_db"
BIOBERT_MODEL      = "dmis-lab/biobert-base-cased-v1.2"
FALLBACK_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM      = 768
TOP_K_DENSE        = 50     # candidates from FAISS before fusion
TOP_K_SPARSE       = 50     # candidates from BM25 before fusion
TOP_K_FINAL        = 20     # chunks returned after RRF (pre-rerank)
RRF_K              = 60     # standard RRF constant
RXNORM_API         = "https://rxnav.nlm.nih.gov/REST"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "drugwatchdog",
    "user":     "postgres",
    "password": "postgres",
}

# ─────────────────────────── Data classes ────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    source:       str           # "dailymed" | "faers" | "pubmed" | "drugbank"
    drug_names:   list[str]
    rrf_score:    float
    dense_rank:   int  = -1
    sparse_rank:  int  = -1
    # Citation metadata (populated by source-specific loaders)
    doc_id:       str  = ""     # FDA label set_id, PMID, etc.
    doc_title:    str  = ""
    doc_url:      str  = ""
    section:      str  = ""     # "WARNINGS", "DRUG INTERACTIONS", etc.

@dataclass
class RetrievalResult:
    query:           str
    expanded_terms:  list[str]
    chunks:          list[RetrievedChunk]
    latency_ms:      float
    dense_hits:      int
    sparse_hits:     int

# ─────────────────────────── RxNorm expander ─────────────────────────────────

class RxNormExpander:
    """
    Expands a drug name to its synonyms and brand names via the free
    RxNorm REST API (no API key needed).
    Results are cached in-memory to avoid repeated network calls.
    """

    def __init__(self):
        self._cache: dict[str, list[str]] = {}

    def expand(self, drug_name: str, max_synonyms: int = 6) -> list[str]:
        """Return [drug_name] + up to max_synonyms synonyms/brand names."""
        key = drug_name.lower().strip()
        if key in self._cache:
            return self._cache[key]

        synonyms = [drug_name]
        try:
            # Step 1: resolve to RxCUI
            r = requests.get(
                f"{RXNORM_API}/rxcui.json",
                params={"name": drug_name, "search": 1},
                timeout=5,
            )
            r.raise_for_status()
            id_group = r.json().get("idGroup", {})
            cuis = id_group.get("rxnormId", [])
            if not cuis:
                self._cache[key] = synonyms
                return synonyms

            rxcui = cuis[0]

            # Step 2: fetch all related names (brand + generic synonyms)
            r2 = requests.get(
                f"{RXNORM_API}/rxcui/{rxcui}/allrelated.json",
                timeout=5,
            )
            r2.raise_for_status()
            concept_groups = (
                r2.json()
                .get("allRelatedGroup", {})
                .get("conceptGroup", [])
            )
            seen = {drug_name.lower()}
            for group in concept_groups:
                for props in group.get("conceptProperties", []):
                    name = props.get("name", "").strip()
                    if name and name.lower() not in seen:
                        synonyms.append(name)
                        seen.add(name.lower())
                        if len(synonyms) >= max_synonyms + 1:
                            break
                if len(synonyms) >= max_synonyms + 1:
                    break

        except Exception as exc:
            log.warning("RxNorm expansion failed for '%s': %s", drug_name, exc)

        self._cache[key] = synonyms
        log.info("RxNorm expanded '%s' → %s", drug_name, synonyms)
        return synonyms


# ─────────────────────────── BioBERT embedder ────────────────────────────────

class BioBERTEmbedder:
    """Encodes text to dense embeddings using the model that built the index."""

    def __init__(self, model_name: str = BIOBERT_MODEL):
        log.info("Loading embedding tokenizer & model: %s …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.eval()
        log.info("Embedding model ready.")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.model(**enc)
            # Mean pool over token dimension
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            # Avoid torch.Tensor.numpy(); some Windows torch builds compiled
            # against NumPy 1.x cannot bridge to NumPy 2.x at runtime.
            emb    = np.asarray((summed / counts).detach().cpu().tolist(), dtype=np.float32)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype("float32")


# ─────────────────────────── BM25 corpus ─────────────────────────────────────

class BM25Index:
    """
    Tokenised BM25 index built over the same corpus stored in PostgreSQL.
    We load chunk texts at startup (fits in RAM for ~1.3M short chunks).
    """

    def __init__(self, chunks: list[dict]):
        log.info("Building BM25 index over %d chunks …", len(chunks))
        self._chunks    = chunks
        tokenised       = [self._tokenise(c["text"]) for c in chunks]
        self._bm25      = BM25Okapi(tokenised)
        log.info("BM25 index ready.")

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = TOP_K_SPARSE) -> list[tuple[int, float]]:
        """Returns list of (corpus_index, bm25_score) sorted descending."""
        scores = self._bm25.get_scores(self._tokenise(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


# ─────────────────────────── Reciprocal Rank Fusion ──────────────────────────

def reciprocal_rank_fusion(
    dense_ranked:  list[str],   # chunk_ids ordered by dense score
    sparse_ranked: list[str],   # chunk_ids ordered by BM25 score
    k:             int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Standard RRF: score(d) = sum_r 1/(k + rank_r(d))
    Returns [(chunk_id, rrf_score)] sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, cid in enumerate(dense_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, cid in enumerate(sparse_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────── Main Retriever ──────────────────────────────────

class HybridRetriever:
    """
    Drug Watchdog Phase 3 — Hybrid Retriever.

    Usage
    -----
    retriever = HybridRetriever()
    result    = retriever.retrieve("warfarin aspirin bleeding risk")
    for chunk in result.chunks[:5]:
        print(chunk.rrf_score, chunk.source, chunk.text[:120])
    """

    def __init__(self):
        # 1. FAISS index + metadata
        log.info("Loading FAISS index from %s …", FAISS_INDEX_PATH)
        self._faiss_index = self._load_vector_index()
        self._faiss_meta = self._load_faiss_metadata()
        log.info("FAISS: %d vectors loaded.", self._faiss_index.ntotal)

        # 2. BioBERT embedder
        model_name = FALLBACK_MODEL if self._faiss_index.d == 384 else BIOBERT_MODEL
        if model_name == FALLBACK_MODEL:
            log.info("Detected 384-dim embeddings; using MiniLM query encoder.")
        self._embedder = BioBERTEmbedder(model_name=model_name)

        # 3. Load all chunks from PostgreSQL for BM25
        log.info("Loading chunks from PostgreSQL for BM25 …")
        raw_chunks = self._load_chunks()
        self._chunk_map: dict[str, dict] = {c["chunk_id"]: c for c in raw_chunks}

        # 4. BM25 index
        self._bm25 = BM25Index(raw_chunks)

        # 5. RxNorm expander
        self._expander = RxNormExpander()

    # ── DB loader ────────────────────────────────────────────────────────────

    def _load_vector_index(self):
        if FAISS_INDEX_PATH.exists() and FAISS_INDEX_PATH.stat().st_size > 0:
            return faiss.read_index(str(FAISS_INDEX_PATH))

        if not FAISS_EMB_PATH.exists():
            raise FileNotFoundError(
                f"No usable FAISS index found. Missing fallback embeddings: {FAISS_EMB_PATH}"
            )

        log.warning("FAISS index is missing or empty; rebuilding from %s", FAISS_EMB_PATH)
        embeddings = np.load(str(FAISS_EMB_PATH)).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        try:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            log.info("Rebuilt FAISS index saved -> %s", FAISS_INDEX_PATH)
        except Exception as exc:
            log.warning("Could not save rebuilt FAISS index: %s", exc)

        return index

    def _load_faiss_metadata(self) -> list[dict]:
        if FAISS_META_JSON.exists():
            with open(FAISS_META_JSON, encoding="utf-8") as f:
                return json.load(f)

        if FAISS_META_PARQUET.exists():
            df = pd.read_parquet(FAISS_META_PARQUET)
            return [self._normalise_chunk_record(row) for row in df.to_dict("records")]

        raise FileNotFoundError(
            f"FAISS metadata not found: {FAISS_META_JSON} or {FAISS_META_PARQUET}"
        )

    @staticmethod
    def _normalise_chunk_record(row: dict) -> dict:
        drug_names = row.get("drug_names") or []
        if isinstance(drug_names, str):
            drug_names = [d for d in drug_names.split("|") if d]

        return {
            "chunk_id":   str(row.get("chunk_id", "")),
            "text":       str(row.get("text", "")),
            "source":     str(row.get("source", "")),
            "drug_names": drug_names,
            "doc_id":     str(row.get("doc_id", "") or ""),
            "doc_title":  str(row.get("doc_title", "") or row.get("doc_id", "") or ""),
            "doc_url":    str(row.get("doc_url", "") or ""),
            "section":    str(row.get("section", "") or row.get("section_type", "") or ""),
        }

    def _load_chunks(self) -> list[dict]:
        try:
            return self._load_chunks_from_db()
        except Exception as exc:
            log.warning("PostgreSQL chunk load failed; falling back to FAISS metadata parquet: %s", exc)
            return self._load_faiss_metadata()

    def _load_chunks_from_db(self) -> list[dict]:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, text, source, drug_names,
                   doc_id, doc_title, doc_url, section
            FROM   rag_chunks
            ORDER  BY chunk_id
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        chunks = []
        for row in rows:
            chunks.append({
                "chunk_id":  row[0],
                "text":      row[1],
                "source":    row[2],
                "drug_names": row[3] or [],
                "doc_id":    row[4] or "",
                "doc_title": row[5] or "",
                "doc_url":   row[6] or "",
                "section":   row[7] or "",
            })
        log.info("Loaded %d chunks from PostgreSQL.", len(chunks))
        return chunks

    # ── Query expansion ───────────────────────────────────────────────────────

    def _expand_query(self, drug_names: list[str]) -> tuple[str, list[str]]:
        """
        Build an expanded query string by appending RxNorm synonyms.
        Returns (expanded_query_string, all_synonym_terms).
        """
        all_terms: list[str] = []
        for name in drug_names:
            all_terms.extend(self._expander.expand(name))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_terms: list[str] = []
        for t in all_terms:
            if t.lower() not in seen:
                unique_terms.append(t)
                seen.add(t.lower())
        expanded_query = " ".join(unique_terms)
        return expanded_query, unique_terms

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> list[tuple[str, float]]:
        vec     = self._embedder.encode([query])
        faiss.normalize_L2(vec)
        distances, indices = self._faiss_index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self._faiss_meta[idx]
            results.append((meta["chunk_id"], float(dist)))
        return results   # (chunk_id, cosine_similarity)

    # ── Sparse retrieval ──────────────────────────────────────────────────────

    def _sparse_search(self, query: str, top_k: int = TOP_K_SPARSE) -> list[tuple[str, float]]:
        hits = self._bm25.search(query, top_k)
        return [(self._bm25._chunks[i]["chunk_id"], score) for i, score in hits]

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        drug_names:  list[str] | None = None,
        top_k:       int              = TOP_K_FINAL,
        sources:     list[str] | None = None,   # filter by source
    ) -> RetrievalResult:
        """
        Parameters
        ----------
        query       : Free-text query, e.g. "warfarin aspirin bleeding"
        drug_names  : Optional list of drug names to expand via RxNorm
        top_k       : Number of chunks to return after fusion
        sources     : Optional filter, e.g. ["dailymed", "faers"]

        Returns
        -------
        RetrievalResult with ranked RetrievedChunk list
        """
        t0 = time.perf_counter()

        # 1. Query expansion
        expanded_terms: list[str] = []
        if drug_names:
            expanded_query, expanded_terms = self._expand_query(drug_names)
            full_query = f"{query} {expanded_query}"
        else:
            full_query = query

        log.info("Retrieval query: '%s'", full_query[:200])

        # 2. Dense search
        dense_hits   = self._dense_search(full_query)
        dense_ranked = [cid for cid, _ in dense_hits]

        # 3. Sparse search
        sparse_hits   = self._sparse_search(full_query)
        sparse_ranked = [cid for cid, _ in sparse_hits]

        # 4. RRF fusion
        fused = reciprocal_rank_fusion(dense_ranked, sparse_ranked)

        # 5. Source filtering (optional)
        if sources:
            sources_set = set(sources)
            fused = [(cid, score) for cid, score in fused
                     if self._chunk_map.get(cid, {}).get("source") in sources_set]

        # 6. Build dense/sparse rank maps for attribution
        dense_rank_map  = {cid: r for r, (cid, _) in enumerate(dense_hits,  start=1)}
        sparse_rank_map = {cid: r for r, (cid, _) in enumerate(sparse_hits, start=1)}

        # 7. Assemble RetrievedChunk objects
        chunks: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused[:top_k]:
            raw = self._chunk_map.get(chunk_id)
            if raw is None:
                continue
            chunks.append(RetrievedChunk(
                chunk_id   = chunk_id,
                text       = raw["text"],
                source     = raw["source"],
                drug_names = raw["drug_names"],
                rrf_score  = rrf_score,
                dense_rank = dense_rank_map.get(chunk_id, -1),
                sparse_rank= sparse_rank_map.get(chunk_id, -1),
                doc_id     = raw["doc_id"],
                doc_title  = raw["doc_title"],
                doc_url    = raw["doc_url"],
                section    = raw["section"],
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Retrieval done — %d chunks in %.1f ms  (dense=%d  sparse=%d  fused=%d)",
            len(chunks), latency_ms, len(dense_hits), len(sparse_hits), len(fused),
        )

        return RetrievalResult(
            query          = query,
            expanded_terms = expanded_terms,
            chunks         = chunks,
            latency_ms     = latency_ms,
            dense_hits     = len(dense_hits),
            sparse_hits    = len(sparse_hits),
        )

    def retrieve_for_pair(
        self,
        drug_a: str,
        drug_b: str,
        top_k:  int = TOP_K_FINAL,
    ) -> RetrievalResult:
        """
        Convenience method for a drug-pair query.
        Builds a structured query and uses both drugs for RxNorm expansion.
        """
        query = f"{drug_a} {drug_b} drug interaction adverse effect warning"
        return self.retrieve(query, drug_names=[drug_a, drug_b], top_k=top_k)


# ─────────────────────────── CLI smoke-test ──────────────────────────────────

if __name__ == "__main__":
    import sys
    drug_a = sys.argv[1] if len(sys.argv) > 1 else "warfarin"
    drug_b = sys.argv[2] if len(sys.argv) > 2 else "aspirin"

    retriever = HybridRetriever()
    result    = retriever.retrieve_for_pair(drug_a, drug_b)

    print(f"\n{'─'*70}")
    print(f"Query    : {result.query}")
    print(f"Expanded : {result.expanded_terms}")
    print(f"Latency  : {result.latency_ms:.1f} ms")
    print(f"Chunks   : {len(result.chunks)}")
    print(f"{'─'*70}\n")

    for i, chunk in enumerate(result.chunks[:5], start=1):
        print(f"[{i}] RRF={chunk.rrf_score:.4f}  source={chunk.source}")
        print(f"    doc  : {chunk.doc_title or chunk.doc_id}")
        print(f"    url  : {chunk.doc_url}")
        print(f"    text : {chunk.text[:200]} …")
        print()
