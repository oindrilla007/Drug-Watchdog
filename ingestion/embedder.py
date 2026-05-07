"""
ingestion/embedder.py
────────────────────────────────────────────────────────────────────────────────
Encodes all text chunks using BioBERT (dmis-lab/biobert-base-cased-v1.2) and
stores the resulting embeddings in two vector stores:

  1. FAISS  — fast approximate nearest-neighbour search (primary retriever)
  2. ChromaDB — persistent vector DB with metadata filtering (secondary)

Why two stores?
  FAISS is faster for bulk ANN search. ChromaDB supports metadata filtering
  (e.g. "only search DailyMed chunks with section=drug_interactions") which
  FAISS alone cannot do. The RAG pipeline in Phase 3 uses both in a hybrid
  retrieval strategy.

Model:
  dmis-lab/biobert-base-cased-v1.2  (HuggingFace Hub, ~440MB)
  Falls back to sentence-transformers/all-MiniLM-L6-v2 if BioBERT unavailable.

Outputs:
  data/embeddings/faiss_index.bin        ← FAISS flat L2 index
  data/embeddings/faiss_metadata.parquet ← chunk_id → metadata mapping
  data/embeddings/chroma_db/             ← ChromaDB persistent directory

Install requirements:
  pip install transformers torch faiss-cpu chromadb sentence-transformers

Usage:
  from ingestion.embedder import Embedder

  embedder = Embedder()
  embedder.load_model()
  embedder.embed_chunks("data/processed/chunks/all_chunks.parquet")
  embedder.save_faiss("data/embeddings/")
  embedder.save_chroma("data/embeddings/chroma_db/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
PRIMARY_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
FALLBACK_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH  = 512       # BioBERT hard limit
EMBEDDING_DIM   = 768       # BioBERT hidden size

# ── Batch sizes ───────────────────────────────────────────────────────────────
ENCODE_BATCH    = 32        # chunks per forward pass (tune down if OOM)
CHROMA_BATCH    = 500       # chunks per ChromaDB upsert call

# ── ChromaDB collection name ──────────────────────────────────────────────────
CHROMA_COLLECTION = "drug_watchdog_chunks"


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class NumpySearchIndex:
    """Small FAISS-like search wrapper backed by NumPy."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.ntotal = int(self.embeddings.shape[0])

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            empty_scores = np.empty((query_vectors.shape[0], 0), dtype=np.float32)
            empty_indices = np.empty((query_vectors.shape[0], 0), dtype=np.int64)
            return empty_scores, empty_indices

        query_vectors = query_vectors.astype(np.float32)
        scores = query_vectors @ self.embeddings.T
        top_k = min(top_k, self.ntotal)

        candidate_indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)

        order = np.argsort(-candidate_scores, axis=1)
        sorted_indices = np.take_along_axis(candidate_indices, order, axis=1)
        sorted_scores = np.take_along_axis(candidate_scores, order, axis=1)
        return sorted_scores, sorted_indices


class Embedder:
    """
    Loads a BioBERT model, encodes text chunks in batches,
    and persists embeddings to FAISS and ChromaDB.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name  = model_name or PRIMARY_MODEL
        self._tokenizer  = None
        self._model      = None
        self._device     = None
        self._embeddings: Optional[np.ndarray] = None
        self._metadata_df: Optional[pd.DataFrame] = None
        self._model_loaded = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> "Embedder":
        """
        Load BioBERT tokenizer + model from HuggingFace Hub.
        Falls back to MiniLM if BioBERT download fails.
        """
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self._device}")

        # Try primary model first
        for model_id in [self.model_name, FALLBACK_MODEL]:
            try:
                logger.info(f"Loading model: {model_id} …")
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model     = AutoModel.from_pretrained(model_id)
                self._model.to(self._device)
                self._model.eval()
                self.model_name = model_id
                self._model_loaded = True
                logger.info(f"✅ Model loaded: {model_id} on {self._device}")
                return self
            except Exception as exc:
                logger.warning(f"Could not load {model_id}: {exc}")

        raise RuntimeError(
            "Failed to load any embedding model. "
            "Run: pip install transformers torch sentence-transformers"
        )

    # ── Embedding pipeline ────────────────────────────────────────────────────

    def embed_chunks(
        self,
        chunks_path: str | Path,
        priority_only: bool = False,
    ) -> "Embedder":
        """
        Load chunks from Parquet, encode with BioBERT, store embeddings + metadata.

        priority_only: if True, only embed priority chunks (faster for dev runs)
        """
        if not self._model_loaded:
            raise RuntimeError("Call load_model() before embed_chunks()")

        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        df = pd.read_parquet(chunks_path)
        if priority_only:
            df = df[df["priority"] == True].copy()
            logger.info(f"Priority-only mode: {len(df):,} chunks selected")

        logger.info(f"Embedding {len(df):,} chunks with {self.model_name} …")

        texts = df["text"].tolist()
        embeddings = self._encode_batched(texts)

        self._embeddings   = embeddings
        self._metadata_df  = df.reset_index(drop=True)

        logger.info(
            f"✅ Embedding complete — "
            f"shape: {embeddings.shape} | "
            f"dtype: {embeddings.dtype}"
        )
        return self

    def _encode_batched(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts in batches using mean pooling over the last hidden state.
        Returns float32 numpy array of shape (n_texts, EMBEDDING_DIM).
        """
        import torch

        all_embeddings = []
        total_batches  = (len(texts) + ENCODE_BATCH - 1) // ENCODE_BATCH

        for batch_idx in range(0, len(texts), ENCODE_BATCH):
            batch_texts = texts[batch_idx: batch_idx + ENCODE_BATCH]
            batch_num   = batch_idx // ENCODE_BATCH + 1

            if batch_num % 50 == 0 or batch_num == 1:
                logger.info(f"  Encoding batch {batch_num}/{total_batches} …")

            # Tokenise
            encoded = self._tokenizer(
                batch_texts,
                padding     = True,
                truncation  = True,
                max_length  = MAX_SEQ_LENGTH,
                return_tensors = "pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            # Forward pass — no gradient needed
            with torch.no_grad():
                outputs = self._model(**encoded)

            # Mean pooling over token dimension, ignoring padding tokens
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state   # (batch, seq_len, hidden)
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                              .expand(token_embeddings.size())
                              .float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask       = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled         = (sum_embeddings / sum_mask).cpu().numpy()

            all_embeddings.append(pooled)

        return np.vstack(all_embeddings).astype(np.float32)

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def save_faiss(self, output_dir: str | Path = "data/embeddings/") -> None:
        """
        Build a FAISS flat index when available, or save a NumPy fallback
        that provides the same search interface on platforms without FAISS.
        """
        self._check_embedded()

        try:
            import faiss
        except ImportError:
            faiss = None

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n, dim = self._embeddings.shape
        logger.info(f"Building vector index - {n:,} vectors, dim={dim} ...")

        # Normalise vectors for cosine similarity (dot product on normalised vectors)
        embeddings_norm = self._normalise(self._embeddings)

        index_path = out / "faiss_index.bin"
        if faiss is not None:
            # Flat index - exact search, no approximation errors.
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings_norm)
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index -> {index_path} ({index.ntotal:,} vectors)")
        else:
            logger.warning("FAISS is not installed; saving a NumPy fallback index instead.")

        np.save(str(out / "faiss_embeddings.npy"), embeddings_norm)

        # Metadata mapping: row ID -> chunk metadata
        meta_path = out / "faiss_metadata.parquet"
        self._metadata_df.to_parquet(meta_path, index=True)
        logger.info(f"Saved metadata -> {meta_path}")

        np.save(str(out / "embeddings.npy"), self._embeddings)

    def load_faiss(self, embeddings_dir: str | Path = "data/embeddings/") -> tuple[Any, pd.DataFrame]:
        """
        Load a saved FAISS index or the NumPy fallback + metadata from disk.
        Returns (index_like_object, metadata_df).
        """
        emb_dir    = Path(embeddings_dir)
        index_path = emb_dir / "faiss_index.bin"
        fallback_path = emb_dir / "faiss_embeddings.npy"
        meta_path  = emb_dir / "faiss_metadata.parquet"

        if not meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {meta_path}")

        try:
            import faiss
        except ImportError:
            faiss = None

        if faiss is not None and index_path.exists():
            index = faiss.read_index(str(index_path))
            metadata_df = pd.read_parquet(meta_path)
            logger.info(f"Loaded FAISS index - {index.ntotal:,} vectors")
            return index, metadata_df

        if fallback_path.exists():
            embeddings = np.load(str(fallback_path))
            index = NumpySearchIndex(embeddings)
            metadata_df = pd.read_parquet(meta_path)
            logger.info(f"Loaded NumPy fallback index - {index.ntotal:,} vectors")
            return index, metadata_df

        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    def query_faiss(
        self,
        query_text: str,
        index,
        metadata_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Encode a query string and search the FAISS index or NumPy fallback.
        Returns top-k results as a DataFrame with similarity scores.
        Used for smoke-testing the index after building it.
        """
        if not self._model_loaded:
            self.load_model()

        query_emb = self._encode_batched([query_text])   # (1, dim)
        query_norm = self._normalise(query_emb)

        scores, indices = index.search(query_norm, top_k)
        if scores.size == 0 or indices.size == 0:
            return pd.DataFrame()

        scores = scores[0]
        indices = indices[0]

        # Filter out -1 indices (FAISS returns -1 for empty slots)
        valid = [(s, i) for s, i in zip(scores, indices) if i >= 0]
        if not valid:
            return pd.DataFrame()

        result_scores, result_indices = zip(*valid)
        results = metadata_df.iloc[list(result_indices)].copy()
        results["similarity_score"] = list(result_scores)
        results = results.sort_values("similarity_score", ascending=False)
        return results

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def save_chroma(
        self,
        chroma_dir: str | Path = "data/embeddings/chroma_db/",
    ) -> None:
        """
        Upsert all chunks + embeddings into a ChromaDB persistent collection.
        ChromaDB enables metadata filtering at query time (e.g. source, section).
        """
        self._check_embedded()

        try:
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        chroma_dir = Path(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        client     = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name     = CHROMA_COLLECTION,
            metadata = {"hnsw:space": "cosine"},
        )

        df   = self._metadata_df
        embs = self._embeddings
        n    = len(df)

        logger.info(f"Upserting {n:,} chunks into ChromaDB collection '{CHROMA_COLLECTION}' …")
        t0 = time.time()

        for start in range(0, n, CHROMA_BATCH):
            end   = min(start + CHROMA_BATCH, n)
            batch = df.iloc[start:end]
            batch_embs = embs[start:end]

            ids        = batch["chunk_id"].tolist()
            documents  = batch["text"].tolist()
            metadatas  = [
                {
                    "source":       str(row.get("source", "")),
                    "doc_id":       str(row.get("doc_id", "")),
                    "drug_names":   str(row.get("drug_names", "")),
                    "section_type": str(row.get("section_type", "")),
                    "priority":     bool(row.get("priority", False)),
                    "char_count":   int(row.get("char_count", 0)),
                }
                for _, row in batch.iterrows()
            ]

            collection.upsert(
                ids        = ids,
                embeddings = batch_embs.tolist(),
                documents  = documents,
                metadatas  = metadatas,
            )

            if (start // CHROMA_BATCH) % 10 == 0:
                logger.info(f"  … upserted {end:,}/{n:,}")

        elapsed = time.time() - t0
        logger.info(
            f"✅ ChromaDB upsert complete — "
            f"{collection.count():,} total documents | {elapsed:.1f}s"
        )

    def query_chroma(
        self,
        query_text: str,
        chroma_dir: str | Path = "data/embeddings/chroma_db/",
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query ChromaDB with optional metadata filter.
        where example: {"source": "dailymed", "section_type": "drug_interactions"}
        Returns list of {text, metadata, distance} dicts.
        Used for smoke-testing and as fallback in the RAG pipeline.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        if not self._model_loaded:
            self.load_model()

        query_emb = self._encode_batched([query_text])[0].tolist()

        client     = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(CHROMA_COLLECTION)

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results":        top_k,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":       doc,
                "metadata":   meta,
                "distance":   dist,
                "similarity": 1 - dist,   # ChromaDB cosine returns distance
            })
        return output

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_embedded(self) -> None:
        if self._embeddings is None or self._metadata_df is None:
            raise RuntimeError("Call embed_chunks() before saving.")

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise each row vector for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return (vectors / norms).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Embed drug knowledge base chunks with BioBERT")
    ap.add_argument("--chunks",        default="data/processed/chunks/all_chunks.parquet")
    ap.add_argument("--emb-dir",       default="data/embeddings/")
    ap.add_argument("--chroma-dir",    default="data/embeddings/chroma_db/")
    ap.add_argument("--model",         default=PRIMARY_MODEL)
    ap.add_argument("--priority-only", action="store_true",
                    help="Only embed priority chunks (faster for dev)")
    ap.add_argument("--query",         default=None,
                    help="Smoke-test query after building index")
    args = ap.parse_args()

    embedder = Embedder(model_name=args.model)
    embedder.load_model()
    embedder.embed_chunks(args.chunks, priority_only=args.priority_only)
    embedder.save_faiss(args.emb_dir)
    embedder.save_chroma(args.chroma_dir)

    # Smoke test
    query = args.query or "warfarin aspirin bleeding risk"
    logger.info(f"\n── Smoke test query: '{query}' ──────────────────────────")

    index, meta_df = embedder.load_faiss(args.emb_dir)
    faiss_results  = embedder.query_faiss(query, index, meta_df, top_k=5)

    print("\nFAISS top-5 results:")
    for _, row in faiss_results.iterrows():
        print(f"\n  [{row['source']} | {row['section_type']} | score={row['similarity_score']:.3f}]")
        print(f"  {row['text'][:200]} …")

    chroma_results = embedder.query_chroma(
        query,
        chroma_dir = args.chroma_dir,
        top_k      = 5,
        where      = {"section_type": "drug_interactions"},
    )

    print("\nChromaDB top-5 (drug_interactions only):")
    for r in chroma_results:
        print(f"\n  [{r['metadata']['source']} | sim={r['similarity']:.3f}]")
        print(f"  {r['text'][:200]} …")
