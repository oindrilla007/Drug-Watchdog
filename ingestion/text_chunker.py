"""
ingestion/text_chunker.py
────────────────────────────────────────────────────────────────────────────────
Unified text chunking pipeline that takes processed text from ALL sources
(DrugBank, DailyMed, FAERS, PubMed) and produces a single, consistently
formatted chunk table ready for embedding and vector store ingestion.

Why a dedicated chunker?
  Each source parser does its own basic splitting, but this module applies
  a unified strategy across all sources with:
    - Consistent chunk sizes tuned for BioBERT (512 token limit)
    - Rich metadata attached to every chunk (source, drug names, section, priority)
    - Overlap between chunks to preserve context at boundaries
    - Deduplication of near-identical chunks across sources
    - Quality filtering (removes chunks that are too short or too noisy)

Output:
  data/processed/chunks/all_chunks.parquet   ← unified chunk table
  data/processed/chunks/priority_chunks.parquet  ← high-signal chunks only

Usage:
  from ingestion.text_chunker import TextChunker

  chunker = TextChunker()
  chunker.add_drugbank("data/processed/drugbank_drugs.parquet",
                       "data/processed/drug_pairs.parquet")
  chunker.add_dailymed("data/processed/dailymed_chunks.parquet")
  chunker.add_faers("data/processed/faers_drug_events.parquet")
  chunker.add_pubmed("data/processed/chunks/pubmed_chunks.parquet")
  chunker.run()
  chunker.save("data/processed/chunks/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Chunking parameters ───────────────────────────────────────────────────────
# BioBERT max tokens = 512. At ~3.5 chars/token, 1400 chars ≈ 400 tokens,
# leaving headroom for the [CLS]/[SEP] tokens and query prefix at retrieval.
TARGET_CHUNK_CHARS  = 1_000    # aim for chunks around this size
MAX_CHUNK_CHARS     = 1_400    # hard ceiling
MIN_CHUNK_CHARS     = 80       # discard chunks shorter than this
OVERLAP_SENTENCES   = 1        # sentences to carry over between chunks

# ── Priority section types (from DailyMed + DrugBank) ────────────────────────
PRIORITY_SECTIONS = {
    "boxed_warning",
    "warnings_and_precautions",
    "drug_interactions",
    "contraindications",
    "adverse_reactions",
    "interaction",       # DrugBank interaction description
    "cyp450",            # CYP450 relationship text
}

# ── Sources ───────────────────────────────────────────────────────────────────
SOURCE_DRUGBANK  = "drugbank"
SOURCE_DAILYMED  = "dailymed"
SOURCE_FAERS     = "faers"
SOURCE_PUBMED    = "pubmed"


# ─────────────────────────────────────────────────────────────────────────────
# Chunk data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:       str          # UUID
    source:         str          # drugbank | dailymed | faers | pubmed
    doc_id:         str          # source document identifier
    drug_names:     list[str]    # drugs mentioned in this chunk
    section_type:   str          # e.g. drug_interactions, boxed_warning
    text:           str          # the actual text
    char_count:     int
    token_estimate: int          # rough token count (chars / 3.5)
    priority:       bool         # True = high-signal safety content
    content_hash:   str          # SHA-1 of text for deduplication


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Ingests processed DataFrames from all sources, applies unified
    chunking strategy, deduplicates, and saves a single chunk table.
    """

    def __init__(self) -> None:
        self._raw_chunks: list[Chunk] = []
        self._final_chunks: list[Chunk] = []
        self._seen_hashes: set[str] = set()
        self._ran = False

    # ── Source ingestion methods ──────────────────────────────────────────────

    def add_drugbank(
        self,
        drugs_path: str | Path,
        interactions_path: Optional[str | Path] = None,
        cyp450_path: Optional[str | Path] = None,
    ) -> "TextChunker":
        """
        Build chunks from DrugBank drug descriptions and interaction texts.
        """
        drugs_path = Path(drugs_path)
        if not drugs_path.exists():
            logger.warning(f"DrugBank drugs file not found: {drugs_path}")
            return self

        df = pd.read_parquet(drugs_path)
        count = 0

        # Drug description / mechanism / pharmacodynamics chunks
        text_cols = {
            "description":      "description",
            "mechanism":        "mechanism_of_action",
            "pharmacodynamics": "pharmacodynamics",
            "indication":       "indication",
        }
        for col, section in text_cols.items():
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                text = str(row.get(col, "")).strip()
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                drug_name = str(row.get("name", "")).strip()
                for chunk_text in self._split(text):
                    self._raw_chunks.append(self._make_chunk(
                        text        = chunk_text,
                        source      = SOURCE_DRUGBANK,
                        doc_id      = str(row.get("drugbank_id", "")),
                        drug_names  = [drug_name] if drug_name else [],
                        section     = section,
                        priority    = section in PRIORITY_SECTIONS,
                    ))
                    count += 1

        logger.info(f"  DrugBank descriptions: {count:,} raw chunks")

        # Interaction description chunks (high priority)
        if interactions_path and Path(interactions_path).exists():
            ix_df = pd.read_parquet(interactions_path)
            ix_count = 0
            for _, row in ix_df.iterrows():
                desc = str(row.get("description", "")).strip()
                if len(desc) < MIN_CHUNK_CHARS:
                    continue
                drug_a = str(row.get("drug_a_name", ""))
                drug_b = str(row.get("drug_b_name", ""))
                # Prepend drug names so the chunk is self-contained for RAG
                full_text = f"{drug_a} and {drug_b} interaction: {desc}"
                for chunk_text in self._split(full_text):
                    self._raw_chunks.append(self._make_chunk(
                        text       = chunk_text,
                        source     = SOURCE_DRUGBANK,
                        doc_id     = f"{row.get('drug_a_id','')}-{row.get('drug_b_id','')}",
                        drug_names = [drug_a, drug_b],
                        section    = "interaction",
                        priority   = True,
                    ))
                    ix_count += 1
            logger.info(f"  DrugBank interactions: {ix_count:,} raw chunks")

        # CYP450 relationship chunks
        if cyp450_path and Path(cyp450_path).exists():
            cyp_df = pd.read_parquet(cyp450_path)
            cyp_count = 0
            # Group by enzyme for compact, informative chunks
            for enzyme, grp in cyp_df.groupby("enzyme"):
                for role, role_grp in grp.groupby("role"):
                    drug_list = role_grp["drug_name"].tolist()
                    # Chunk in batches of 15 drugs per chunk
                    for i in range(0, len(drug_list), 15):
                        batch = drug_list[i:i+15]
                        text = (
                            f"{enzyme} {role}s include: {', '.join(batch)}. "
                            f"These drugs are {role}s of the {enzyme} cytochrome P450 enzyme, "
                            f"which may affect their metabolism and plasma concentrations."
                        )
                        self._raw_chunks.append(self._make_chunk(
                            text       = text,
                            source     = SOURCE_DRUGBANK,
                            doc_id     = f"cyp450_{enzyme}_{role}",
                            drug_names = batch,
                            section    = "cyp450",
                            priority   = True,
                        ))
                        cyp_count += 1
            logger.info(f"  DrugBank CYP450: {cyp_count:,} raw chunks")

        return self

    def add_dailymed(self, chunks_path: str | Path) -> "TextChunker":
        """Ingest pre-chunked DailyMed text from dailymed_parser output."""
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            logger.warning(f"DailyMed chunks file not found: {chunks_path}")
            return self

        df = pd.read_parquet(chunks_path)
        count = 0

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # Re-chunk if DailyMed chunk exceeds our target size
            for chunk_text in self._split(text):
                section = str(row.get("section_type", "general"))
                self._raw_chunks.append(self._make_chunk(
                    text       = chunk_text,
                    source     = SOURCE_DAILYMED,
                    doc_id     = str(row.get("doc_id", "")),
                    drug_names = [str(row.get("drug_name", ""))],
                    section    = section,
                    priority   = section in PRIORITY_SECTIONS,
                ))
                count += 1

        logger.info(f"  DailyMed: {count:,} raw chunks")
        return self

    def add_faers(self, drug_events_path: str | Path) -> "TextChunker":
        """
        Convert FAERS drug-event aggregates into narrative text chunks.
        Each chunk describes a drug's top adverse events with report counts.
        """
        drug_events_path = Path(drug_events_path)
        if not drug_events_path.exists():
            logger.warning(f"FAERS drug events file not found: {drug_events_path}")
            return self

        df = pd.read_parquet(drug_events_path)
        count = 0

        # Group by drug and build one narrative chunk per drug
        for drug_name, grp in df.groupby("drugname"):
            # Top 20 reactions by report count
            top = grp.nlargest(20, "report_count")

            # Serious reactions (severity ≥ 3) get their own chunk
            serious = grp[grp["max_severity"] >= 3].nlargest(10, "report_count")
            if not serious.empty:
                reactions_str = "; ".join(
                    f"{row['reaction_term']} ({int(row['report_count'])} reports)"
                    for _, row in serious.iterrows()
                )
                text = (
                    f"FDA FAERS data for {drug_name}: serious adverse events reported include "
                    f"{reactions_str}. These events were associated with high severity outcomes "
                    f"including hospitalization or death."
                )
                self._raw_chunks.append(self._make_chunk(
                    text       = text,
                    source     = SOURCE_FAERS,
                    doc_id     = f"faers_{drug_name}_serious",
                    drug_names = [str(drug_name)],
                    section    = "adverse_reactions",
                    priority   = True,
                ))
                count += 1

            # General top-reactions chunk
            reactions_str = "; ".join(
                f"{row['reaction_term']} ({int(row['report_count'])} reports)"
                for _, row in top.iterrows()
            )
            text = (
                f"FDA FAERS adverse event data for {drug_name}: the most commonly reported "
                f"adverse reactions include {reactions_str}."
            )
            self._raw_chunks.append(self._make_chunk(
                text       = text,
                source     = SOURCE_FAERS,
                doc_id     = f"faers_{drug_name}_general",
                drug_names = [str(drug_name)],
                section    = "adverse_reactions",
                priority   = False,
            ))
            count += 1

        logger.info(f"  FAERS: {count:,} raw chunks")
        return self

    def add_pubmed(self, pubmed_chunks_path: str | Path) -> "TextChunker":
        """Ingest pre-chunked PubMed abstract text from pubmed_fetcher output."""
        pubmed_chunks_path = Path(pubmed_chunks_path)
        if not pubmed_chunks_path.exists():
            logger.warning(f"PubMed chunks file not found: {pubmed_chunks_path}")
            return self

        df = pd.read_parquet(pubmed_chunks_path)
        count = 0

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            for chunk_text in self._split(text):
                self._raw_chunks.append(self._make_chunk(
                    text       = chunk_text,
                    source     = SOURCE_PUBMED,
                    doc_id     = str(row.get("pmid", "")),
                    drug_names = self._extract_drug_names_from_query(
                                    str(row.get("drug_query", ""))),
                    section    = "literature",
                    priority   = False,
                ))
                count += 1

        logger.info(f"  PubMed: {count:,} raw chunks")
        return self

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "TextChunker":
        """
        Deduplicate and quality-filter all raw chunks.
        Must be called after all add_*() calls.
        """
        logger.info(f"Running deduplication + quality filter on {len(self._raw_chunks):,} raw chunks …")

        kept = skipped_short = skipped_dup = 0

        for chunk in self._raw_chunks:
            # Quality filter: too short
            if chunk.char_count < MIN_CHUNK_CHARS:
                skipped_short += 1
                continue

            # Deduplication: exact content hash
            if chunk.content_hash in self._seen_hashes:
                skipped_dup += 1
                continue

            self._seen_hashes.add(chunk.content_hash)
            self._final_chunks.append(chunk)
            kept += 1

        self._ran = True
        logger.info(
            f"✅ Chunking complete — "
            f"kept: {kept:,} | "
            f"deduped: {skipped_dup:,} | "
            f"too short: {skipped_short:,}"
        )
        return self

    def get_chunks_df(self) -> pd.DataFrame:
        self._check_ran()
        rows = [
            {
                "chunk_id":       c.chunk_id,
                "source":         c.source,
                "doc_id":         c.doc_id,
                "drug_names":     "|".join(c.drug_names),
                "section_type":   c.section_type,
                "text":           c.text,
                "char_count":     c.char_count,
                "token_estimate": c.token_estimate,
                "priority":       c.priority,
                "content_hash":   c.content_hash,
            }
            for c in self._final_chunks
        ]
        return pd.DataFrame(rows)

    def save(self, output_dir: str | Path = "data/processed/chunks/") -> None:
        self._check_ran()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = self.get_chunks_df()

        all_path = out / "all_chunks.parquet"
        df.to_parquet(all_path, index=False)
        logger.info(f"💾 Saved {len(df):,} chunks → {all_path}")

        priority_df = df[df["priority"]].copy()
        pri_path = out / "priority_chunks.parquet"
        priority_df.to_parquet(pri_path, index=False)
        logger.info(f"💾 Saved {len(priority_df):,} priority chunks → {pri_path}")

        # Per-source breakdown saved separately (useful for debugging)
        for source in df["source"].unique():
            source_df = df[df["source"] == source]
            source_path = out / f"{source}_chunks.parquet"
            source_df.to_parquet(source_path, index=False)
            logger.info(f"   └─ {source}: {len(source_df):,} chunks → {source_path}")

    def stats(self) -> None:
        """Print a summary of the final chunk table."""
        self._check_ran()
        df = self.get_chunks_df()
        print("\n── Chunk statistics ───────────────────────────────────────")
        print(f"Total chunks:     {len(df):,}")
        print(f"Priority chunks:  {df['priority'].sum():,} ({df['priority'].mean()*100:.1f}%)")
        print(f"\nBy source:")
        print(df["source"].value_counts().to_string())
        print(f"\nBy section type (top 10):")
        print(df["section_type"].value_counts().head(10).to_string())
        print(f"\nChar count distribution:")
        print(df["char_count"].describe().round(0).to_string())
        print(f"\nToken estimate distribution:")
        print(df["token_estimate"].describe().round(0).to_string())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("Call .run() before accessing results.")

    def _make_chunk(
        self,
        text:       str,
        source:     str,
        doc_id:     str,
        drug_names: list[str],
        section:    str,
        priority:   bool,
    ) -> Chunk:
        cleaned = self._clean(text)
        return Chunk(
            chunk_id       = str(uuid.uuid4()),
            source         = source,
            doc_id         = doc_id,
            drug_names     = [d for d in drug_names if d and d.strip()],
            section_type   = section,
            text           = cleaned,
            char_count     = len(cleaned),
            token_estimate = max(1, len(cleaned) // 4),   # ~4 chars/token for biomedical
            priority       = priority,
            content_hash   = hashlib.sha1(cleaned.lower().encode()).hexdigest(),
        )

    def _split(self, text: str) -> list[str]:
        """
        Split text into chunks of TARGET_CHUNK_CHARS with OVERLAP_SENTENCES
        sentence overlap. Uses sentence-boundary splitting (no NLTK needed).
        """
        text = self._clean(text)
        if len(text) <= MAX_CHUNK_CHARS:
            return [text] if len(text) >= MIN_CHUNK_CHARS else []

        sentences = self._sentence_split(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > TARGET_CHUNK_CHARS and current:
                chunk_text = " ".join(current)
                if len(chunk_text) >= MIN_CHUNK_CHARS:
                    chunks.append(chunk_text)
                # Overlap: carry last N sentences into next chunk
                current = current[-OVERLAP_SENTENCES:] + [sent]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sent)
                current_len += len(sent)

        if current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """
        Split text into sentences.
        Handles abbreviations like 'Dr.', 'mg.', 'U.S.' to avoid false splits.
        """
        # Protect common abbreviations
        text = re.sub(r"\b(Dr|Mr|Mrs|Prof|mg|mcg|vs|approx|e\.g|i\.e|U\.S)\.", r"\1<PERIOD>", text)
        # Split on sentence-ending punctuation followed by whitespace + capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        # Restore protected periods
        return [s.replace("<PERIOD>", ".") for s in sentences if s.strip()]

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and remove control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_drug_names_from_query(query: str) -> list[str]:
        """
        Pull drug names from a PubMed query string.
        e.g. '"warfarin"[Title] AND "aspirin"[Title]' → ['warfarin', 'aspirin']
        """
        return re.findall(r'"([^"]+)"', query)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Unified text chunker for all data sources")
    ap.add_argument("--drugbank-drugs",    default="data/processed/drugbank_drugs.parquet")
    ap.add_argument("--drugbank-pairs",    default="data/processed/drug_pairs.parquet")
    ap.add_argument("--drugbank-cyp450",   default="data/processed/cyp450_relationships.parquet")
    ap.add_argument("--dailymed-chunks",   default="data/processed/dailymed_chunks.parquet")
    ap.add_argument("--faers-events",      default="data/processed/faers_drug_events.parquet")
    ap.add_argument("--pubmed-chunks",     default="data/processed/chunks/pubmed_chunks.parquet")
    ap.add_argument("--out",               default="data/processed/chunks/")
    args = ap.parse_args()

    chunker = TextChunker()

    chunker.add_drugbank(
        drugs_path        = args.drugbank_drugs,
        interactions_path = args.drugbank_pairs,
        cyp450_path       = args.drugbank_cyp450,
    )
    chunker.add_dailymed(args.dailymed_chunks)
    chunker.add_faers(args.faers_events)
    chunker.add_pubmed(args.pubmed_chunks)

    chunker.run()
    chunker.stats()
    chunker.save(args.out)