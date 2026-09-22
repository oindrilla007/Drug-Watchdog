"""
citation_tracker.py  ·  Drug Watchdog Phase 3
================================================
Citation and provenance utilities for the RAG evidence pipeline.

The retriever returns source metadata with every evidence chunk. This module
turns those chunks into stable citation keys such as [FDA-1], [FAERS-1],
[PUB-1], and [DB-1], deduplicates repeated sources, fills source URLs when
possible, and formats references for the context assembler / explanation agent.

It is intentionally lightweight: no model loading, no database connection, and
no network calls. It can be run as a standalone smoke test:

    python citation_tracker.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


SOURCE_PREFIXES = {
    "dailymed": "FDA",
    "fda": "FDA",
    "faers": "FAERS",
    "pubmed": "PUB",
    "pmid": "PUB",
    "drugbank": "DB",
    "rxnorm": "RXN",
}

SOURCE_LABELS = {
    "dailymed": "FDA DailyMed",
    "fda": "FDA DailyMed",
    "faers": "FDA FAERS",
    "pubmed": "PubMed",
    "pmid": "PubMed",
    "drugbank": "DrugBank",
    "rxnorm": "RxNorm",
}

HIGH_VALUE_SECTIONS = {
    "BOXED WARNING": 1,
    "CONTRAINDICATIONS": 2,
    "WARNINGS AND PRECAUTIONS": 3,
    "DRUG INTERACTIONS": 4,
    "ADVERSE REACTIONS": 5,
}

PMID_RE = re.compile(r"\b(?:pmid[:\s]*)?(\d{6,9})\b", re.IGNORECASE)
CITATION_KEY_RE = re.compile(r"\[(FDA|FAERS|PUB|DB|RXN|SRC)-\d+\]")


@dataclass
class Citation:
    """One source used as evidence in an interaction explanation."""

    key: str
    source: str
    doc_id: str = ""
    doc_title: str = ""
    doc_url: str = ""
    section: str = ""
    snippet: str = ""
    chunk_id: str = ""
    drug_names: list[str] = field(default_factory=list)
    evidence_rank: int | None = None
    relevance_score: float | None = None
    retrieval_score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def display_source(self) -> str:
        return SOURCE_LABELS.get(normalize_source(self.source), self.source.upper() or "Source")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["display_source"] = self.display_source
        return data


@dataclass
class CitationValidation:
    """Result of checking whether generated text cites known references."""

    used_keys: list[str]
    known_keys: list[str]
    missing_keys: list[str]
    unused_keys: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_keys


def normalize_source(source: str | None) -> str:
    """Normalize source labels used across ingestion/retrieval modules."""
    value = (source or "").strip().lower()
    if value in {"daily_med", "daily-med", "label", "fda_label"}:
        return "dailymed"
    if value in {"pmc", "pub_med"}:
        return "pubmed"
    return value or "unknown"


def source_prefix(source: str | None) -> str:
    return SOURCE_PREFIXES.get(normalize_source(source), "SRC")


def clean_snippet(text: str | None, max_chars: int = 280) -> str:
    """Compact whitespace and trim evidence text for citations."""
    snippet = re.sub(r"\s+", " ", (text or "")).strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 1].rstrip() + "..."


def infer_doc_url(source: str, doc_id: str, existing_url: str = "") -> str:
    """Fill canonical public source URLs when retriever metadata omits them."""
    if existing_url:
        return existing_url

    src = normalize_source(source)
    doc = (doc_id or "").strip()
    if not doc:
        return ""

    if src == "pubmed":
        match = PMID_RE.search(doc)
        pmid = match.group(1) if match else doc
        if pmid.isdigit():
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    if src == "dailymed":
        return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={doc}"

    if src == "drugbank":
        return f"https://go.drugbank.com/drugs/{doc}" if doc.upper().startswith("DB") else ""

    return ""


def stable_source_id(
    source: str,
    doc_id: str = "",
    section: str = "",
    chunk_id: str = "",
    text: str = "",
) -> str:
    """
    Build a dedupe identity.

    Prefer document + section so multiple retrieved chunks from the same label
    section map to one citation. Fall back to chunk_id, then text hash.
    """
    src = normalize_source(source)
    doc = (doc_id or "").strip().lower()
    sec = (section or "").strip().upper()
    if doc:
        return f"{src}|{doc}|{sec}"
    if chunk_id:
        return f"{src}|chunk|{chunk_id}"
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]
    return f"{src}|text|{digest}"


class CitationTracker:
    """
    Assigns stable citation keys and formats references for RAG outputs.

    Accepted inputs:
      - retriever.RetrievedChunk
      - reranker.RankedChunk
      - dict evidence blocks from ContextAssembler
      - plain metadata via add(...)
    """

    def __init__(self) -> None:
        self._citations: list[Citation] = []
        self._by_identity: dict[str, Citation] = {}
        self._counters: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._citations)

    def __iter__(self):
        return iter(self._citations)

    @property
    def citations(self) -> list[Citation]:
        return list(self._citations)

    def add(
        self,
        evidence: Any | None = None,
        *,
        source: str = "",
        doc_id: str = "",
        doc_title: str = "",
        doc_url: str = "",
        section: str = "",
        snippet: str = "",
        text: str = "",
        chunk_id: str = "",
        drug_names: Iterable[str] | None = None,
        evidence_rank: int | None = None,
        relevance_score: float | None = None,
        retrieval_score: float | None = None,
    ) -> str:
        """Register one evidence source and return its citation key."""
        if evidence is not None:
            if hasattr(evidence, "chunk") and hasattr(evidence, "relevance_score"):
                return self.add_ranked_chunk(evidence)
            if isinstance(evidence, dict):
                return self.add_evidence_block(evidence)
            return self.add_chunk(evidence)

        src = normalize_source(source)
        identity = stable_source_id(src, doc_id, section, chunk_id, text or snippet)

        existing = self._by_identity.get(identity)
        if existing:
            self._merge(existing, snippet=snippet or text, relevance_score=relevance_score, evidence_rank=evidence_rank)
            return existing.key

        prefix = source_prefix(src)
        number = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = number
        key = f"[{prefix}-{number}]"

        citation = Citation(
            key=key,
            source=src,
            doc_id=doc_id or "",
            doc_title=doc_title or "",
            doc_url=infer_doc_url(src, doc_id or "", doc_url or ""),
            section=(section or "").strip(),
            snippet=clean_snippet(snippet or text),
            chunk_id=chunk_id or "",
            drug_names=list(drug_names or []),
            evidence_rank=evidence_rank,
            relevance_score=relevance_score,
            retrieval_score=retrieval_score,
        )
        self._citations.append(citation)
        self._by_identity[identity] = citation
        return key

    def add_chunk(self, chunk: Any, *, evidence_rank: int | None = None, relevance_score: float | None = None) -> str:
        """Register a retriever.RetrievedChunk-like object."""
        return self.add(
            source=getattr(chunk, "source", ""),
            doc_id=getattr(chunk, "doc_id", ""),
            doc_title=getattr(chunk, "doc_title", ""),
            doc_url=getattr(chunk, "doc_url", ""),
            section=getattr(chunk, "section", ""),
            text=getattr(chunk, "text", ""),
            chunk_id=getattr(chunk, "chunk_id", ""),
            drug_names=getattr(chunk, "drug_names", []) or [],
            evidence_rank=evidence_rank,
            relevance_score=relevance_score,
            retrieval_score=getattr(chunk, "rrf_score", None),
        )

    def add_ranked_chunk(self, ranked_chunk: Any) -> str:
        """Register a reranker.RankedChunk-like object."""
        return self.add_chunk(
            ranked_chunk.chunk,
            evidence_rank=getattr(ranked_chunk, "rerank_position", None),
            relevance_score=getattr(ranked_chunk, "relevance_score", None),
        )

    def add_evidence_block(self, block: dict[str, Any], *, text_key: str = "text") -> str:
        """Register an assembled evidence block dict."""
        return self.add(
            source=block.get("source", ""),
            doc_id=block.get("doc_id", ""),
            doc_title=block.get("doc_title", ""),
            doc_url=block.get("doc_url", ""),
            section=block.get("section", ""),
            text=block.get(text_key, ""),
            chunk_id=block.get("chunk_id", ""),
            drug_names=block.get("drug_names", []) or [],
            evidence_rank=block.get("rank") or block.get("evidence_rank"),
            relevance_score=block.get("relevance_score"),
            retrieval_score=block.get("rrf_score"),
        )

    def add_reranked_result(self, reranked_result: Any, *, max_chunks: int | None = None) -> list[str]:
        """Register all chunks from a reranker.RerankedResult-like object."""
        ranked = list(getattr(reranked_result, "ranked_chunks", []) or [])
        if max_chunks is not None:
            ranked = ranked[:max_chunks]
        return [self.add_ranked_chunk(item) for item in ranked]

    def key_for_chunk(self, chunk: Any) -> str | None:
        identity = stable_source_id(
            getattr(chunk, "source", ""),
            getattr(chunk, "doc_id", ""),
            getattr(chunk, "section", ""),
            getattr(chunk, "chunk_id", ""),
            getattr(chunk, "text", ""),
        )
        citation = self._by_identity.get(identity)
        return citation.key if citation else None

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._citations]

    def to_key_map(self) -> dict[str, dict[str, Any]]:
        return {c.key: c.to_dict() for c in self._citations}

    def by_source(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for citation in self._citations:
            grouped.setdefault(citation.source, []).append(citation.to_dict())
        return grouped

    def format_references(self, *, include_snippets: bool = False) -> str:
        """Format references for prompt_context or a CLI display."""
        if not self._citations:
            return "REFERENCES:\n  (no citations registered)"

        lines = ["REFERENCES:"]
        for citation in self._citations:
            title = citation.doc_title or citation.doc_id or "Untitled source"
            section = f" · {citation.section}" if citation.section else ""
            url = f" -> {citation.doc_url}" if citation.doc_url else ""
            lines.append(f"  {citation.key} {citation.display_source} · {title}{section}{url}")
            if include_snippets and citation.snippet:
                lines.append(f"      {citation.snippet}")
        return "\n".join(lines)

    def format_inline_evidence(self) -> str:
        """Compact evidence list suitable for showing before LLM generation."""
        lines: list[str] = []
        for citation in self._citations:
            score = ""
            if citation.relevance_score is not None:
                score = f" score={citation.relevance_score:.3f}"
            title = citation.doc_title or citation.doc_id or citation.source
            lines.append(f"{citation.key} {title}{score}: {citation.snippet}")
        return "\n".join(lines)

    def validate_text(self, text: str) -> CitationValidation:
        used = list(dict.fromkeys(CITATION_KEY_RE.findall(text or "")))
        # findall with a capturing group returns prefixes, so use finditer instead
        used_keys = list(dict.fromkeys(match.group(0) for match in CITATION_KEY_RE.finditer(text or "")))
        known_keys = [c.key for c in self._citations]
        missing = [key for key in used_keys if key not in known_keys]
        unused = [key for key in known_keys if key not in used_keys]
        return CitationValidation(
            used_keys=used_keys or used,
            known_keys=known_keys,
            missing_keys=missing,
            unused_keys=unused,
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)
        log.info("Citations saved -> %s", path)

    @classmethod
    def from_reranked_result(cls, reranked_result: Any, *, max_chunks: int | None = None) -> "CitationTracker":
        tracker = cls()
        tracker.add_reranked_result(reranked_result, max_chunks=max_chunks)
        return tracker

    @classmethod
    def from_evidence_blocks(cls, blocks: Iterable[dict[str, Any]]) -> "CitationTracker":
        tracker = cls()
        for block in blocks:
            tracker.add_evidence_block(block)
        return tracker

    def _merge(
        self,
        citation: Citation,
        *,
        snippet: str = "",
        relevance_score: float | None = None,
        evidence_rank: int | None = None,
    ) -> None:
        if relevance_score is not None and (
            citation.relevance_score is None or relevance_score > citation.relevance_score
        ):
            citation.relevance_score = relevance_score
        if evidence_rank is not None and (
            citation.evidence_rank is None or evidence_rank < citation.evidence_rank
        ):
            citation.evidence_rank = evidence_rank
        if not citation.snippet and snippet:
            citation.snippet = clean_snippet(snippet)


# Backward-compatible alias for modules/docs that expect CitationIndex naming.
CitationIndex = CitationTracker


def rank_citations_for_display(citations: Iterable[Citation]) -> list[Citation]:
    """Sort by source importance, section importance, then evidence rank."""
    def sort_key(citation: Citation) -> tuple[int, int, int, str]:
        source_order = {"dailymed": 0, "faers": 1, "pubmed": 2, "drugbank": 3}
        section_order = HIGH_VALUE_SECTIONS.get((citation.section or "").upper(), 99)
        rank = citation.evidence_rank if citation.evidence_rank is not None else 999
        return (source_order.get(citation.source, 9), section_order, rank, citation.key)

    return sorted(citations, key=sort_key)


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockChunk:
        chunk_id: str
        text: str
        source: str
        drug_names: list[str]
        rrf_score: float
        doc_id: str = ""
        doc_title: str = ""
        doc_url: str = ""
        section: str = ""

    chunks = [
        MockChunk(
            chunk_id="dailymed-warfarin-1",
            text="Warfarin sodium can cause major or fatal bleeding. Concomitant antiplatelet agents may increase bleeding risk.",
            source="dailymed",
            drug_names=["warfarin"],
            rrf_score=0.031,
            doc_id="warfarin-label-setid",
            doc_title="Warfarin Sodium label",
            section="BOXED WARNING",
        ),
        MockChunk(
            chunk_id="pubmed-12345678",
            text="Combined anticoagulant and aspirin therapy was associated with an increased rate of clinically significant bleeding.",
            source="pubmed",
            drug_names=["warfarin", "aspirin"],
            rrf_score=0.028,
            doc_id="PMID:12345678",
            doc_title="Bleeding risk with warfarin and aspirin",
        ),
        MockChunk(
            chunk_id="dailymed-warfarin-duplicate",
            text="Duplicate chunk from the same label section should reuse the first FDA citation key.",
            source="dailymed",
            drug_names=["warfarin"],
            rrf_score=0.025,
            doc_id="warfarin-label-setid",
            doc_title="Warfarin Sodium label",
            section="BOXED WARNING",
        ),
    ]

    tracker = CitationTracker()
    for i, chunk in enumerate(chunks, start=1):
        key = tracker.add_chunk(chunk, evidence_rank=i)
        print(f"registered {chunk.chunk_id} -> {key}")

    print()
    print(tracker.format_references(include_snippets=True))
    print()
    example = "Warfarin plus aspirin may increase bleeding risk [FDA-1], supported by literature [PUB-1]."
    validation = tracker.validate_text(example)
    print("Validation:", validation)
