"""
ingestion/dailymed_parser.py
────────────────────────────────────────────────────────────────────────────────
Parses FDA DailyMed Structured Product Label (SPL) XML files and extracts:
  - Drug name + RxNorm / NDC identifiers
  - Warnings & Boxed Warnings  (highest-priority safety text)
  - Contraindications
  - Drug Interactions section
  - Dosage & Administration
  - Adverse Reactions

DailyMed bulk download (free, no login):
  https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
  → Download "All Human Drug Labels" ZIP (~8 GB unzipped)
  → Unzip to:  data/raw/dailymed/
    Each drug label lives in its own subfolder as an XML file.

Usage:
  from ingestion.dailymed_parser import DailyMedParser

  parser = DailyMedParser("data/raw/dailymed/")
  parser.parse()                        # walks all XML files in the folder
  labels_df = parser.get_labels_df()
  chunks    = parser.get_chunks()       # RAG-ready text chunks with metadata
  parser.save_all("data/processed/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── SPL XML namespace ─────────────────────────────────────────────────────────
NS = {"spl": "urn:hl7-org:v3"}

# ── LOINC section codes we care about ────────────────────────────────────────
# DailyMed uses LOINC codes to tag each label section.
SECTION_CODES: dict[str, str] = {
    "34066-1": "boxed_warning",
    "43685-7": "warnings_and_precautions",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "34070-3": "contraindications",
    "34068-7": "dosage_and_administration",
    "34089-3": "description",
    "43679-0": "mechanism_of_action",
    "34090-1": "clinical_pharmacology",
    "34092-7": "clinical_studies",
}

# Sections that go into the RAG knowledge base (highest signal for safety)
RAG_PRIORITY_SECTIONS = {
    "boxed_warning",
    "warnings_and_precautions",
    "drug_interactions",
    "contraindications",
    "adverse_reactions",
}

# Chunk size guard — SPL sections can be very long; we split above this limit
MAX_CHUNK_CHARS = 1_200


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPLLabel:
    """One FDA drug label."""
    set_id:          str                        # SPL setId (stable across revisions)
    doc_id:          str                        # SPL document UUID (version-specific)
    drug_name:       str
    rxnorm_ids:      list[str]  = field(default_factory=list)
    ndc_codes:       list[str]  = field(default_factory=list)
    sections:        dict[str, str] = field(default_factory=dict)
    # sections keys = section_type strings from SECTION_CODES values above


@dataclass
class TextChunk:
    """A RAG-ready chunk of text from a DailyMed label."""
    chunk_id:       str          # unique UUID
    source:         str          # "dailymed"
    doc_id:         str          # SPL document UUID
    drug_name:      str
    section_type:   str          # e.g. "drug_interactions"
    text:           str
    char_count:     int
    priority:       bool         # True if in RAG_PRIORITY_SECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class DailyMedParser:
    """
    Walks a directory of SPL XML files (one per drug label) and extracts
    structured label data + RAG-ready text chunks.
    """

    def __init__(self, spl_dir: str | Path) -> None:
        self.spl_dir = Path(spl_dir)
        if not self.spl_dir.exists():
            raise FileNotFoundError(
                f"DailyMed directory not found: {self.spl_dir}\n"
                "Download from https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"
            )

        self._labels: list[SPLLabel]  = []
        self._chunks: list[TextChunk] = []
        self._parsed = False

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, max_files: Optional[int] = None) -> "DailyMedParser":
        """
        Walk all XML files under spl_dir and parse each one.
        max_files: cap for development / testing (None = parse all).
        """
        xml_files = list(self.spl_dir.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found under {self.spl_dir}")

        if max_files:
            xml_files = xml_files[:max_files]

        logger.info(f"Parsing {len(xml_files):,} SPL XML files from {self.spl_dir}")

        success = fail = 0
        for i, xml_path in enumerate(xml_files, 1):
            try:
                label = self._parse_single(xml_path)
                if label:
                    self._labels.append(label)
                    self._chunks.extend(self._make_chunks(label))
                    success += 1
            except Exception as exc:
                logger.debug(f"  ⚠ Skipped {xml_path.name}: {exc}")
                fail += 1

            if i % 1_000 == 0:
                logger.info(f"  … processed {i:,}/{len(xml_files):,} files")

        self._parsed = True
        logger.info(
            f"✅ DailyMed parse complete — "
            f"{success:,} labels | {len(self._chunks):,} chunks | {fail:,} skipped"
        )
        return self

    def get_labels_df(self) -> pd.DataFrame:
        self._check_parsed()
        rows = []
        for lb in self._labels:
            rows.append({
                "set_id":          lb.set_id,
                "doc_id":          lb.doc_id,
                "drug_name":       lb.drug_name,
                "rxnorm_ids":      "|".join(lb.rxnorm_ids),
                "ndc_codes":       "|".join(lb.ndc_codes[:5]),  # cap for readability
                "has_boxed_warn":  "boxed_warning" in lb.sections,
                "has_interactions": "drug_interactions" in lb.sections,
                "section_count":   len(lb.sections),
            })
        return pd.DataFrame(rows)

    def get_chunks(self) -> list[TextChunk]:
        self._check_parsed()
        return self._chunks

    def get_chunks_df(self) -> pd.DataFrame:
        self._check_parsed()
        rows = [
            {
                "chunk_id":     c.chunk_id,
                "source":       c.source,
                "doc_id":       c.doc_id,
                "drug_name":    c.drug_name,
                "section_type": c.section_type,
                "text":         c.text,
                "char_count":   c.char_count,
                "priority":     c.priority,
            }
            for c in self._chunks
        ]
        return pd.DataFrame(rows)

    def save_all(self, output_dir: str | Path = "data/processed/") -> None:
        self._check_parsed()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        labels_df = self.get_labels_df()
        labels_df.to_parquet(out / "dailymed_labels.parquet", index=False)
        logger.info(f"💾 Saved {len(labels_df):,} labels → {out / 'dailymed_labels.parquet'}")

        chunks_df = self.get_chunks_df()
        chunks_df.to_parquet(out / "dailymed_chunks.parquet", index=False)
        logger.info(f"💾 Saved {len(chunks_df):,} chunks → {out / 'dailymed_chunks.parquet'}")

        # Also save priority-only chunks separately (what the RAG system uses first)
        priority_df = chunks_df[chunks_df["priority"]].copy()
        priority_df.to_parquet(out / "dailymed_priority_chunks.parquet", index=False)
        logger.info(f"💾 Saved {len(priority_df):,} priority chunks → {out / 'dailymed_priority_chunks.parquet'}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_parsed(self) -> None:
        if not self._parsed:
            raise RuntimeError("Call .parse() before accessing data.")

    def _parse_single(self, xml_path: Path) -> Optional[SPLLabel]:
        """Parse one SPL XML file → SPLLabel or None if not a drug label."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Confirm this is an SPL document
        if "v3" not in root.tag:
            return None

        # ── Document identifiers ──────────────────────────────────────────────
        set_id_elem = root.find("spl:setId", NS)
        doc_id_elem = root.find("spl:id", NS)
        set_id = set_id_elem.get("root", "") if set_id_elem is not None else ""
        doc_id = doc_id_elem.get("root", "") if doc_id_elem is not None else str(uuid.uuid4())

        # ── Drug name ─────────────────────────────────────────────────────────
        # SPL stores the manufactured product name under subject → manufacturedProduct
        drug_name = self._extract_drug_name(root)
        if not drug_name:
            return None

        # ── RxNorm IDs ────────────────────────────────────────────────────────
        rxnorm_ids = self._extract_codes(root, "2.16.840.1.113883.6.88")   # RxNorm OID
        ndc_codes  = self._extract_codes(root, "2.16.840.1.113883.6.69")   # NDC OID

        # ── Sections ─────────────────────────────────────────────────────────
        sections = self._extract_sections(root)
        if not sections:
            return None

        return SPLLabel(
            set_id     = set_id,
            doc_id     = doc_id,
            drug_name  = drug_name,
            rxnorm_ids = rxnorm_ids,
            ndc_codes  = ndc_codes,
            sections   = sections,
        )

    def _extract_drug_name(self, root: ET.Element) -> str:
        """Pull the generic or brand name from the SPL header."""
        # Try generic name first
        for path in [
            ".//spl:manufacturedProduct/spl:manufacturedProduct/spl:name",
            ".//spl:manufacturedProduct/spl:name",
            ".//spl:subject/spl:manufacturedProduct/spl:name",
        ]:
            elem = root.find(path, NS)
            if elem is not None and elem.text:
                return elem.text.strip()
        return ""

    def _extract_codes(self, root: ET.Element, oid: str) -> list[str]:
        """Extract all coding system codes matching a given OID."""
        codes = []
        for code_elem in root.iter():
            # Look for elements with codeSystem attribute matching our OID
            if code_elem.get("codeSystem") == oid:
                code = code_elem.get("code", "")
                if code:
                    codes.append(code)
        return list(set(codes))  # deduplicate

    def _extract_sections(self, root: ET.Element) -> dict[str, str]:
        """
        Walk all <section> elements, match their LOINC code to our
        SECTION_CODES map, and return {section_type: cleaned_text}.
        """
        sections: dict[str, str] = {}

        for section in root.iter(f"{{{NS['spl']}}}section"):
            code_elem = section.find("spl:code", NS)
            if code_elem is None:
                continue

            loinc_code = code_elem.get("code", "")
            section_type = SECTION_CODES.get(loinc_code)
            if not section_type:
                continue

            # Collect all text inside this section
            raw_text = self._extract_text(section)
            cleaned  = self._clean_text(raw_text)
            if cleaned:
                # If we already have this section type, append (some labels
                # split warnings across multiple <section> elements)
                if section_type in sections:
                    sections[section_type] += "\n\n" + cleaned
                else:
                    sections[section_type] = cleaned

        return sections

    def _extract_text(self, elem: ET.Element) -> str:
        """Recursively extract all text from an XML element tree."""
        parts = []
        if elem.text:
            parts.append(elem.text.strip())
        for child in elem:
            parts.append(self._extract_text(child))
            if child.tail:
                parts.append(child.tail.strip())
        return " ".join(filter(None, parts))

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove boilerplate artifacts."""
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove very short sections (likely parsing artifacts)
        if len(text) < 40:
            return ""
        return text

    def _make_chunks(self, label: SPLLabel) -> list[TextChunk]:
        """
        Convert each label section into one or more TextChunks.
        Sections longer than MAX_CHUNK_CHARS are split at sentence boundaries.
        """
        chunks: list[TextChunk] = []

        for section_type, text in label.sections.items():
            is_priority = section_type in RAG_PRIORITY_SECTIONS
            sub_texts   = self._split_text(text)

            for sub in sub_texts:
                chunks.append(TextChunk(
                    chunk_id     = str(uuid.uuid4()),
                    source       = "dailymed",
                    doc_id       = label.doc_id,
                    drug_name    = label.drug_name,
                    section_type = section_type,
                    text         = sub,
                    char_count   = len(sub),
                    priority     = is_priority,
                ))

        return chunks

    @staticmethod
    def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
        """
        Split long text into overlapping chunks at sentence boundaries.
        Uses a simple sentence splitter (no NLTK dependency).
        Overlap: last sentence of previous chunk is prepended to next chunk
        to preserve context across boundaries.
        """
        if len(text) <= max_chars:
            return [text]

        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > max_chars and current:
                chunks.append(" ".join(current))
                # Overlap: carry last sentence into next chunk
                current = [current[-1], sent]
                current_len = len(current[-2]) + len(sent)
            else:
                current.append(sent)
                current_len += len(sent)

        if current:
            chunks.append(" ".join(current))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse FDA DailyMed SPL XML labels")
    ap.add_argument(
        "--dir",
        default="data/raw/dailymed/",
        help="Directory containing SPL XML files",
    )
    ap.add_argument(
        "--out",
        default="data/processed/",
        help="Output directory for Parquet files",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max files to parse (dev/test mode)",
    )
    args = ap.parse_args()

    parser = DailyMedParser(args.dir)
    parser.parse(max_files=args.max)
    parser.save_all(args.out)

    # Quick sanity report
    labels_df = parser.get_labels_df()
    chunks_df = parser.get_chunks_df()

    print("\n── Labels sample ─────────────────────────────────────")
    print(labels_df[["drug_name", "has_boxed_warn", "has_interactions", "section_count"]].head(10))

    print("\n── Chunk section distribution ────────────────────────")
    print(chunks_df["section_type"].value_counts())

    print("\n── Priority chunk sample (drug_interactions) ─────────")
    sample = chunks_df[chunks_df["section_type"] == "drug_interactions"].head(3)
    for _, row in sample.iterrows():
        print(f"\n[{row['drug_name']}]\n{row['text'][:300]}…")