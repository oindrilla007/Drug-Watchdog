"""
ingestion/drugbank_parser.py
────────────────────────────────────────────────────────────────────────────────
Parses the DrugBank full-database XML dump and extracts:
  - Drug entries  (name, synonyms, description, molecular properties)
  - Drug-drug interactions  (drug_a ↔ drug_b, severity, description)
  - CYP450 enzyme relationships  (substrate / inhibitor / inducer flags)

DrugBank XML download (free academic licence):
  https://go.drugbank.com/releases/latest  →  drugbank_all_full_database.xml.zip

Place the unzipped XML at:
  data/raw/drugbank/drugbank_all_full_database.xml

Usage:
  from ingestion.drugbank_parser import DrugBankParser

  parser = DrugBankParser("data/raw/drugbank/drugbank_all_full_database.xml")
  parser.parse()

  drugs_df        = parser.get_drugs_df()
  interactions_df = parser.get_interactions_df()
  cyp450_df       = parser.get_cyp450_df()
  parser.save_all("data/processed/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── DrugBank XML namespace ────────────────────────────────────────────────────
NS = {"db": "http://www.drugbank.ca"}

# ── Severity keyword mapping ──────────────────────────────────────────────────
# DrugBank encodes severity in free text; we normalise to 0-3.
SEVERITY_MAP: dict[str, int] = {
    "major":    3,
    "severe":   3,
    "serious":  3,
    "moderate": 2,
    "minor":    1,
    "mild":     1,
    "minimal":  1,
}

# ── Known CYP450 enzyme names we care about ───────────────────────────────────
CYP_PATTERN = re.compile(
    r"CYP\s?(?:1A2|2B6|2C8|2C9|2C19|2D6|2E1|3A4|3A5)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DrugEntry:
    drugbank_id:    str
    name:           str
    synonyms:       list[str]       = field(default_factory=list)
    description:    str             = ""
    indication:     str             = ""
    pharmacodynamics: str           = ""
    mechanism:      str             = ""
    # Molecular properties
    molecular_weight: Optional[float] = None
    logp:           Optional[float] = None
    half_life:      str             = ""
    protein_binding: str            = ""
    # Classification
    drug_type:      str             = ""          # small-molecule / biologic
    groups:         list[str]       = field(default_factory=list)   # approved / experimental …
    # ATC codes
    atc_codes:      list[str]       = field(default_factory=list)
    # RxNorm / other external IDs
    rxnorm_id:      Optional[str]   = None


@dataclass
class DrugInteraction:
    drug_a_id:      str
    drug_a_name:    str
    drug_b_id:      str
    drug_b_name:    str
    description:    str  = ""
    severity:       int  = 0     # 0 = none/unknown, 1 = minor, 2 = moderate, 3 = major


@dataclass
class CYP450Relationship:
    drugbank_id:    str
    drug_name:      str
    enzyme:         str           # e.g. "CYP3A4"
    role:           str           # substrate | inhibitor | inducer


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class DrugBankParser:
    """
    Iteratively parses the DrugBank XML dump using iterparse so the full
    ~1 GB file never needs to sit in RAM all at once.
    """

    def __init__(self, xml_path: str | Path) -> None:
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"DrugBank XML not found at {self.xml_path}\n"
                "Download from https://go.drugbank.com/releases/latest "
                "and place at data/raw/drugbank/drugbank_all_full_database.xml"
            )

        self._drugs:        list[DrugEntry]         = []
        self._interactions: list[DrugInteraction]   = []
        self._cyp450:       list[CYP450Relationship] = []
        self._parsed = False

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> "DrugBankParser":
        """Parse the XML file. Returns self for chaining."""
        logger.info(f"Parsing DrugBank XML: {self.xml_path}")
        context = ET.iterparse(self.xml_path, events=("end",))
        drug_tag = f"{{{NS['db']}}}drug"
        count = 0

        for event, elem in context:
            if elem.tag == drug_tag and elem.get("type") in ("small molecule", "biotech"):
                drug = self._parse_drug_elem(elem)
                if drug:
                    self._drugs.append(drug)
                    self._extract_interactions(elem, drug)
                    self._extract_cyp450(elem, drug)
                    count += 1
                    if count % 500 == 0:
                        logger.info(f"  … parsed {count} drugs")
                # Free memory — we don't need this element anymore
                elem.clear()

        self._parsed = True
        logger.info(
            f"✅ Parsing complete — "
            f"{len(self._drugs):,} drugs | "
            f"{len(self._interactions):,} interactions | "
            f"{len(self._cyp450):,} CYP450 records"
        )
        return self

    def get_drugs_df(self) -> pd.DataFrame:
        self._check_parsed()
        rows = []
        for d in self._drugs:
            rows.append({
                "drugbank_id":      d.drugbank_id,
                "name":             d.name,
                "synonyms":         "|".join(d.synonyms),
                "description":      d.description,
                "indication":       d.indication,
                "pharmacodynamics": d.pharmacodynamics,
                "mechanism":        d.mechanism,
                "molecular_weight": d.molecular_weight,
                "logp":             d.logp,
                "half_life":        d.half_life,
                "protein_binding":  d.protein_binding,
                "drug_type":        d.drug_type,
                "groups":           "|".join(d.groups),
                "atc_codes":        "|".join(d.atc_codes),
                "rxnorm_id":        d.rxnorm_id,
            })
        return pd.DataFrame(rows)

    def get_interactions_df(self) -> pd.DataFrame:
        self._check_parsed()
        rows = [
            {
                "drug_a_id":    i.drug_a_id,
                "drug_a_name":  i.drug_a_name,
                "drug_b_id":    i.drug_b_id,
                "drug_b_name":  i.drug_b_name,
                "description":  i.description,
                "severity":     i.severity,
            }
            for i in self._interactions
        ]
        df = pd.DataFrame(rows)
        # Deduplicate symmetric pairs (A↔B == B↔A)
        if not df.empty:
            df["pair_key"] = df.apply(
                lambda r: "_".join(sorted([r.drug_a_id, r.drug_b_id])), axis=1
            )
            df = df.drop_duplicates(subset="pair_key").drop(columns="pair_key")
        return df

    def get_cyp450_df(self) -> pd.DataFrame:
        self._check_parsed()
        rows = [
            {
                "drugbank_id": c.drugbank_id,
                "drug_name":   c.drug_name,
                "enzyme":      c.enzyme,
                "role":        c.role,
            }
            for c in self._cyp450
        ]
        return pd.DataFrame(rows, columns=["drugbank_id", "drug_name", "enzyme", "role"])

    def save_all(self, output_dir: str | Path = "data/processed/") -> None:
        """Save all three DataFrames as Parquet files."""
        self._check_parsed()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        drugs_df = self.get_drugs_df()
        drugs_df.to_parquet(out / "drugbank_drugs.parquet", index=False)
        logger.info(f"💾 Saved {len(drugs_df):,} drugs → {out / 'drugbank_drugs.parquet'}")

        interactions_df = self.get_interactions_df()
        interactions_df.to_parquet(out / "drug_pairs.parquet", index=False)
        logger.info(f"💾 Saved {len(interactions_df):,} interactions → {out / 'drug_pairs.parquet'}")

        cyp450_df = self.get_cyp450_df()
        cyp450_df.to_parquet(out / "cyp450_relationships.parquet", index=False)
        logger.info(f"💾 Saved {len(cyp450_df):,} CYP450 records → {out / 'cyp450_relationships.parquet'}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_parsed(self) -> None:
        if not self._parsed:
            raise RuntimeError("Call .parse() before accessing data.")

    def _text(self, elem: ET.Element, tag: str) -> str:
        """Return stripped text of the first matching child tag, or ''."""
        node = elem.find(f"db:{tag}", NS)
        if node is not None and node.text:
            return node.text.strip()
        return ""

    def _parse_drug_elem(self, elem: ET.Element) -> Optional[DrugEntry]:
        """Extract a DrugEntry from a <drug> XML element."""
        # Primary DrugBank ID (first <drugbank-id primary="true">)
        db_id = None
        for id_elem in elem.findall("db:drugbank-id", NS):
            if id_elem.get("primary") == "true":
                db_id = id_elem.text
                break
        if db_id is None:
            return None

        name = self._text(elem, "name")
        if not name:
            return None

        # Synonyms
        synonyms = [
            s.text.strip()
            for s in elem.findall("db:synonyms/db:synonym", NS)
            if s.text
        ]

        # Groups (approved, experimental, …)
        groups = [
            g.text.strip()
            for g in elem.findall("db:groups/db:group", NS)
            if g.text
        ]

        # ATC codes
        atc_codes = [
            a.get("code", "")
            for a in elem.findall("db:atc-codes/db:atc-code", NS)
        ]

        # Molecular properties
        mol_weight = None
        logp = None
        props_elem = elem.find("db:calculated-properties", NS)
        if props_elem is not None:
            for prop in props_elem.findall("db:property", NS):
                kind = self._text(prop, "kind")
                val  = self._text(prop, "value")
                if kind == "Molecular Weight":
                    try:
                        mol_weight = float(val)
                    except ValueError:
                        pass
                elif kind == "logP":
                    try:
                        logp = float(val)
                    except ValueError:
                        pass

        # RxNorm external identifier
        rxnorm_id = None
        for ext in elem.findall("db:external-identifiers/db:external-identifier", NS):
            resource = self._text(ext, "resource")
            if resource == "RxCUI":
                rxnorm_id = self._text(ext, "identifier")
                break

        return DrugEntry(
            drugbank_id      = db_id,
            name             = name,
            synonyms         = synonyms,
            description      = self._text(elem, "description"),
            indication       = self._text(elem, "indication"),
            pharmacodynamics = self._text(elem, "pharmacodynamics"),
            mechanism        = self._text(elem, "mechanism-of-action"),
            molecular_weight = mol_weight,
            logp             = logp,
            half_life        = self._text(elem, "half-life"),
            protein_binding  = self._text(elem, "protein-binding"),
            drug_type        = elem.get("type", ""),
            groups           = groups,
            atc_codes        = [c for c in atc_codes if c],
            rxnorm_id        = rxnorm_id,
        )

    def _extract_interactions(self, elem: ET.Element, drug: DrugEntry) -> None:
        """Extract all drug-drug interactions listed under this drug entry."""
        for ix in elem.findall("db:drug-interactions/db:drug-interaction", NS):
            partner_id   = self._text(ix, "drugbank-id")
            partner_name = self._text(ix, "name")
            description  = self._text(ix, "description")

            severity = self._infer_severity(description)

            self._interactions.append(DrugInteraction(
                drug_a_id   = drug.drugbank_id,
                drug_a_name = drug.name,
                drug_b_id   = partner_id,
                drug_b_name = partner_name,
                description = description,
                severity    = severity,
            ))

    def _extract_cyp450(self, elem: ET.Element, drug: DrugEntry) -> None:
        """
        Extract CYP450 substrate / inhibitor / inducer relationships.
        DrugBank stores these under <enzymes> with <actions> per enzyme.
        """
        for enzyme_elem in elem.findall("db:enzymes/db:enzyme", NS):
            enzyme_name_elem = enzyme_elem.find("db:name", NS)
            if enzyme_name_elem is None or not enzyme_name_elem.text:
                continue
            enzyme_name = enzyme_name_elem.text.strip()

            # Only keep CYP enzymes we recognise
            if not CYP_PATTERN.search(enzyme_name):
                continue

            # Normalise enzyme name  →  "CYP3A4"
            match = CYP_PATTERN.search(enzyme_name)
            canonical_enzyme = match.group(0).replace(" ", "").upper() if match else enzyme_name

            actions = [
                a.text.strip().lower()
                for a in enzyme_elem.findall("db:actions/db:action", NS)
                if a.text
            ]

            for action in actions:
                if action in ("substrate", "inhibitor", "inducer",
                              "inducer/substrate", "inhibitor/substrate"):
                    # For compound roles, emit one record per role
                    for role in ("substrate", "inhibitor", "inducer"):
                        if role in action:
                            self._cyp450.append(CYP450Relationship(
                                drugbank_id = drug.drugbank_id,
                                drug_name   = drug.name,
                                enzyme      = canonical_enzyme,
                                role        = role,
                            ))

    @staticmethod
    def _infer_severity(description: str) -> int:
        """
        Map free-text interaction description to severity integer 0-3.
        Scans for severity keywords; returns the highest match found.
        """
        desc_lower = description.lower()
        best = 0
        for keyword, score in SEVERITY_MAP.items():
            if keyword in desc_lower:
                best = max(best, score)
        return best


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse DrugBank XML dump")
    ap.add_argument(
        "--xml",
        default="data/raw/drugbank/drugbank_all_full_database.xml",
        help="Path to DrugBank full-database XML",
    )
    ap.add_argument(
        "--out",
        default="data/processed/",
        help="Output directory for Parquet files",
    )
    args = ap.parse_args()

    parser = DrugBankParser(args.xml)
    parser.parse()
    parser.save_all(args.out)

    # Quick sanity report
    drugs_df        = parser.get_drugs_df()
    interactions_df = parser.get_interactions_df()
    cyp450_df       = parser.get_cyp450_df()

    print("\n── Drugs sample ──────────────────────────────────────")
    print(drugs_df[["drugbank_id", "name", "molecular_weight", "logp"]].head())

    print("\n── Interactions sample (severity ≥ 2) ────────────────")
    high = interactions_df[interactions_df["severity"] >= 2]
    print(high[["drug_a_name", "drug_b_name", "severity"]].head(10))

    print("\n── CYP450 sample ─────────────────────────────────────")
    if cyp450_df.empty:
        print("No CYP450 relationships found in this DrugBank release.")
    else:
        expected_cols = {"enzyme", "role"}
        if expected_cols.issubset(cyp450_df.columns):
            print(cyp450_df.groupby(["enzyme", "role"]).size().reset_index(name="count"))
        else:
            print(cyp450_df.head())
