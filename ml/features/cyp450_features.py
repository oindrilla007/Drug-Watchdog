"""
ml/features/cyp450_features.py
────────────────────────────────────────────────────────────────────────────────
Extracts CYP450 enzyme pathway features for every drug pair.

Since our DrugBank release returned 0 CYP450 records from the XML enzyme
elements, this module extracts CYP450 signal from TWO alternative sources:

  Source A — DrugBank interaction description text
             "The metabolism of Drug B can be decreased when combined with
              Drug A which is a CYP3A4 inhibitor."
             → parsed with regex to get enzyme + role

  Source B — DailyMed clinical_pharmacology + drug_interactions chunks
             FDA label text routinely describes CYP450 metabolism

Output features per drug pair (drug_a, drug_b):
  - shared_cyp_count        : # enzymes both drugs share (substrate overlap)
  - inhibitor_substrate_pairs: drug_a inhibits an enzyme that drug_b uses
  - inducer_substrate_pairs  : drug_a induces an enzyme that drug_b uses
  - cyp3a4_involved         : boolean — CYP3A4 in the interaction pathway
  - cyp2d6_involved         : boolean
  - cyp2c9_involved         : boolean
  - cyp2c19_involved        : boolean
  - cyp1a2_involved         : boolean
  - max_cyp_risk_score      : heuristic 0–3 based on known high-risk enzymes
  - cyp_description_snippet : extracted sentence mentioning CYP (for RAG)

Usage:
  from ml.features.cyp450_features import CYP450FeatureExtractor

  extractor = CYP450FeatureExtractor()
  extractor.fit(
      interactions_path = "data/processed/drug_pairs.parquet",
      dailymed_path     = "data/processed/dailymed_chunks.parquet",
  )
  features_df = extractor.transform(drug_pairs_df)
  extractor.save("data/processed/cyp450_features.parquet")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── CYP enzyme names we track ─────────────────────────────────────────────────
CYP_ENZYMES = [
    "CYP3A4", "CYP3A5",
    "CYP2D6",
    "CYP2C9", "CYP2C19", "CYP2C8",
    "CYP1A2",
    "CYP2B6",
    "CYP2E1",
]

# Regex to find CYP enzyme mentions in text
CYP_REGEX = re.compile(
    r"CYP\s?(?:3A[45]|2D6|2C(?:8|9|19)|1A2|2B6|2E1)",
    re.IGNORECASE,
)

# Regex to detect role near a CYP mention
INHIBITOR_REGEX  = re.compile(r"\b(inhibit(?:or|s|ing|ion)?)\b", re.IGNORECASE)
INDUCER_REGEX    = re.compile(r"\b(induc(?:er|es|ing|tion)?)\b", re.IGNORECASE)
SUBSTRATE_REGEX  = re.compile(r"\b(substrate|metaboli[sz]ed\s+by|metaboli[sz]ed\s+via)\b", re.IGNORECASE)

# ── Risk weights per enzyme (how clinically impactful is this enzyme?) ────────
# Based on frequency of clinical DDIs mediated by each enzyme
CYP_RISK_WEIGHT = {
    "CYP3A4":  3,   # ~50% of all drug metabolism — highest risk
    "CYP2D6":  3,   # narrow therapeutic index drugs (codeine, tamoxifen)
    "CYP2C9":  2,   # warfarin, phenytoin
    "CYP2C19": 2,   # clopidogrel, PPIs
    "CYP1A2":  1,
    "CYP2B6":  1,
    "CYP2C8":  1,
    "CYP3A5":  1,
    "CYP2E1":  1,
}

# ── Window around a CYP mention to search for role keywords (chars) ───────────
CONTEXT_WINDOW = 120


# ─────────────────────────────────────────────────────────────────────────────
# Per-drug CYP profile
# ─────────────────────────────────────────────────────────────────────────────

class DrugCYPProfile:
    """Stores CYP450 roles extracted for a single drug."""

    def __init__(self, drug_name: str) -> None:
        self.drug_name  = drug_name
        self.substrates: set[str] = set()   # enzymes this drug is substrate of
        self.inhibitors: set[str] = set()   # enzymes this drug inhibits
        self.inducers:   set[str] = set()   # enzymes this drug induces
        self.snippets:   list[str] = []     # source text snippets

    def add_role(self, enzyme: str, role: str, snippet: str = "") -> None:
        enzyme = enzyme.upper().replace(" ", "")
        if role == "substrate":
            self.substrates.add(enzyme)
        elif role == "inhibitor":
            self.inhibitors.add(enzyme)
        elif role == "inducer":
            self.inducers.add(enzyme)
        if snippet:
            self.snippets.append(snippet[:200])

    @property
    def all_enzymes(self) -> set[str]:
        return self.substrates | self.inhibitors | self.inducers

    def __repr__(self) -> str:
        return (f"DrugCYPProfile({self.drug_name}: "
                f"sub={self.substrates}, inh={self.inhibitors}, ind={self.inducers})")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class CYP450FeatureExtractor:
    """
    Builds per-drug CYP450 profiles from text, then generates
    pairwise interaction features for every drug pair.
    """

    def __init__(self) -> None:
        # drug_name (lowercase) → DrugCYPProfile
        self._profiles: dict[str, DrugCYPProfile] = {}
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        interactions_path: Optional[str | Path] = None,
        dailymed_path:     Optional[str | Path] = None,
        drugbank_drugs_path: Optional[str | Path] = None,
    ) -> "CYP450FeatureExtractor":
        """
        Build CYP450 profiles for all drugs from available text sources.
        At least one path must be provided.
        """
        n_sources = 0

        # Source A: DrugBank interaction descriptions
        if interactions_path and Path(interactions_path).exists():
            df = pd.read_parquet(interactions_path)
            self._extract_from_interactions(df)
            n_sources += 1
            logger.info(f"  Profiles after DrugBank interactions: {len(self._profiles):,}")

        # Source B: DrugBank drug descriptions / mechanism text
        if drugbank_drugs_path and Path(drugbank_drugs_path).exists():
            df = pd.read_parquet(drugbank_drugs_path)
            self._extract_from_drug_descriptions(df)
            n_sources += 1
            logger.info(f"  Profiles after DrugBank descriptions: {len(self._profiles):,}")

        # Source C: DailyMed clinical_pharmacology + drug_interactions chunks
        if dailymed_path and Path(dailymed_path).exists():
            df = pd.read_parquet(dailymed_path)
            self._extract_from_dailymed(df)
            n_sources += 1
            logger.info(f"  Profiles after DailyMed: {len(self._profiles):,}")

        if n_sources == 0:
            raise ValueError("At least one text source path must be provided and exist.")

        self._fitted = True
        drugs_with_cyp = sum(1 for p in self._profiles.values() if p.all_enzymes)
        logger.info(
            f"✅ CYP450 fit complete — "
            f"{len(self._profiles):,} drug profiles | "
            f"{drugs_with_cyp:,} with CYP450 data"
        )
        return self

    def _extract_from_interactions(self, df: pd.DataFrame) -> None:
        """
        Parse DrugBank interaction description text for CYP mentions.
        Each description mentions two drugs — we attribute the role to the
        correct drug based on sentence context.
        """
        desc_col = "description"
        if desc_col not in df.columns:
            return

        for _, row in df.iterrows():
            text   = str(row.get(desc_col, ""))
            drug_a = str(row.get("drug_a_name", "")).strip()
            drug_b = str(row.get("drug_b_name", "")).strip()

            if not text or len(text) < 20:
                continue

            cyp_roles = self._parse_cyp_roles_from_text(text)
            for enzyme, role, snippet in cyp_roles:
                # Heuristic: assign role to whichever drug is mentioned
                # closer to the CYP mention in the sentence
                drug_a_pos = text.lower().find(drug_a.lower())
                drug_b_pos = text.lower().find(drug_b.lower())
                cyp_pos    = text.lower().find(enzyme.lower())

                if cyp_pos < 0:
                    continue

                dist_a = abs(drug_a_pos - cyp_pos) if drug_a_pos >= 0 else 9999
                dist_b = abs(drug_b_pos - cyp_pos) if drug_b_pos >= 0 else 9999

                target_drug = drug_a if dist_a <= dist_b else drug_b
                self._get_profile(target_drug).add_role(enzyme, role, snippet)

    def _extract_from_drug_descriptions(self, df: pd.DataFrame) -> None:
        """
        Parse DrugBank drug-level text fields for CYP mentions.
        """
        text_cols = ["description", "mechanism", "pharmacodynamics"]
        for _, row in df.iterrows():
            drug_name = str(row.get("name", "")).strip()
            if not drug_name:
                continue
            combined = " ".join(
                str(row.get(col, "")) for col in text_cols if col in df.columns
            )
            cyp_roles = self._parse_cyp_roles_from_text(combined)
            for enzyme, role, snippet in cyp_roles:
                self._get_profile(drug_name).add_role(enzyme, role, snippet)

    def _extract_from_dailymed(self, df: pd.DataFrame) -> None:
        """
        Parse DailyMed chunks — focus on clinical_pharmacology and
        drug_interactions sections which reliably describe CYP metabolism.
        """
        relevant_sections = {"clinical_pharmacology", "drug_interactions", "mechanism_of_action"}
        if "section_type" in df.columns:
            df = df[df["section_type"].isin(relevant_sections)]

        for _, row in df.iterrows():
            drug_name = str(row.get("drug_name", "")).strip()
            text      = str(row.get("text", "")).strip()
            if not drug_name or not text:
                continue
            cyp_roles = self._parse_cyp_roles_from_text(text)
            for enzyme, role, snippet in cyp_roles:
                self._get_profile(drug_name).add_role(enzyme, role, snippet)

    # ── Transformation ────────────────────────────────────────────────────────

    def transform(self, drug_pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate CYP450 features for every row in drug_pairs_df.
        Required columns: drug_a_name, drug_b_name
        Returns original df with new CYP feature columns appended.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        logger.info(f"Generating CYP450 features for {len(drug_pairs_df):,} drug pairs …")

        feature_rows = []
        for _, row in drug_pairs_df.iterrows():
            drug_a = str(row.get("drug_a_name", ""))
            drug_b = str(row.get("drug_b_name", ""))
            feature_rows.append(self._pair_features(drug_a, drug_b))

        features_df = pd.DataFrame(feature_rows)
        result = pd.concat(
            [drug_pairs_df.reset_index(drop=True), features_df],
            axis=1,
        )
        logger.info(f"✅ CYP450 features generated — shape: {result.shape}")
        return result

    def _pair_features(self, drug_a: str, drug_b: str) -> dict:
        """Compute all CYP450 features for a single drug pair."""
        prof_a = self._profiles.get(drug_a.lower(), DrugCYPProfile(drug_a))
        prof_b = self._profiles.get(drug_b.lower(), DrugCYPProfile(drug_b))

        # Shared substrates (both metabolised by same enzyme → competition)
        shared_substrates = prof_a.substrates & prof_b.substrates

        # Inhibitor-substrate pairs (A inhibits enzyme that B needs → B levels ↑)
        a_inhibits_b_sub = prof_a.inhibitors & prof_b.substrates
        b_inhibits_a_sub = prof_b.inhibitors & prof_a.substrates
        inhibitor_substrate_pairs = len(a_inhibits_b_sub | b_inhibits_a_sub)

        # Inducer-substrate pairs (A induces enzyme that B needs → B levels ↓)
        a_induces_b_sub = prof_a.inducers & prof_b.substrates
        b_induces_a_sub = prof_b.inducers & prof_a.substrates
        inducer_substrate_pairs = len(a_induces_b_sub | b_induces_a_sub)

        # All enzymes involved in this pair
        all_enzymes = prof_a.all_enzymes | prof_b.all_enzymes

        # Max risk score based on enzymes involved
        max_risk = max(
            (CYP_RISK_WEIGHT.get(e, 0) for e in all_enzymes),
            default=0,
        )

        # Key enzyme boolean flags
        cyp_snippet = ""
        if prof_a.snippets:
            cyp_snippet = prof_a.snippets[0]
        elif prof_b.snippets:
            cyp_snippet = prof_b.snippets[0]

        return {
            "shared_cyp_count":          len(shared_substrates),
            "inhibitor_substrate_pairs": inhibitor_substrate_pairs,
            "inducer_substrate_pairs":   inducer_substrate_pairs,
            "cyp3a4_involved":           int("CYP3A4" in all_enzymes),
            "cyp2d6_involved":           int("CYP2D6" in all_enzymes),
            "cyp2c9_involved":           int("CYP2C9" in all_enzymes),
            "cyp2c19_involved":          int("CYP2C19" in all_enzymes),
            "cyp1a2_involved":           int("CYP1A2" in all_enzymes),
            "cyp2b6_involved":           int("CYP2B6" in all_enzymes),
            "max_cyp_risk_score":        max_risk,
            "any_cyp_interaction":       int(bool(all_enzymes)),
            "n_enzymes_involved":        len(all_enzymes),
            "cyp_description_snippet":   cyp_snippet,
        }

    def save(self, output_path: str | Path = "data/processed/cyp450_features.parquet") -> None:
        """
        Save the profile data as a flat parquet for inspection / debugging.
        Note: call transform() to get the actual pairwise feature matrix.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before save()")

        rows = []
        for drug_name, prof in self._profiles.items():
            for enzyme in prof.substrates:
                rows.append({"drug_name": drug_name, "enzyme": enzyme, "role": "substrate"})
            for enzyme in prof.inhibitors:
                rows.append({"drug_name": drug_name, "enzyme": enzyme, "role": "inhibitor"})
            for enzyme in prof.inducers:
                rows.append({"drug_name": drug_name, "enzyme": enzyme, "role": "inducer"})

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["drug_name", "enzyme", "role"]
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"💾 Saved {len(df):,} CYP450 profile records → {output_path}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_profile(self, drug_name: str) -> DrugCYPProfile:
        """Get or create a DrugCYPProfile for a drug name."""
        key = drug_name.lower().strip()
        if key not in self._profiles:
            self._profiles[key] = DrugCYPProfile(drug_name)
        return self._profiles[key]

    @staticmethod
    def _parse_cyp_roles_from_text(text: str) -> list[tuple[str, str, str]]:
        """
        Scan text for CYP enzyme mentions and determine role (substrate /
        inhibitor / inducer) from surrounding context.

        Returns list of (enzyme, role, snippet) tuples.
        """
        results = []
        for match in CYP_REGEX.finditer(text):
            enzyme = match.group(0).upper().replace(" ", "")
            # Normalise e.g. CYP3A 4 → CYP3A4
            enzyme = re.sub(r"\s", "", enzyme)

            start  = max(0, match.start() - CONTEXT_WINDOW)
            end    = min(len(text), match.end() + CONTEXT_WINDOW)
            window = text[start:end]
            snippet = window.strip()

            role = "unknown"
            if INHIBITOR_REGEX.search(window):
                role = "inhibitor"
            elif INDUCER_REGEX.search(window):
                role = "inducer"
            elif SUBSTRATE_REGEX.search(window):
                role = "substrate"

            if role != "unknown":
                results.append((enzyme, role, snippet))

        return results

    def coverage_report(self) -> None:
        """Print a summary of CYP450 coverage across drugs."""
        total   = len(self._profiles)
        with_cyp = sum(1 for p in self._profiles.values() if p.all_enzymes)
        substrate_counts = defaultdict(int)
        inhibitor_counts = defaultdict(int)

        for prof in self._profiles.values():
            for e in prof.substrates:
                substrate_counts[e] += 1
            for e in prof.inhibitors:
                inhibitor_counts[e] += 1

        print(f"\n── CYP450 Coverage Report ──────────────────────────────────")
        print(f"Total drug profiles:        {total:,}")
        print(f"Drugs with CYP450 data:     {with_cyp:,} ({with_cyp/max(total,1)*100:.1f}%)")
        print(f"\nTop substrate enzymes:")
        for e, n in sorted(substrate_counts.items(), key=lambda x: -x[1])[:9]:
            print(f"  {e:<12}  {n:>5,} drugs")
        print(f"\nTop inhibited enzymes:")
        for e, n in sorted(inhibitor_counts.items(), key=lambda x: -x[1])[:9]:
            print(f"  {e:<12}  {n:>5,} drugs")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract CYP450 features from drug interaction text")
    ap.add_argument("--interactions", default="data/processed/drug_pairs.parquet")
    ap.add_argument("--dailymed",     default="data/processed/dailymed_chunks.parquet")
    ap.add_argument("--drugs",        default="data/processed/drugbank_drugs.parquet")
    ap.add_argument("--out",          default="data/processed/cyp450_features.parquet")
    args = ap.parse_args()

    extractor = CYP450FeatureExtractor()
    extractor.fit(
        interactions_path   = args.interactions,
        dailymed_path       = args.dailymed,
        drugbank_drugs_path = args.drugs,
    )
    extractor.coverage_report()
    extractor.save(args.out)

    # Quick transform test on first 1000 pairs
    pairs_df = pd.read_parquet(args.interactions).head(1000)
    features_df = extractor.transform(pairs_df)
    print(f"\n── Feature sample (first 5 rows) ───────────────────────────")
    cyp_cols = [c for c in features_df.columns if "cyp" in c.lower() or "enzyme" in c.lower()]
    print(features_df[["drug_a_name", "drug_b_name", "severity"] + cyp_cols].head())
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"Non-zero CYP rows: {(features_df['any_cyp_interaction'] == 1).sum():,}")