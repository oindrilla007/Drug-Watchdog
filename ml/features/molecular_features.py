"""
ml/features/molecular_features.py
────────────────────────────────────────────────────────────────────────────────
Builds physicochemical and pharmacokinetic features for every drug pair.

Feature sources:
  1. DrugBank structured fields   (molecular_weight, logP, half_life,
                                   protein_binding, drug_type)
  2. FAERS co-occurrence signals  (report_count, death_count, max_severity)
  3. RxNorm mapping               (resolved CUI → indicates drug is well-known)
  4. DailyMed label flags         (has_boxed_warning, has_interactions)

Features generated per drug pair:
  Molecular property features (per drug + pair-level diff/ratio):
    mw_a, mw_b, mw_diff, mw_ratio
    logp_a, logp_b, logp_diff, logp_ratio
    protein_binding_a, protein_binding_b, pb_diff
    half_life_hours_a, half_life_hours_b, hl_ratio

  Drug type features:
    both_small_molecule, one_biologic

  FAERS pharmacovigilance features:
    faers_report_count          (co-occurrence reports for this pair)
    faers_death_count
    faers_max_severity
    faers_death_rate            (death_count / report_count)
    faers_signal_strength       (log-transformed report count)

  Label safety flags:
    drug_a_has_boxed_warning
    drug_b_has_boxed_warning
    either_has_boxed_warning
    drug_a_has_interactions_section
    drug_b_has_interactions_section

  Data quality:
    mol_data_completeness       (fraction of mol fields available 0–1)

Usage:
  from ml.features.molecular_features import MolecularFeatureExtractor

  extractor = MolecularFeatureExtractor()
  extractor.fit(
      drugbank_path = "data/processed/drugbank_drugs.parquet",
      faers_path    = "data/processed/faers_co_occurrence.parquet",
      dailymed_path = "data/processed/dailymed_labels.parquet",
      rxnorm_path   = "data/processed/rxnorm_mapping.parquet",
  )
  features_df = extractor.transform(drug_pairs_df)
  extractor.save("data/processed/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Half-life parsing patterns ────────────────────────────────────────────────
# DrugBank stores half-life as free text: "22 hours", "1-2 days", "30 min"
HL_PATTERNS = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*hour", re.I), "hours_range"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*hour",                         re.I), "hours"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*day",   re.I), "days_range"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*day",                          re.I), "days"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*min",   re.I), "min_range"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*min",                          re.I), "min"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*week",                         re.I), "weeks"),
]

# ── Protein binding parsing ───────────────────────────────────────────────────
# DrugBank: "Approximately 99%", "~85%", "55 to 65%", "High (>90%)"
PB_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)?\s*%|(\d+(?:\.\d+)?)\s*%")


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MolecularFeatureExtractor:
    """
    Builds a rich molecular + pharmacovigilance feature matrix for drug pairs.
    """

    def __init__(self) -> None:
        self._drug_props:   pd.DataFrame = pd.DataFrame()   # per-drug properties
        self._faers_pairs:  pd.DataFrame = pd.DataFrame()   # FAERS co-occurrence
        self._label_flags:  pd.DataFrame = pd.DataFrame()   # DailyMed safety flags
        self._rxnorm_known: set[str] = set()                 # names with CUI
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        drugbank_path: Optional[str | Path] = None,
        faers_path:    Optional[str | Path] = None,
        dailymed_path: Optional[str | Path] = None,
        rxnorm_path:   Optional[str | Path] = None,
    ) -> "MolecularFeatureExtractor":

        if drugbank_path and Path(drugbank_path).exists():
            self._load_drugbank(drugbank_path)

        if faers_path and Path(faers_path).exists():
            self._load_faers(faers_path)

        if dailymed_path and Path(dailymed_path).exists():
            self._load_dailymed(dailymed_path)

        if rxnorm_path and Path(rxnorm_path).exists():
            self._load_rxnorm(rxnorm_path)

        self._fitted = True
        logger.info(
            f"✅ MolecularFeatureExtractor fitted — "
            f"{len(self._drug_props):,} drug property records | "
            f"{len(self._faers_pairs):,} FAERS co-occurrence pairs | "
            f"{len(self._label_flags):,} DailyMed label flags"
        )
        return self

    def _load_drugbank(self, path: str | Path) -> None:
        df = pd.read_parquet(path)
        logger.info(f"  Loading DrugBank properties: {len(df):,} drugs")

        # Parse half-life text → numeric hours
        if "half_life" in df.columns:
            df["half_life_hours"] = df["half_life"].apply(self._parse_half_life)

        # Parse protein binding text → numeric %
        if "protein_binding" in df.columns:
            df["protein_binding_pct"] = df["protein_binding"].apply(self._parse_protein_binding)

        # Build lookup by name (lowercase) and by drugbank_id
        self._drug_props = df.copy()

        # Create fast name → row index lookup
        self._name_to_idx: dict[str, int] = {}
        for i, row in df.iterrows():
            name = str(row.get("name", "")).lower().strip()
            if name:
                self._name_to_idx[name] = i
            # Also index by synonyms
            for syn in str(row.get("synonyms", "")).split("|"):
                syn = syn.lower().strip()
                if syn and syn not in self._name_to_idx:
                    self._name_to_idx[syn] = i

        logger.info(f"  Drug name index: {len(self._name_to_idx):,} entries")

    def _load_faers(self, path: str | Path) -> None:
        df = pd.read_parquet(path)
        logger.info(f"  Loading FAERS co-occurrence: {len(df):,} pairs")

        # Normalise drug names to lowercase for matching
        if "drug_a" in df.columns and "drug_b" in df.columns:
            df["drug_a_lower"] = df["drug_a"].str.lower().str.strip()
            df["drug_b_lower"] = df["drug_b"].str.lower().str.strip()
            # Build canonical pair key for fast lookup
            df["pair_key"] = df.apply(
                lambda r: "_".join(sorted([r["drug_a_lower"], r["drug_b_lower"]])),
                axis=1,
            )
            self._faers_pairs = df.set_index("pair_key")

    def _load_dailymed(self, path: str | Path) -> None:
        df = pd.read_parquet(path)
        logger.info(f"  Loading DailyMed label flags: {len(df):,} labels")
        if "drug_name" in df.columns:
            df["drug_name_lower"] = df["drug_name"].str.lower().str.strip()
            self._label_flags = df.set_index("drug_name_lower")

    def _load_rxnorm(self, path: str | Path) -> None:
        df = pd.read_parquet(path)
        if "canonical_name" in df.columns:
            resolved = df[df["resolved"] == True]["canonical_name"].dropna()
            self._rxnorm_known = set(resolved.str.lower().str.strip().tolist())
            logger.info(f"  RxNorm known drugs: {len(self._rxnorm_known):,}")

    # ── Transformation ────────────────────────────────────────────────────────

    def transform(self, drug_pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate molecular features for every drug pair.
        Appends feature columns to a copy of drug_pairs_df.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        logger.info(f"Generating molecular features for {len(drug_pairs_df):,} drug pairs …")

        rows = []
        for _, row in drug_pairs_df.iterrows():
            drug_a = str(row.get("drug_a_name", ""))
            drug_b = str(row.get("drug_b_name", ""))
            rows.append(self._pair_features(drug_a, drug_b))

        features_df = pd.DataFrame(rows)
        result = pd.concat(
            [drug_pairs_df.reset_index(drop=True), features_df],
            axis=1,
        )
        logger.info(f"✅ Molecular features generated — shape: {result.shape}")
        return result

    def _pair_features(self, drug_a: str, drug_b: str) -> dict:
        """Compute all molecular features for a single drug pair."""
        props_a = self._get_drug_props(drug_a)
        props_b = self._get_drug_props(drug_b)

        mw_a  = props_a.get("molecular_weight")
        mw_b  = props_b.get("molecular_weight")
        lp_a  = props_a.get("logp")
        lp_b  = props_b.get("logp")
        pb_a  = props_a.get("protein_binding_pct")
        pb_b  = props_b.get("protein_binding_pct")
        hl_a  = props_a.get("half_life_hours")
        hl_b  = props_b.get("half_life_hours")

        # Pair-level molecular diffs / ratios
        mw_diff   = self._safe_diff(mw_a, mw_b)
        mw_ratio  = self._safe_ratio(mw_a, mw_b)
        lp_diff   = self._safe_diff(lp_a, lp_b)
        lp_ratio  = self._safe_ratio(lp_a, lp_b)
        pb_diff   = self._safe_diff(pb_a, pb_b)
        hl_ratio  = self._safe_ratio(hl_a, hl_b)

        # Drug type flags
        type_a = str(props_a.get("drug_type", "")).lower()
        type_b = str(props_b.get("drug_type", "")).lower()
        both_small = int("small" in type_a and "small" in type_b)
        one_bio    = int("biotech" in type_a or "biotech" in type_b)

        # FAERS pharmacovigilance signal
        faers = self._get_faers_signal(drug_a, drug_b)

        # DailyMed label flags
        flags_a = self._get_label_flags(drug_a)
        flags_b = self._get_label_flags(drug_b)

        # Data completeness score
        mol_fields = [mw_a, mw_b, lp_a, lp_b, pb_a, pb_b, hl_a, hl_b]
        completeness = sum(1 for f in mol_fields if f is not None) / len(mol_fields)

        # RxNorm known flag (well-characterised drug)
        rxnorm_a = int(drug_a.lower() in self._rxnorm_known)
        rxnorm_b = int(drug_b.lower() in self._rxnorm_known)

        return {
            # Molecular weight
            "mw_a":                       self._fill(mw_a),
            "mw_b":                       self._fill(mw_b),
            "mw_diff":                    self._fill(mw_diff),
            "mw_ratio":                   self._fill(mw_ratio, default=1.0),
            # LogP (lipophilicity)
            "logp_a":                     self._fill(lp_a),
            "logp_b":                     self._fill(lp_b),
            "logp_diff":                  self._fill(lp_diff),
            "logp_ratio":                 self._fill(lp_ratio, default=1.0),
            # Protein binding
            "protein_binding_a":          self._fill(pb_a),
            "protein_binding_b":          self._fill(pb_b),
            "protein_binding_diff":       self._fill(pb_diff),
            # Half-life
            "half_life_hours_a":          self._fill(hl_a),
            "half_life_hours_b":          self._fill(hl_b),
            "half_life_ratio":            self._fill(hl_ratio, default=1.0),
            # Drug type
            "both_small_molecule":        both_small,
            "one_biologic":               one_bio,
            # FAERS signals
            "faers_report_count":         faers.get("report_count", 0),
            "faers_death_count":          faers.get("death_count", 0),
            "faers_max_severity":         faers.get("max_severity", 0.0),
            "faers_death_rate":           faers.get("death_rate", 0.0),
            "faers_signal_strength":      faers.get("signal_strength", 0.0),
            "faers_pair_seen":            faers.get("seen", 0),
            # DailyMed label flags
            "drug_a_has_boxed_warning":   flags_a.get("has_boxed_warn", 0),
            "drug_b_has_boxed_warning":   flags_b.get("has_boxed_warn", 0),
            "either_has_boxed_warning":   int(
                flags_a.get("has_boxed_warn", 0) or flags_b.get("has_boxed_warn", 0)
            ),
            "drug_a_has_interactions":    flags_a.get("has_interactions", 0),
            "drug_b_has_interactions":    flags_b.get("has_interactions", 0),
            # RxNorm
            "drug_a_rxnorm_known":        rxnorm_a,
            "drug_b_rxnorm_known":        rxnorm_b,
            # Data quality
            "mol_data_completeness":      round(completeness, 3),
        }

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def _get_drug_props(self, drug_name: str) -> dict:
        """Look up per-drug properties by name."""
        if self._drug_props.empty:
            return {}
        key = drug_name.lower().strip()
        idx = self._name_to_idx.get(key)
        if idx is None:
            return {}
        row = self._drug_props.loc[idx]
        return row.to_dict()

    def _get_faers_signal(self, drug_a: str, drug_b: str) -> dict:
        """Look up FAERS co-occurrence signal for a drug pair."""
        if self._faers_pairs.empty:
            return {}
        pair_key = "_".join(sorted([drug_a.lower().strip(), drug_b.lower().strip()]))
        if pair_key not in self._faers_pairs.index:
            return {"seen": 0}

        row = self._faers_pairs.loc[pair_key]
        # Handle duplicate index entries — take the row with most reports
        if isinstance(row, pd.DataFrame):
            row = row.nlargest(1, "report_count").iloc[0]

        report_count = float(row.get("report_count", 0) or 0)
        death_count  = float(row.get("death_count",  0) or 0)
        max_severity = float(row.get("max_severity", 0) or 0)
        death_rate   = death_count / report_count if report_count > 0 else 0.0
        signal       = float(np.log1p(report_count))

        return {
            "seen":            1,
            "report_count":    report_count,
            "death_count":     death_count,
            "max_severity":    max_severity,
            "death_rate":      round(death_rate, 4),
            "signal_strength": round(signal, 4),
        }

    def _get_label_flags(self, drug_name: str) -> dict:
        """Look up DailyMed label safety flags for a drug."""
        if self._label_flags.empty:
            return {}
        key = drug_name.lower().strip()
        if key not in self._label_flags.index:
            return {}
        row = self._label_flags.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "has_boxed_warn":   int(bool(row.get("has_boxed_warn", False))),
            "has_interactions": int(bool(row.get("has_interactions", False))),
        }

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_half_life(text: str) -> Optional[float]:
        """Convert half-life text string → hours (float)."""
        if not text or pd.isna(text):
            return None
        text = str(text)
        for pattern, kind in HL_PATTERNS:
            m = pattern.search(text)
            if m:
                if "range" in kind:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    val = (lo + hi) / 2
                else:
                    val = float(m.group(1))
                if "days"  in kind: val *= 24
                if "weeks" in kind: val *= 168
                if "min"   in kind: val /= 60
                return round(val, 2)
        return None

    @staticmethod
    def _parse_protein_binding(text: str) -> Optional[float]:
        """Convert protein binding text → percentage float (0–100)."""
        if not text or pd.isna(text):
            return None
        text = str(text)

        # Handle qualitative labels
        qualitative = {
            "high":     90.0, "extensively": 90.0,
            "moderate": 65.0, "low":         20.0,
            "negligible": 5.0, "minimal":     5.0,
        }
        for label, val in qualitative.items():
            if label in text.lower():
                return val

        m = PB_PATTERN.search(text)
        if m:
            if m.group(1) and m.group(2):   # range
                return (float(m.group(1)) + float(m.group(2))) / 2
            elif m.group(3):                # single value
                return float(m.group(3))
            elif m.group(1):
                return float(m.group(1))
        return None

    @staticmethod
    def _safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None:
            return round(abs(a - b), 4)
        return None

    @staticmethod
    def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None and b != 0:
            return round(a / b, 4)
        return None

    @staticmethod
    def _fill(val: Optional[float], default: float = 0.0) -> float:
        """Replace None with a default sentinel value."""
        return float(val) if val is not None else default

    def save(self, output_dir: str | Path = "data/processed/") -> None:
        """Save the fitted lookup tables for inspection."""
        if not self._fitted:
            raise RuntimeError("Call fit() before save()")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save parsed drug properties
        props_path = out / "molecular_props.parquet"
        self._drug_props.to_parquet(props_path, index=False)
        logger.info(f"💾 Saved molecular properties → {props_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract molecular features for drug pairs")
    ap.add_argument("--drugbank",  default="data/processed/drugbank_drugs.parquet")
    ap.add_argument("--faers",     default="data/processed/faers_co_occurrence.parquet")
    ap.add_argument("--dailymed",  default="data/processed/dailymed_labels.parquet")
    ap.add_argument("--rxnorm",    default="data/processed/rxnorm_mapping.parquet")
    ap.add_argument("--pairs",     default="data/processed/drug_pairs.parquet")
    ap.add_argument("--out",       default="data/processed/")
    ap.add_argument("--sample",    type=int, default=5000,
                    help="Rows to test transform on (default 5000)")
    args = ap.parse_args()

    extractor = MolecularFeatureExtractor()
    extractor.fit(
        drugbank_path = args.drugbank,
        faers_path    = args.faers,
        dailymed_path = args.dailymed,
        rxnorm_path   = args.rxnorm,
    )
    extractor.save(args.out)

    # Test transform on a sample
    pairs_df = pd.read_parquet(args.pairs).head(args.sample)
    features_df = extractor.transform(pairs_df)

    mol_cols = [
        "drug_a_name", "drug_b_name", "severity",
        "mw_a", "logp_a", "protein_binding_a", "half_life_hours_a",
        "faers_report_count", "faers_death_rate",
        "either_has_boxed_warning", "mol_data_completeness",
    ]
    available = [c for c in mol_cols if c in features_df.columns]

    print(f"\n── Molecular feature sample ────────────────────────────────")
    print(features_df[available].head(10).to_string(index=False))

    print(f"\n── Feature completeness ────────────────────────────────────")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    non_zero = (features_df[numeric_cols] != 0).mean()
    print(non_zero.sort_values(ascending=False).head(15).to_string())

    print(f"\n── Half-life parsing sample ────────────────────────────────")
    if "half_life" in extractor._drug_props.columns:
        hl_df = extractor._drug_props[["name", "half_life", "half_life_hours"]].dropna(
            subset=["half_life_hours"]
        ).head(10)
        print(hl_df.to_string(index=False))

    print(f"\n── FAERS signal summary ────────────────────────────────────")
    faers_cols = [c for c in features_df.columns if "faers" in c]
    seen = features_df[features_df["faers_pair_seen"] == 1]
    print(f"Pairs with FAERS signal: {len(seen):,} / {len(features_df):,}")
    if not seen.empty:
        print(seen[faers_cols].describe().round(2).to_string())