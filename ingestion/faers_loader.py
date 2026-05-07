"""
ingestion/faers_loader.py
────────────────────────────────────────────────────────────────────────────────
Loads and normalises FDA FAERS (FDA Adverse Event Reporting System) quarterly
CSV dumps and extracts:
  - Drug-adverse event pairs
  - Drug-drug co-occurrence in the same report (signal for interactions)
  - Outcome severity per report (death / hospitalisation / disability …)
  - De-duplicated, cleaned report table ready for ML feature engineering

FAERS quarterly downloads (free, no login):
  https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
  → Download the latest 2–4 quarters as ZIP files
  → Unzip each quarter into its own folder:

  data/raw/faers/
  ├── 2024Q4/
  │   ├── DEMO24Q4.txt   ← patient demographics
  │   ├── DRUG24Q4.txt   ← drugs in each report
  │   ├── REAC24Q4.txt   ← adverse reactions (MedDRA terms)
  │   ├── OUTC24Q4.txt   ← outcomes (death, hosp, etc.)
  │   ├── INDI24Q4.txt   ← indications (why drug was taken)
  │   └── RPSR24Q4.txt   ← report source
  ├── 2024Q3/
  │   └── ...
  └── 2025Q1/
      └── ...

Usage:
  from ingestion.faers_loader import FAERSLoader

  loader = FAERSLoader("data/raw/faers/")
  loader.load()

  reports_df    = loader.get_reports_df()
  drug_events   = loader.get_drug_event_pairs()
  co_occurrence = loader.get_drug_co_occurrence()
  loader.save_all("data/processed/")
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── FAERS file prefixes per table ─────────────────────────────────────────────
# Each quarter folder contains these files (case-insensitive prefix match)
FAERS_TABLES = {
    "demo": "demographics",
    "drug": "drugs",
    "reac": "reactions",
    "outc": "outcomes",
    "indi": "indications",
}

# ── Outcome codes → human-readable severity ───────────────────────────────────
OUTCOME_MAP = {
    "DE": "death",
    "LT": "life_threatening",
    "HO": "hospitalization",
    "DS": "disability",
    "CA": "congenital_anomaly",
    "RI": "required_intervention",
    "OT": "other",
}

# Numeric severity weight for sorting / ML features
OUTCOME_SEVERITY = {
    "death":               5,
    "life_threatening":    4,
    "hospitalization":     3,
    "disability":          3,
    "congenital_anomaly":  3,
    "required_intervention": 2,
    "other":               1,
}

# ── Drug role codes ───────────────────────────────────────────────────────────
# PS = primary suspect, SS = secondary suspect, C = concomitant, I = interacting
SUSPECT_ROLES = {"PS", "SS", "I"}

# ── Minimum report count for a drug-drug pair to be kept ─────────────────────
MIN_CO_OCCURRENCE_COUNT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

class FAERSLoader:
    """
    Loads multiple FAERS quarterly CSV dumps, merges them, deduplicates
    reports, and produces analysis-ready DataFrames.
    """

    def __init__(self, faers_dir: str | Path) -> None:
        self.faers_dir = Path(faers_dir)
        if not self.faers_dir.exists():
            raise FileNotFoundError(
                f"FAERS directory not found: {self.faers_dir}\n"
                "Download quarterly files from:\n"
                "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
            )

        # Raw merged tables
        self._demo_df: Optional[pd.DataFrame] = None
        self._drug_df: Optional[pd.DataFrame] = None
        self._reac_df: Optional[pd.DataFrame] = None
        self._outc_df: Optional[pd.DataFrame] = None

        # Output DataFrames
        self._reports_df:      Optional[pd.DataFrame] = None
        self._drug_events_df:  Optional[pd.DataFrame] = None
        self._co_occur_df:     Optional[pd.DataFrame] = None

        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "FAERSLoader":
        """Discover all quarter folders, load CSVs, merge, and clean."""
        quarter_dirs = self._discover_quarters()
        if not quarter_dirs:
            raise FileNotFoundError(
                f"No FAERS quarter folders found under {self.faers_dir}\n"
                "Expected subfolders like: 2024Q4/, 2025Q1/ etc."
            )

        logger.info(f"Found {len(quarter_dirs)} FAERS quarter(s): "
                    f"{[d.name for d in quarter_dirs]}")

        demo_parts, drug_parts, reac_parts, outc_parts = [], [], [], []

        for qdir in quarter_dirs:
            logger.info(f"  Loading quarter: {qdir.name}")
            tables = self._load_quarter(qdir)
            if "demo" in tables: demo_parts.append(tables["demo"])
            if "drug" in tables: drug_parts.append(tables["drug"])
            if "reac" in tables: reac_parts.append(tables["reac"])
            if "outc" in tables: outc_parts.append(tables["outc"])

        # Merge all quarters
        self._demo_df = pd.concat(demo_parts, ignore_index=True) if demo_parts else pd.DataFrame()
        self._drug_df = pd.concat(drug_parts, ignore_index=True) if drug_parts else pd.DataFrame()
        self._reac_df = pd.concat(reac_parts, ignore_index=True) if reac_parts else pd.DataFrame()
        self._outc_df = pd.concat(outc_parts, ignore_index=True) if outc_parts else pd.DataFrame()

        logger.info(
            f"Raw merged — reports: {len(self._demo_df):,} | "
            f"drug records: {len(self._drug_df):,} | "
            f"reactions: {len(self._reac_df):,}"
        )

        # Clean + build output tables
        self._clean_and_build()
        self._loaded = True

        logger.info(
            f"✅ FAERS load complete — "
            f"{len(self._reports_df):,} unique reports | "
            f"{len(self._drug_events_df):,} drug-event pairs | "
            f"{len(self._co_occur_df):,} drug co-occurrence pairs"
        )
        return self

    def get_reports_df(self) -> pd.DataFrame:
        self._check_loaded()
        return self._reports_df

    def get_drug_event_pairs(self) -> pd.DataFrame:
        """
        Returns a DataFrame of (drug_name, reaction_term, outcome_severity, report_count).
        Used to enrich the RAG knowledge base with real-world adverse event signals.
        """
        self._check_loaded()
        return self._drug_events_df

    def get_drug_co_occurrence(self) -> pd.DataFrame:
        """
        Returns drug pairs that appear together as suspects in the same reports.
        This is a pharmacovigilance signal for potential interactions.
        Columns: drug_a, drug_b, report_count, death_count, hosp_count, max_severity
        """
        self._check_loaded()
        return self._co_occur_df

    def save_all(self, output_dir: str | Path = "data/processed/") -> None:
        self._check_loaded()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._reports_df.to_parquet(out / "faers_reports.parquet", index=False)
        logger.info(f"💾 Saved {len(self._reports_df):,} reports → {out / 'faers_reports.parquet'}")

        self._drug_events_df.to_parquet(out / "faers_drug_events.parquet", index=False)
        logger.info(f"💾 Saved {len(self._drug_events_df):,} drug-event pairs → {out / 'faers_drug_events.parquet'}")

        self._co_occur_df.to_parquet(out / "faers_co_occurrence.parquet", index=False)
        logger.info(f"💾 Saved {len(self._co_occur_df):,} co-occurrence pairs → {out / 'faers_co_occurrence.parquet'}")

    # ── Private: discovery & loading ─────────────────────────────────────────

    def _discover_quarters(self) -> list[Path]:
        """Find all subdirectories that look like FAERS quarter folders."""
        pattern = re.compile(r"^\d{4}Q[1-4]$", re.IGNORECASE)
        quarters = [
            d for d in self.faers_dir.iterdir()
            if d.is_dir() and pattern.match(d.name)
        ]
        return sorted(quarters)

    def _load_quarter(self, qdir: Path) -> dict[str, pd.DataFrame]:
        """Load all FAERS tables from one quarter directory."""
        tables: dict[str, pd.DataFrame] = {}
        files  = {f.stem.lower(): f for f in qdir.glob("*.txt")}
        # Also handle .csv extension
        files.update({f.stem.lower(): f for f in qdir.glob("*.csv")})

        for prefix, table_name in FAERS_TABLES.items():
            # Find the file whose name starts with this prefix
            matched = next(
                (path for stem, path in files.items() if stem.startswith(prefix)),
                None
            )
            if matched is None:
                logger.debug(f"  ⚠ Missing {prefix}* in {qdir.name}")
                continue

            try:
                df = self._read_faers_csv(matched)
                df.columns = df.columns.str.lower().str.strip()
                df["quarter"] = qdir.name
                tables[prefix] = df
                logger.debug(f"  ✓ {matched.name}: {len(df):,} rows")
            except Exception as exc:
                logger.warning(f"  ⚠ Failed to load {matched.name}: {exc}")

        return tables

    @staticmethod
    def _read_faers_csv(path: Path) -> pd.DataFrame:
        """
        FAERS files use '$' as delimiter and have inconsistent encodings.
        Try multiple encodings gracefully.
        """
        for encoding in ("latin-1", "utf-8", "cp1252"):
            try:
                return pd.read_csv(
                    path,
                    sep="$",
                    encoding=encoding,
                    dtype=str,          # read everything as string first
                    low_memory=False,
                )
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {path} with any known encoding")

    # ── Private: cleaning & building ─────────────────────────────────────────

    def _clean_and_build(self) -> None:
        """Orchestrates all cleaning and output-table construction."""
        self._clean_demo()
        self._clean_drugs()
        self._clean_reactions()
        self._clean_outcomes()
        self._build_reports()
        self._build_drug_events()
        self._build_co_occurrence()

    def _clean_demo(self) -> None:
        if self._demo_df.empty:
            return
        df = self._demo_df.copy()

        # Standardise primary key column name (varies across quarters)
        for col in ("primaryid", "isr"):
            if col in df.columns:
                df = df.rename(columns={col: "primaryid"})
                break

        # Deduplicate: keep the latest caseid version
        if "caseid" in df.columns and "caseversion" in df.columns:
            df["caseversion"] = pd.to_numeric(df["caseversion"], errors="coerce").fillna(0)
            df = (
                df.sort_values("caseversion", ascending=False)
                  .drop_duplicates(subset="caseid", keep="first")
            )

        # Age normalisation → numeric years
        if "age" in df.columns and "age_cod" in df.columns:
            df["age_years"] = df.apply(self._normalise_age, axis=1)
        else:
            df["age_years"] = None

        self._demo_df = df

    def _clean_drugs(self) -> None:
        if self._drug_df.empty:
            return
        df = self._drug_df.copy()

        # Normalise drug name
        name_col = next((c for c in df.columns if "drugname" in c.lower()), None)
        if name_col:
            df = df.rename(columns={name_col: "drugname"})
            df["drugname"] = (
                df["drugname"]
                .str.upper()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

        # Keep only suspect drugs (primary / secondary / interacting)
        if "role_cod" in df.columns:
            df = df[df["role_cod"].isin(SUSPECT_ROLES)]

        self._drug_df = df

    def _clean_reactions(self) -> None:
        if self._reac_df.empty:
            return
        df = self._reac_df.copy()

        # Normalise reaction term (MedDRA preferred term)
        pt_col = next((c for c in df.columns if "pt" in c.lower()), None)
        if pt_col and pt_col != "pt":
            df = df.rename(columns={pt_col: "pt"})

        if "pt" in df.columns:
            df["pt"] = df["pt"].str.upper().str.strip()

        self._reac_df = df

    def _clean_outcomes(self) -> None:
        if self._outc_df.empty:
            return
        df = self._outc_df.copy()

        if "outc_cod" in df.columns:
            df["outcome_label"]    = df["outc_cod"].map(OUTCOME_MAP).fillna("other")
            df["outcome_severity"] = df["outcome_label"].map(OUTCOME_SEVERITY).fillna(1)

        self._outc_df = df

    def _build_reports(self) -> None:
        """
        Master report table: one row per unique report with demographics
        and worst outcome attached.
        """
        if self._demo_df.empty:
            self._reports_df = pd.DataFrame()
            return

        df = self._demo_df.copy()

        # Attach worst outcome per report
        if not self._outc_df.empty and "primaryid" in self._outc_df.columns:
            worst = (
                self._outc_df
                .groupby("primaryid")
                .agg(
                    worst_outcome      = ("outcome_label", lambda x: x.iloc[x.map(OUTCOME_SEVERITY).argmax()]),
                    max_severity_score = ("outcome_severity", "max"),
                )
                .reset_index()
            )
            df = df.merge(worst, on="primaryid", how="left")

        keep_cols = [c for c in [
            "primaryid", "caseid", "caseversion", "quarter",
            "age_years", "sex", "reporter_country",
            "worst_outcome", "max_severity_score",
        ] if c in df.columns]

        self._reports_df = df[keep_cols].copy()

    def _build_drug_events(self) -> None:
        """
        Drug-event pair table: how often does drug X appear with reaction Y,
        and how severe are those outcomes on average?
        """
        if self._drug_df.empty or self._reac_df.empty:
            self._drug_events_df = pd.DataFrame()
            return

        pid = "primaryid"
        drug_cols = [c for c in [pid, "drugname"] if c in self._drug_df.columns]
        reac_cols = [c for c in [pid, "pt"]       if c in self._reac_df.columns]

        if len(drug_cols) < 2 or len(reac_cols) < 2:
            self._drug_events_df = pd.DataFrame()
            return

        merged = self._drug_df[drug_cols].merge(
            self._reac_df[reac_cols], on=pid, how="inner"
        )

        # Attach outcome severity
        if not self._outc_df.empty and pid in self._outc_df.columns:
            worst = (
                self._outc_df
                .groupby(pid)["outcome_severity"]
                .max()
                .reset_index()
                .rename(columns={"outcome_severity": "max_severity"})
            )
            merged = merged.merge(worst, on=pid, how="left")
            merged["max_severity"] = merged["max_severity"].fillna(1)
        else:
            merged["max_severity"] = 1

        # Aggregate
        agg = (
            merged
            .groupby(["drugname", "pt"])
            .agg(
                report_count   = (pid, "nunique"),
                avg_severity   = ("max_severity", "mean"),
                max_severity   = ("max_severity", "max"),
            )
            .reset_index()
            .rename(columns={"pt": "reaction_term"})
            .sort_values("report_count", ascending=False)
        )

        self._drug_events_df = agg

    def _build_co_occurrence(self) -> None:
        """
        Drug co-occurrence table: pairs of suspect drugs in the same report.
        High co-occurrence of two drugs with severe outcomes is a
        pharmacovigilance signal that feeds our RAG evidence base.
        """
        if self._drug_df.empty:
            self._co_occur_df = pd.DataFrame()
            return

        pid = "primaryid"
        if pid not in self._drug_df.columns or "drugname" not in self._drug_df.columns:
            self._co_occur_df = pd.DataFrame()
            return

        # Self-join on primaryid to get all drug pairs per report
        df = self._drug_df[[pid, "drugname"]].drop_duplicates()

        pairs = df.merge(df, on=pid, suffixes=("_a", "_b"))
        # Remove self-pairs and keep canonical order (A < B alphabetically)
        pairs = pairs[pairs["drugname_a"] < pairs["drugname_b"]]

        # Attach outcome info
        if not self._outc_df.empty and pid in self._outc_df.columns:
            outc = self._outc_df[[pid, "outcome_label", "outcome_severity"]].copy()
            pairs = pairs.merge(outc, on=pid, how="left")
        else:
            pairs["outcome_label"]    = "other"
            pairs["outcome_severity"] = 1

        # Aggregate
        agg = (
            pairs
            .groupby(["drugname_a", "drugname_b"])
            .agg(
                report_count  = (pid, "nunique"),
                death_count   = ("outcome_label", lambda x: (x == "death").sum()),
                hosp_count    = ("outcome_label", lambda x: (x == "hospitalization").sum()),
                max_severity  = ("outcome_severity", "max"),
            )
            .reset_index()
            .rename(columns={"drugname_a": "drug_a", "drugname_b": "drug_b"})
        )

        # Filter noise — keep pairs with enough reports to be meaningful
        agg = agg[agg["report_count"] >= MIN_CO_OCCURRENCE_COUNT]
        agg = agg.sort_values("report_count", ascending=False)

        self._co_occur_df = agg

    # ── Private: utilities ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_age(row: pd.Series) -> Optional[float]:
        """Convert FAERS age + age_cod to numeric years."""
        try:
            age = float(row.get("age", None))
        except (TypeError, ValueError):
            return None

        code = str(row.get("age_cod", "YR")).upper().strip()
        conversion = {
            "YR": 1.0,
            "DEC": 10.0,
            "MON": 1 / 12,
            "WK":  1 / 52,
            "DY":  1 / 365,
            "HR":  1 / 8760,
        }
        factor = conversion.get(code, 1.0)
        result = age * factor
        # Sanity check: ages outside 0-120 are data errors
        return result if 0 <= result <= 120 else None

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing data.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load and clean FDA FAERS quarterly data")
    ap.add_argument(
        "--dir",
        default="data/raw/faers/",
        help="Directory containing FAERS quarter subfolders (e.g. 2024Q4/)",
    )
    ap.add_argument(
        "--out",
        default="data/processed/",
        help="Output directory for Parquet files",
    )
    args = ap.parse_args()

    loader = FAERSLoader(args.dir)
    loader.load()
    loader.save_all(args.out)

    # Quick sanity report
    reports_df = loader.get_reports_df()
    drug_events = loader.get_drug_event_pairs()
    co_occur    = loader.get_drug_co_occurrence()

    print("\n── Reports sample ────────────────────────────────────")
    print(reports_df[["primaryid", "age_years", "sex", "worst_outcome"]].head(10))

    print("\n── Top drug-event pairs ──────────────────────────────")
    print(drug_events[["drugname", "reaction_term", "report_count", "max_severity"]].head(15))

    print("\n── Top drug co-occurrence pairs ──────────────────────")
    print(co_occur[["drug_a", "drug_b", "report_count", "death_count", "max_severity"]].head(15))

    print(f"\n── Outcome distribution ──────────────────────────────")
    if "worst_outcome" in reports_df.columns:
        print(reports_df["worst_outcome"].value_counts())