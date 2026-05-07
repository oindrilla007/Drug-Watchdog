"""
ml/features/feature_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Unified feature pipeline that:
  1. Loads drug pairs (labels) from DrugBank
  2. Runs CYP450FeatureExtractor   → 12 features
  3. Runs MolecularFeatureExtractor → 32 features
  4. Cleans, imputes, and scales the combined 44-feature matrix
  5. Splits into train / validation / test sets
  6. Saves everything needed for Colab training

Output files (saved to data/processed/):
  feature_matrix.parquet     ← full 44-feature matrix with labels
  X_train.npy                ← training features
  X_val.npy                  ← validation features
  X_test.npy                 ← test features
  y_train.npy                ← training labels (severity 0-3)
  y_val.npy
  y_test.npy
  feature_names.json         ← ordered list of feature column names
  feature_scaler.pkl         ← fitted StandardScaler (for inference)
  split_indices.json         ← train/val/test indices for reproducibility

Run on: VS Code
  python -m ml.features.feature_pipeline

This is the LAST step before Colab — after this you upload the .npy files
to Google Drive and run the training notebooks.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Label column ──────────────────────────────────────────────────────────────
LABEL_COL = "severity"   # integer 0, 1, 2, 3

# ── Columns to drop before building feature matrix ───────────────────────────
# These are identifiers / text — not numeric features
DROP_COLS = [
    "drug_a_id", "drug_b_id",
    "drug_a_name", "drug_b_name",
    "description",
    "pair_key",
    "cyp_description_snippet",
]

# ── Imputation strategy per feature type ─────────────────────────────────────
# median for continuous, 0 for binary/count
MEDIAN_FEATURES = [
    "mw_a", "mw_b", "mw_diff", "mw_ratio",
    "logp_a", "logp_b", "logp_diff", "logp_ratio",
    "protein_binding_a", "protein_binding_b", "protein_binding_diff",
    "half_life_hours_a", "half_life_hours_b", "half_life_ratio",
]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.
    Combines CYP450 + molecular features, cleans data, and
    produces train/val/test splits ready for model training.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed/",
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._feature_matrix: Optional[pd.DataFrame] = None
        self._feature_names:  Optional[list[str]]    = None
        self._scaler:         Optional[StandardScaler] = None
        self._class_weights:  Optional[dict] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        interactions_path:   str | Path = "data/processed/drug_pairs.parquet",
        drugbank_drugs_path: str | Path = "data/processed/drugbank_drugs.parquet",
        dailymed_path:       str | Path = "data/processed/dailymed_chunks.parquet",
        dailymed_labels_path:str | Path = "data/processed/dailymed_labels.parquet",
        faers_cooccur_path:  str | Path = "data/processed/faers_co_occurrence.parquet",
        rxnorm_path:         str | Path = "data/processed/rxnorm_mapping.parquet",
        max_pairs:           Optional[int] = None,
        min_severity:        int = 0,
    ) -> "FeaturePipeline":
        """
        Full pipeline: load → extract features → clean → split → save.

        max_pairs:    cap for dev testing (None = all pairs)
        min_severity: only keep pairs with severity >= this value
                      Set to 1 to drop unknown-severity pairs from training.
        """
        logger.info("=" * 60)
        logger.info("  FEATURE PIPELINE — Phase 2")
        logger.info("=" * 60)

        # ── Step 1: Load interaction pairs (labels) ───────────────────────────
        logger.info("Step 1: Loading drug interaction pairs …")
        pairs_df = self._load_pairs(
            interactions_path, max_pairs=max_pairs, min_severity=min_severity
        )

        # ── Step 2: CYP450 features ───────────────────────────────────────────
        logger.info("Step 2: Extracting CYP450 features …")
        from ml.features.cyp450_features import CYP450FeatureExtractor
        cyp_extractor = CYP450FeatureExtractor()
        cyp_extractor.fit(
            interactions_path   = interactions_path,
            dailymed_path       = dailymed_path,
            drugbank_drugs_path = drugbank_drugs_path,
        )
        pairs_df = cyp_extractor.transform(pairs_df)
        cyp_extractor.save(self.output_dir / "cyp450_features.parquet")

        # ── Step 3: Molecular features ────────────────────────────────────────
        logger.info("Step 3: Extracting molecular features …")
        from ml.features.molecular_features import MolecularFeatureExtractor
        mol_extractor = MolecularFeatureExtractor()
        mol_extractor.fit(
            drugbank_path = drugbank_drugs_path,
            faers_path    = faers_cooccur_path,
            dailymed_path = dailymed_labels_path,
            rxnorm_path   = rxnorm_path,
        )
        pairs_df = mol_extractor.transform(pairs_df)
        mol_extractor.save(self.output_dir)

        # ── Step 4: Clean + impute ────────────────────────────────────────────
        logger.info("Step 4: Cleaning and imputing feature matrix …")
        pairs_df = self._clean(pairs_df)

        # ── Step 5: Build feature matrix ──────────────────────────────────────
        logger.info("Step 5: Building feature matrix …")
        X, y, feature_names = self._build_matrix(pairs_df)
        self._feature_names = feature_names

        # ── Step 6: Scale ─────────────────────────────────────────────────────
        logger.info("Step 6: Fitting StandardScaler …")
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # ── Step 7: Train / val / test split ──────────────────────────────────
        logger.info("Step 7: Splitting into train/val/test …")
        splits = self._split(X_scaled, y, pairs_df)

        # ── Step 8: Compute class weights ─────────────────────────────────────
        classes = np.unique(splits["y_train"])
        weights = compute_class_weight("balanced", classes=classes, y=splits["y_train"])
        self._class_weights = dict(zip(classes.tolist(), weights.tolist()))
        logger.info(f"  Class weights: {self._class_weights}")

        # ── Step 9: Save everything ───────────────────────────────────────────
        logger.info("Step 8: Saving outputs …")
        self._feature_matrix = pairs_df
        self._save_all(splits, pairs_df)

        logger.info("=" * 60)
        logger.info("  FEATURE PIPELINE COMPLETE")
        logger.info(f"  Total pairs:    {len(pairs_df):,}")
        logger.info(f"  Features:       {len(feature_names)}")
        logger.info(f"  Train samples:  {len(splits['y_train']):,}")
        logger.info(f"  Val samples:    {len(splits['y_val']):,}")
        logger.info(f"  Test samples:   {len(splits['y_test']):,}")
        logger.info("=" * 60)
        return self

    # ── Private steps ─────────────────────────────────────────────────────────

    def _load_pairs(
        self,
        path: str | Path,
        max_pairs: Optional[int],
        min_severity: int,
    ) -> pd.DataFrame:
        df = pd.read_parquet(path)
        logger.info(f"  Loaded {len(df):,} drug pairs")

        # Keep only labelled pairs
        df = df[df[LABEL_COL].notna()].copy()
        df[LABEL_COL] = df[LABEL_COL].astype(int)

        if min_severity > 0:
            before = len(df)
            df = df[df[LABEL_COL] >= min_severity]
            logger.info(f"  Filtered to severity >= {min_severity}: {len(df):,} (dropped {before-len(df):,})")

        if max_pairs:
            df = df.head(max_pairs)
            logger.info(f"  Capped at {max_pairs:,} pairs (dev mode)")

        # Label distribution
        dist = df[LABEL_COL].value_counts().sort_index()
        logger.info(f"  Label distribution:\n{dist.to_string()}")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and handle edge cases.
        Strategy: median imputation for continuous molecular features,
        0-fill for count/binary/flag features.
        """
        # Median imputation for continuous features
        for col in MEDIAN_FEATURES:
            if col in df.columns:
                median_val = df[col].replace(0, np.nan).median()
                df[col] = df[col].replace(0, np.nan).fillna(
                    median_val if pd.notna(median_val) else 0.0
                )

        # Fill remaining NaN with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        # Cap extreme values (molecular weight outliers, etc.)
        if "mw_a" in df.columns:
            cap = df["mw_a"].quantile(0.999)
            df["mw_a"] = df["mw_a"].clip(upper=cap)
            df["mw_b"] = df["mw_b"].clip(upper=cap)

        if "faers_report_count" in df.columns:
            cap = df["faers_report_count"].quantile(0.999)
            df["faers_report_count"] = df["faers_report_count"].clip(upper=cap)

        logger.info(f"  After cleaning: {df.shape[0]:,} rows, {df.shape[1]} cols")
        return df

    def _build_matrix(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract feature matrix X and label vector y from the DataFrame.
        Returns (X, y, feature_names).
        """
        # Drop non-feature columns
        drop = [c for c in DROP_COLS if c in df.columns]
        drop.append(LABEL_COL)

        feature_cols = [
            c for c in df.columns
            if c not in drop
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]
        ]

        # Remove any remaining text / object columns
        feature_cols = [
            c for c in feature_cols
            if df[c].dtype != object
        ]

        X = df[feature_cols].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.int32)

        logger.info(f"  Feature matrix: {X.shape} | Labels: {y.shape}")
        logger.info(f"  Features selected: {feature_cols}")
        return X, y, feature_cols

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Stratified train / val / test split.
        Stratification ensures all severity classes are represented in each split.
        """
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(y)),
            test_size   = VAL_RATIO + TEST_RATIO,
            random_state = self.random_seed,
            stratify    = y,
        )

        # Second split: val vs test
        relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size    = relative_test,
            random_state = self.random_seed,
            stratify     = y_temp,
        )

        logger.info(
            f"  Split: train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}"
        )

        # Verify class distributions
        for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            unique, counts = np.unique(split_y, return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            logger.info(f"  {split_name} label dist: {dist}")

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
        }

    def _save_all(self, splits: dict, df: pd.DataFrame) -> None:
        out = self.output_dir

        # NumPy arrays — uploaded to Colab/Kaggle for training
        for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            path = out / f"{name}.npy"
            np.save(str(path), splits[name])
            logger.info(f"  Saved {name}.npy → {path}  shape={splits[name].shape}")

        # Feature names — needed to interpret SHAP values
        fn_path = out / "feature_names.json"
        with open(fn_path, "w") as f:
            json.dump(self._feature_names, f, indent=2)
        logger.info(f"  Saved feature_names.json ({len(self._feature_names)} features)")

        # Fitted scaler — needed for inference time
        scaler_path = out / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"  Saved feature_scaler.pkl")

        # Class weights — passed to XGBoost / GNN training
        cw_path = out / "class_weights.json"
        with open(cw_path, "w") as f:
            json.dump(self._class_weights, f, indent=2)
        logger.info(f"  Saved class_weights.json: {self._class_weights}")

        # Split indices — for reproducibility / traceability
        idx_path = out / "split_indices.json"
        with open(idx_path, "w") as f:
            json.dump({
                "train": splits["idx_train"].tolist(),
                "val":   splits["idx_val"].tolist(),
                "test":  splits["idx_test"].tolist(),
            }, f)
        logger.info(f"  Saved split_indices.json")

        # Full feature matrix as Parquet — useful for EDA in notebooks
        fm_path = out / "feature_matrix.parquet"
        df.to_parquet(fm_path, index=False)
        logger.info(f"  Saved feature_matrix.parquet ({len(df):,} rows)")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Run pipeline first.")
        return self._feature_names

    def get_class_weights(self) -> dict:
        if self._class_weights is None:
            raise RuntimeError("Run pipeline first.")
        return self._class_weights

    def summary(self) -> None:
        """Print a quick summary of what was produced."""
        if self._feature_matrix is None:
            logger.warning("Pipeline has not been run yet.")
            return
        df = self._feature_matrix
        print("\n── Feature Pipeline Summary ────────────────────────────────")
        print(f"Total pairs in matrix: {len(df):,}")
        print(f"Feature count:         {len(self._feature_names)}")
        print(f"\nLabel distribution:")
        print(df[LABEL_COL].value_counts().sort_index().to_string())
        print(f"\nClass weights (for imbalanced training):")
        print(self._class_weights)
        print(f"\nFeature list:")
        for i, name in enumerate(self._feature_names, 1):
            print(f"  {i:>2}. {name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build feature matrix for drug interaction severity classification"
    )
    ap.add_argument("--interactions",    default="data/processed/drug_pairs.parquet")
    ap.add_argument("--drugbank-drugs",  default="data/processed/drugbank_drugs.parquet")
    ap.add_argument("--dailymed-chunks", default="data/processed/dailymed_chunks.parquet")
    ap.add_argument("--dailymed-labels", default="data/processed/dailymed_labels.parquet")
    ap.add_argument("--faers-cooccur",   default="data/processed/faers_co_occurrence.parquet")
    ap.add_argument("--rxnorm",          default="data/processed/rxnorm_mapping.parquet")
    ap.add_argument("--out",             default="data/processed/")
    ap.add_argument("--max-pairs",       type=int, default=None,
                    help="Cap pairs for dev mode (e.g. 50000)")
    ap.add_argument("--min-severity",    type=int, default=0,
                    help="Only keep pairs with severity >= N")
    args = ap.parse_args()

    pipeline = FeaturePipeline(output_dir=args.out)
    pipeline.run(
        interactions_path    = args.interactions,
        drugbank_drugs_path  = args.drugbank_drugs,
        dailymed_path        = args.dailymed_chunks,
        dailymed_labels_path = args.dailymed_labels,
        faers_cooccur_path   = args.faers_cooccur,
        rxnorm_path          = args.rxnorm,
        max_pairs            = args.max_pairs,
        min_severity         = args.min_severity,
    )
    pipeline.summary()

    print("\n── Files saved to data/processed/ ──────────────────────────")
    print("Upload these to Google Drive before running Colab notebooks:")
    for fname in [
        "X_train.npy", "X_val.npy", "X_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "feature_names.json", "class_weights.json",
    ]:
        fpath = Path(args.out) / fname
        size  = fpath.stat().st_size / 1024 if fpath.exists() else 0
        print(f"  {'OK' if fpath.exists() else 'MISSING':<8} {fname:<30} {size:>8.1f} KB")