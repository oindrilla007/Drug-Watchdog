"""
ml/training/train_xgboost.py
────────────────────────────────────────────────────────────────────────────────
Full XGBoost training pipeline for drug interaction severity classification.

Run on: Google Colab (notebook 03) OR VS Code
  python -m ml.training.train_xgboost

Stages:
  1. Load train/val/test splits from .npy files
  2. (Optional) Optuna hyperparameter search — 50 trials, ~15 min on Colab
  3. Train final model with best params + early stopping
  4. Evaluate on val + test sets
  5. Generate SHAP feature importance plot
  6. Save model artifact to ml/artifacts/

Colab usage:
  Upload to Google Drive:
    data/processed/X_train.npy, X_val.npy, X_test.npy
    data/processed/y_train.npy, y_val.npy, y_test.npy
    data/processed/feature_names.json
    data/processed/class_weights.json
  Then run notebook 03 which calls this script.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostTrainer:

    def __init__(
        self,
        data_dir:     str | Path = "data/processed/",
        artifact_dir: str | Path = "ml/artifacts/",
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.feature_names = []
        self.class_weights = {}

    # ── Load data ─────────────────────────────────────────────────────────────

    def load_data(self) -> "XGBoostTrainer":
        logger.info("Loading train/val/test splits …")

        self.X_train = np.load(self.data_dir / "X_train.npy")
        self.X_val   = np.load(self.data_dir / "X_val.npy")
        self.X_test  = np.load(self.data_dir / "X_test.npy")
        self.y_train = np.load(self.data_dir / "y_train.npy")
        self.y_val   = np.load(self.data_dir / "y_val.npy")
        self.y_test  = np.load(self.data_dir / "y_test.npy")

        with open(self.data_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)
        with open(self.data_dir / "class_weights.json") as f:
            self.class_weights = {int(k): v for k, v in json.load(f).items()}

        logger.info(f"  X_train: {self.X_train.shape} | y_train: {self.y_train.shape}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Class weights: {self.class_weights}")

        # Label distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        logger.info(f"  Train label dist: {dict(zip(unique.tolist(), counts.tolist()))}")
        return self

    # ── Optuna tuning ─────────────────────────────────────────────────────────

    def tune(self, n_trials: int = 50) -> dict:
        """Run Optuna hyperparameter search. Returns best params."""
        logger.info(f"Starting Optuna search ({n_trials} trials) …")
        from ml.models.xgboost_classifier import DrugInteractionXGB

        best_params = DrugInteractionXGB.tune(
            X_train       = self.X_train,
            y_train       = self.y_train,
            X_val         = self.X_val,
            y_val         = self.y_val,
            n_trials      = n_trials,
            class_weights = self.class_weights,
        )

        # Save best params
        params_path = self.artifact_dir / "xgb_best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"💾 Best params saved → {params_path}")
        return best_params

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        params:         dict  | None = None,
        early_stopping: int          = 50,
        use_mlflow:     bool         = False,
    ) -> "XGBoostTrainer":
        """
        Train the final XGBoost model.
        params: if None, loads from xgb_best_params.json or uses defaults.
        """
        from ml.models.xgboost_classifier import DrugInteractionXGB, DEFAULT_PARAMS

        # Load tuned params if available
        if params is None:
            best_path = self.artifact_dir / "xgb_best_params.json"
            if best_path.exists():
                with open(best_path) as f:
                    params = json.load(f)
                logger.info(f"Loaded tuned params from {best_path}")
            else:
                params = DEFAULT_PARAMS
                logger.info("Using default params (no tuning results found)")

        # MLflow tracking (optional)
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("drug_watchdog_xgboost")
                mlflow.start_run()
                mlflow.log_params(params)
            except ImportError:
                logger.warning("mlflow not installed — skipping tracking")
                use_mlflow = False

        logger.info("Training final XGBoost model …")
        t0 = time.time()

        self._model = DrugInteractionXGB(params=params)
        self._model.fit(
            X_train       = self.X_train,
            y_train       = self.y_train,
            X_val         = self.X_val,
            y_val         = self.y_val,
            feature_names = self.feature_names,
            class_weights = self.class_weights,
            early_stopping = early_stopping,
        )

        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed:.1f}s")

        if use_mlflow:
            mlflow.log_metric("train_time_s", elapsed)

        return self

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, use_mlflow: bool = False) -> dict:
        """Evaluate on val and test sets. Returns metrics dict."""
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, average_precision_score,
        )
        from sklearn.preprocessing import label_binarize

        results = {}

        for split_name, X, y in [
            ("val",  self.X_val,  self.y_val),
            ("test", self.X_test, self.y_test),
        ]:
            y_pred  = self._model.predict(X)
            y_proba = self._model.predict_proba(X)

            # Classification report
            present_classes = sorted(np.unique(y).tolist())
            label_map = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
            target_names = [label_map[c] for c in present_classes]

            report = classification_report(
                y, y_pred,
                labels       = present_classes,
                target_names = target_names,
                output_dict  = True,
                zero_division = 0,
            )

            # AUC (one-vs-rest, only for classes present in y)
            auc = None
            try:
                y_bin = label_binarize(y, classes=present_classes)
                proba_subset = y_proba[:, present_classes]
                if y_bin.shape[1] > 1:
                    auc = roc_auc_score(y_bin, proba_subset,
                                       multi_class="ovr", average="macro")
            except Exception as e:
                logger.warning(f"AUC skipped for {split_name}: {e}")

            # Confusion matrix
            cm = confusion_matrix(y, y_pred, labels=present_classes)

            results[split_name] = {
                "report":    report,
                "auc":       auc,
                "cm":        cm.tolist(),
                "accuracy":  report["accuracy"],
                "macro_f1":  report["macro avg"]["f1-score"],
            }

            logger.info(f"\n── {split_name.upper()} Results ──────────────────────────────")
            logger.info(f"  Accuracy:  {report['accuracy']:.4f}")
            logger.info(f"  Macro F1:  {report['macro avg']['f1-score']:.4f}")
            if auc:
                logger.info(f"  Macro AUC: {auc:.4f}")
            print(classification_report(
                y, y_pred,
                labels=present_classes, target_names=target_names,
                zero_division=0,
            ))

            if use_mlflow:
                try:
                    import mlflow
                    mlflow.log_metric(f"{split_name}_accuracy", report["accuracy"])
                    mlflow.log_metric(f"{split_name}_macro_f1", report["macro avg"]["f1-score"])
                    if auc:
                        mlflow.log_metric(f"{split_name}_auc", auc)
                except Exception:
                    pass

        # Save results
        results_path = self.artifact_dir / "xgb_eval_results.json"
        serialisable = {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in results.items()
        }
        with open(results_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"💾 Eval results saved → {results_path}")
        return results

    # ── SHAP analysis ─────────────────────────────────────────────────────────

    def explain(self, n_samples: int = 1000) -> None:
        """
        Generate SHAP feature importance analysis.
        Saves a summary plot and importance CSV.
        Requires: pip install shap matplotlib
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap or matplotlib not installed — skipping SHAP analysis")
            return

        logger.info(f"Computing SHAP values on {n_samples} samples …")

        # Sample from test set
        idx     = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)),
                                   replace=False)
        X_sample = self.X_test[idx]

        importance_df = self._model.explain(X_sample, top_n=20)
        logger.info("Top 10 features by SHAP:")
        print(importance_df.head(10).to_string(index=False))

        # Save importance CSV
        csv_path = self.artifact_dir / "xgb_shap_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"💾 SHAP importance saved → {csv_path}")

        # Plot (works in Colab inline)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df["feature"], importance_df["importance"])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("XGBoost — Feature Importance (SHAP)")
            ax.invert_yaxis()
            plt.tight_layout()
            plot_path = self.artifact_dir / "xgb_shap_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            logger.info(f"💾 SHAP plot saved → {plot_path}")
            plt.show()
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

    # ── Training curve ────────────────────────────────────────────────────────

    def plot_training_curve(self) -> None:
        """Plot train vs val loss over boosting rounds."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        history = self._model.training_history()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for metric, ax in zip(["mlogloss", "merror"], axes):
            for split in ["train", "val"]:
                sub = history[(history["split"] == split) & (history["metric"] == metric)]
                ax.plot(sub["round"], sub["value"], label=split)
            ax.set_title(f"XGBoost {metric}")
            ax.set_xlabel("Boosting round")
            ax.legend()

        plt.tight_layout()
        curve_path = self.artifact_dir / "xgb_training_curve.png"
        plt.savefig(curve_path, dpi=150)
        logger.info(f"💾 Training curve saved → {curve_path}")
        plt.show()

    # ── Save model ────────────────────────────────────────────────────────────

    def save_model(self) -> None:
        self._model.save(self.artifact_dir / "xgb_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Colab entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("  XGBoost Training — Drug Interaction Severity")
    logger.info("=" * 60)

    trainer = XGBoostTrainer(
        data_dir     = args.data_dir,
        artifact_dir = args.artifact_dir,
    )

    # Step 1: Load
    trainer.load_data()

    # Step 2: Tune (optional)
    params = None
    if args.tune:
        params = trainer.tune(n_trials=args.n_trials)

    # Step 3: Train
    trainer.train(
        params         = params,
        early_stopping = args.early_stopping,
        use_mlflow     = args.mlflow,
    )

    # Step 4: Evaluate
    trainer.evaluate(use_mlflow=args.mlflow)

    # Step 5: SHAP
    if args.shap:
        trainer.explain(n_samples=args.shap_samples)

    # Step 6: Training curve
    if args.plot:
        trainer.plot_training_curve()

    # Step 7: Save
    trainer.save_model()

    logger.info("=" * 60)
    logger.info("  XGBoost training complete!")
    logger.info(f"  Model saved → {args.artifact_dir}/xgb_model.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train XGBoost drug interaction classifier")
    ap.add_argument("--data-dir",      default="data/processed/")
    ap.add_argument("--artifact-dir",  default="ml/artifacts/")
    ap.add_argument("--tune",          action="store_true",
                    help="Run Optuna hyperparameter search first")
    ap.add_argument("--n-trials",      type=int, default=50,
                    help="Optuna trials (default 50)")
    ap.add_argument("--early-stopping",type=int, default=50)
    ap.add_argument("--shap",          action="store_true",
                    help="Generate SHAP feature importance")
    ap.add_argument("--shap-samples",  type=int, default=1000)
    ap.add_argument("--plot",          action="store_true",
                    help="Plot training curves")
    ap.add_argument("--mlflow",        action="store_true",
                    help="Log to MLflow")
    args = ap.parse_args()
    main(args)