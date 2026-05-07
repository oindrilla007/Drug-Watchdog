"""
ml/training/evaluate.py
────────────────────────────────────────────────────────────────────────────────
Unified evaluation module for Phase 2.

Loads both trained models (XGBoost + GNN) and produces:
  1. Side-by-side metric comparison table
  2. Confusion matrices for both models
  3. ROC curves (one-vs-rest per severity class)
  4. SHAP feature importance (XGBoost)
  5. Drug embedding visualisation (GNN t-SNE)
  6. Final evaluation report saved as JSON

Run on: VS Code (after downloading artifacts from Colab)
  python -m ml.training.evaluate

Requires artifacts in ml/artifacts/:
  xgb_model.pkl
  gnn_model.pt
  drug_graph.pkl
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Loads both models and runs comprehensive evaluation on the test set.
    Produces comparison report + visualisations.
    """

    def __init__(
        self,
        data_dir:     str | Path = "data/processed/",
        artifact_dir: str | Path = "ml/artifacts/",
        output_dir:   str | Path = "ml/artifacts/eval/",
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._xgb_model  = None
        self._gnn_model  = None
        self._drug_graph = None

        # Test data
        self.X_test:  Optional[np.ndarray] = None
        self.y_test:  Optional[np.ndarray] = None
        self.feature_names: list[str] = []

    # ── Load everything ───────────────────────────────────────────────────────

    def load_all(self) -> "ModelEvaluator":
        self._load_data()
        self._load_xgb()
        self._load_gnn()
        return self

    def _load_data(self) -> None:
        logger.info("Loading test data …")
        self.X_test = np.load(self.data_dir / "X_test.npy")
        self.y_test = np.load(self.data_dir / "y_test.npy")

        fn_path = self.data_dir / "feature_names.json"
        if fn_path.exists():
            with open(fn_path) as f:
                self.feature_names = json.load(f)

        logger.info(f"  X_test: {self.X_test.shape} | y_test: {self.y_test.shape}")
        unique, counts = np.unique(self.y_test, return_counts=True)
        logger.info(f"  Test label dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    def _load_xgb(self) -> None:
        xgb_path = self.artifact_dir / "xgb_model.pkl"
        if not xgb_path.exists():
            logger.warning(f"XGBoost model not found: {xgb_path}")
            return
        from ml.models.xgboost_classifier import DrugInteractionXGB
        self._xgb_model = DrugInteractionXGB.load(xgb_path)
        logger.info("✅ XGBoost model loaded")

    def _load_gnn(self) -> None:
        gnn_path   = self.artifact_dir / "gnn_model.pt"
        graph_path = self.artifact_dir / "drug_graph.pkl"
        if not gnn_path.exists():
            logger.warning(f"GNN model not found: {gnn_path}")
            return
        from ml.models.gnn_model import DrugInteractionGNN, DrugGraph
        self._gnn_model  = DrugInteractionGNN.load(gnn_path)
        if graph_path.exists():
            self._drug_graph = DrugGraph.load(graph_path)
        logger.info("✅ GNN model loaded")

    # ── XGBoost evaluation ────────────────────────────────────────────────────

    def evaluate_xgb(self) -> dict:
        if self._xgb_model is None:
            logger.warning("XGBoost model not loaded — skipping")
            return {}

        logger.info("Evaluating XGBoost …")
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, average_precision_score,
        )
        from sklearn.preprocessing import label_binarize

        y_pred  = self._xgb_model.predict(self.X_test)
        y_proba = self._xgb_model.predict_proba(self.X_test)

        present = sorted(np.unique(self.y_test).tolist())
        label_map = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
        target_names = [label_map[c] for c in present]

        report = classification_report(
            self.y_test, y_pred,
            labels=present, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(self.y_test, y_pred, labels=present)

        # AUC
        auc = None
        try:
            y_bin  = label_binarize(self.y_test, classes=present)
            p_sub  = y_proba[:, present]
            if y_bin.shape[1] > 1:
                auc = roc_auc_score(y_bin, p_sub, multi_class="ovr", average="macro")
        except Exception as e:
            logger.warning(f"XGB AUC skipped: {e}")

        metrics = {
            "model":       "xgboost",
            "accuracy":    report["accuracy"],
            "macro_f1":    report["macro avg"]["f1-score"],
            "macro_auc":   auc,
            "report":      report,
            "confusion_matrix": cm.tolist(),
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }

        auc_text = f"{auc:.4f}" if auc is not None else "N/A"

        logger.info(f"  XGB — Accuracy: {metrics['accuracy']:.4f} | "
                    f"Macro F1: {metrics['macro_f1']:.4f} | "
                    f"AUC: {auc_text}")
        print("\n── XGBoost Classification Report ───────────────────────────")
        print(classification_report(
            self.y_test, y_pred,
            labels=present, target_names=target_names, zero_division=0
        ))
        return metrics

    # ── GNN evaluation ────────────────────────────────────────────────────────

    def evaluate_gnn(self) -> dict:
        """
        For the GNN we use the saved eval results from Colab training,
        and additionally run XGBoost on the same features as a proxy
        since the GNN requires the full graph at inference.
        """
        if self._gnn_model is None:
            logger.warning("GNN model not loaded — skipping")
            return {}

        # Load saved Colab results
        gnn_results_path = self.artifact_dir / "gnn_eval_results.json"
        if gnn_results_path.exists():
            with open(gnn_results_path) as f:
                saved = json.load(f)
            logger.info(f"GNN results loaded from Colab: {saved}")
            return {"model": "gnn", **saved}

        logger.warning("gnn_eval_results.json not found — GNN metrics unavailable")
        return {"model": "gnn"}

    # ── Comparison report ─────────────────────────────────────────────────────

    def comparison_report(
        self,
        xgb_metrics: dict,
        gnn_metrics: dict,
    ) -> dict:
        """Print side-by-side comparison and save as JSON."""

        report = {
            "xgboost": {
                "accuracy":  xgb_metrics.get("accuracy"),
                "macro_f1":  xgb_metrics.get("macro_f1"),
                "macro_auc": xgb_metrics.get("macro_auc"),
            },
            "gnn": {
                "accuracy":     gnn_metrics.get("accuracy"),
                "macro_f1":     gnn_metrics.get("macro_f1"),
                "auc":          gnn_metrics.get("auc"),
                "severity_mae": gnn_metrics.get("severity_mae"),
            },
            "recommendation": self._recommend(xgb_metrics, gnn_metrics),
        }

        print("\n" + "═" * 58)
        print("  MODEL COMPARISON — Drug Interaction Severity")
        print("═" * 58)
        print(f"  {'Metric':<22} {'XGBoost':>12} {'GNN':>12}")
        print("─" * 58)

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "N/A"

        rows = [
            ("Accuracy",     xgb_metrics.get("accuracy"),  gnn_metrics.get("accuracy")),
            ("Macro F1",     xgb_metrics.get("macro_f1"),  gnn_metrics.get("macro_f1")),
            ("Macro AUC",    xgb_metrics.get("macro_auc"), gnn_metrics.get("auc")),
            ("Severity MAE", None,                          gnn_metrics.get("severity_mae")),
        ]
        for name, xgb_val, gnn_val in rows:
            print(f"  {name:<22} {fmt(xgb_val):>12} {fmt(gnn_val):>12}")

        print("═" * 58)
        print(f"  Recommendation: {report['recommendation']}")
        print("═" * 58)

        # Save
        out_path = self.output_dir / "model_comparison.json"
        serialisable = {
            k: {kk: float(vv) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in report.items() if isinstance(v, dict)
        }
        serialisable["recommendation"] = report["recommendation"]
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"💾 Comparison report saved → {out_path}")
        return report

    def _recommend(self, xgb: dict, gnn: dict) -> str:
        xgb_auc = xgb.get("macro_auc") or 0
        gnn_auc = gnn.get("auc") or 0
        if xgb_auc >= gnn_auc:
            return (
                "XGBoost as primary (higher AUC, faster inference, SHAP explainability). "
                "GNN as secondary for novel/unseen drug pairs via graph neighbourhood."
            )
        return (
            "GNN as primary (higher AUC, graph-aware). "
            "XGBoost as fallback for drugs not in the interaction graph."
        )

    # ── SHAP analysis ─────────────────────────────────────────────────────────

    def shap_analysis(self, n_samples: int = 500) -> None:
        """Full SHAP analysis with summary plot and dependency plots."""
        if self._xgb_model is None:
            return

        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap/matplotlib not installed")
            return

        logger.info(f"Running SHAP analysis on {n_samples} test samples …")

        idx      = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)),
                                    replace=False)
        X_sample = self.X_test[idx]

        # SHAP values
        import xgboost as xgb
        explainer   = shap.TreeExplainer(self._xgb_model._model)
        shap_values = explainer.shap_values(X_sample)

        # Handle 3D array (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            mean_shap = np.abs(shap_values).mean(axis=2)
        else:
            mean_shap = np.abs(shap_values)

        names = self.feature_names or [f"f{i}" for i in range(mean_shap.shape[1])]

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        mean_importance = mean_shap.mean(axis=0)
        sorted_idx = np.argsort(mean_importance)[-20:]
        ax.barh(
            [names[i] for i in sorted_idx],
            mean_importance[sorted_idx],
            color="steelblue",
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance — Drug Interaction Severity (SHAP)")
        plt.tight_layout()
        plot_path = self.output_dir / "shap_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"💾 SHAP summary plot → {plot_path}")
        plt.show()
        plt.close()

        # Top 5 dependency plots
        top5_idx = np.argsort(mean_importance)[-5:][::-1]
        fig, axes = plt.subplots(1, min(5, len(top5_idx)), figsize=(20, 4))
        if len(top5_idx) == 1:
            axes = [axes]
        for ax, feat_idx in zip(axes, top5_idx):
            ax.scatter(X_sample[:, feat_idx], mean_shap[:, feat_idx],
                       alpha=0.3, s=5, c="steelblue")
            ax.set_xlabel(names[feat_idx])
            ax.set_ylabel("SHAP value")
            ax.set_title(f"{names[feat_idx]}")
        plt.suptitle("SHAP Dependency Plots — Top 5 Features")
        plt.tight_layout()
        dep_path = self.output_dir / "shap_dependency.png"
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        logger.info(f"💾 SHAP dependency plots → {dep_path}")
        plt.show()
        plt.close()

    # ── GNN embedding visualisation ───────────────────────────────────────────

    def visualise_drug_embeddings(self, n_drugs: int = 500) -> None:
        """
        t-SNE visualisation of GNN drug node embeddings.
        Drugs coloured by number of known interactions.
        """
        if self._gnn_model is None or self._drug_graph is None:
            logger.warning("GNN or drug graph not available for embedding viz")
            return

        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("matplotlib/sklearn not installed")
            return

        import torch
        logger.info("Generating GNN drug embedding visualisation …")

        x, edge_index, node_map = self._drug_graph.get_tensors()

        # Get embeddings for a sample of drugs
        n_drugs = min(n_drugs, x.shape[0])
        idx     = np.random.choice(x.shape[0], n_drugs, replace=False)
        x_sub   = x[idx]
        ei_sub  = edge_index   # use full graph for message passing

        self._gnn_model.eval()
        with torch.no_grad():
            embeddings = self._gnn_model.get_node_embeddings(
                x, edge_index
            )[idx].numpy()

        # Count interactions per drug
        node_names  = list(node_map.keys())
        edge_counts = np.zeros(x.shape[0])
        for i in range(edge_index.shape[1]):
            edge_counts[edge_index[0, i].item()] += 1
        sample_counts = np.log1p(edge_counts[idx])

        # t-SNE
        logger.info("Running t-SNE …")
        tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=sample_counts, cmap="viridis",
            s=20, alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="log(interaction count)")
        ax.set_title("GNN Drug Embeddings (t-SNE)\nColoured by interaction count")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        emb_path = self.output_dir / "gnn_embeddings_tsne.png"
        plt.savefig(emb_path, dpi=150, bbox_inches="tight")
        logger.info(f"💾 GNN embedding plot → {emb_path}")
        plt.show()
        plt.close()

    # ── Confusion matrix plot ─────────────────────────────────────────────────

    def plot_confusion_matrix(self, cm: list, title: str, labels: list) -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            return

        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        thresh = cm_arr.max() / 2
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, str(cm_arr[i, j]),
                        ha="center", va="center",
                        color="white" if cm_arr[i, j] > thresh else "black")

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(title)
        plt.tight_layout()

        safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = self.output_dir / f"cm_{safe_title}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"💾 Confusion matrix → {path}")
        plt.show()
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate XGBoost + GNN models")
    ap.add_argument("--data-dir",     default="data/processed/")
    ap.add_argument("--artifact-dir", default="ml/artifacts/")
    ap.add_argument("--output-dir",   default="ml/artifacts/eval/")
    ap.add_argument("--shap",         action="store_true",
                    help="Run full SHAP analysis")
    ap.add_argument("--embeddings",   action="store_true",
                    help="Visualise GNN drug embeddings (t-SNE)")
    ap.add_argument("--shap-samples", type=int, default=500)
    args = ap.parse_args()

    evaluator = ModelEvaluator(
        data_dir     = args.data_dir,
        artifact_dir = args.artifact_dir,
        output_dir   = args.output_dir,
    )

    evaluator.load_all()

    # Evaluate both models
    xgb_metrics = evaluator.evaluate_xgb()
    gnn_metrics = evaluator.evaluate_gnn()

    # Comparison report
    report = evaluator.comparison_report(xgb_metrics, gnn_metrics)

    # Confusion matrix for XGBoost
    if xgb_metrics.get("confusion_matrix"):
        present   = sorted(np.unique(evaluator.y_test).tolist())
        label_map = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
        evaluator.plot_confusion_matrix(
            cm     = xgb_metrics["confusion_matrix"],
            title  = "XGBoost Confusion Matrix",
            labels = [label_map[c] for c in present],
        )

    # Optional SHAP
    if args.shap:
        evaluator.shap_analysis(n_samples=args.shap_samples)

    # Optional GNN embeddings
    if args.embeddings:
        evaluator.visualise_drug_embeddings()

    print(f"\n── Evaluation outputs saved to {args.output_dir} ─────────────")
    for f in Path(args.output_dir).iterdir():
        size = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size:>8.1f} KB")
