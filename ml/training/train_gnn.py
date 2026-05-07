"""
ml/training/train_gnn.py
────────────────────────────────────────────────────────────────────────────────
Full GNN training pipeline for drug-drug interaction prediction.

Run on: Google Colab (notebook 04) — GPU required for reasonable speed
  ~20 minutes on T4 GPU | ~3 hours on CPU

Stages:
  1. Build drug interaction graph from feature matrix
  2. Create train/val/test edge splits
  3. Train GAT model with combined interaction + severity loss
  4. Evaluate on held-out edges
  5. Save model + graph artifacts

Colab setup — upload to Google Drive drug_watchdog/ folder:
  data/processed/feature_matrix.parquet
  data/processed/drug_pairs.parquet
  data/processed/feature_names.json
  data/processed/class_weights.json
  ml/models/gnn_model.py
  ml/training/train_gnn.py

Colab install cell:
  !pip install torch torchvision torch-scatter torch-sparse \
    torch-geometric -q
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Training hyperparameters ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "epochs":           100,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "batch_size":       2048,    # pairs per batch
    "patience":         15,      # early stopping
    "interaction_weight": 1.0,   # weight for binary classification loss
    "severity_weight":  0.5,     # weight for severity regression loss
    "neg_pos_ratio":    5,       # negative (no-interaction) samples per positive
}


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

class GNNDataPrep:
    """
    Prepares graph data and edge-level train/val/test splits for GNN training.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def prepare(
        self,
        feature_matrix_path: str | Path,
        drug_pairs_path:     str | Path,
        feature_names_path:  str | Path,
        val_ratio:  float = 0.15,
        test_ratio: float = 0.15,
        seed:       int   = 42,
    ) -> dict:
        """
        Build graph + edge splits. Returns a data dict with all tensors.
        """
        from ml.models.gnn_model import DrugGraph

        logger.info("Preparing GNN data …")

        # Build graph
        graph_path = self.base_dir / "ml/artifacts/drug_graph.pkl"

        graph = DrugGraph()

        if graph_path.exists():
            logger.info("Loading cached drug graph …")
            graph = DrugGraph.load(graph_path)   # ← assign the returned instance
        else:
            logger.info("Building drug interaction graph …")
            graph = DrugGraph()
            graph.build(
                feature_matrix_path = feature_matrix_path,
                drug_pairs_path     = drug_pairs_path,
                feature_names_path  = feature_names_path,
            )
            graph.save(graph_path)

        x, edge_index, node_map = graph.get_tensors()

        # Save graph for inference
        graph.save(self.base_dir / "ml/artifacts/drug_graph.pkl")

        # Load pairs for edge-level supervision
        pairs_df      = pd.read_parquet(drug_pairs_path)
        feature_names = json.load(open(feature_names_path))
        fm_df         = pd.read_parquet(feature_matrix_path)

        # Build supervised edge dataset
        edges, labels, severities, pair_features = self._build_edge_dataset(
            pairs_df, fm_df, node_map, feature_names
        )

        # Split edges into train/val/test
        n = len(labels)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        n_val  = int(n * val_ratio)
        n_test = int(n * test_ratio)

        idx_test  = idx[:n_test]
        idx_val   = idx[n_test:n_test + n_val]
        idx_train = idx[n_test + n_val:]

        logger.info(
            f"Edge splits — train: {len(idx_train):,} | "
            f"val: {len(idx_val):,} | test: {len(idx_test):,}"
        )

        return {
            "x":             x,
            "edge_index":    edge_index,
            "node_map":      node_map,
            "graph":         graph,
            # Edge supervision tensors
            "edges":         edges,           # (n_edges, 2) — node idx pairs
            "labels":        labels,          # (n_edges,) — 0/1 interaction
            "severities":    severities,      # (n_edges,) — 0-3 float
            "pair_features": pair_features,   # (n_edges, n_features)
            # Splits
            "idx_train": idx_train,
            "idx_val":   idx_val,
            "idx_test":  idx_test,
            # Metadata
            "n_nodes":    x.shape[0],
            "n_features": x.shape[1],
        }

    def _build_edge_dataset(
        self,
        pairs_df:      pd.DataFrame,
        fm_df:         pd.DataFrame,
        node_map:      dict[str, int],
        feature_names: list[str],
    ) -> tuple:
        """
        Convert drug pairs DataFrame into tensor arrays for supervised training.
        Adds negative samples (drug pairs with no known interaction).
        """
        feat_cols = [c for c in feature_names if c in fm_df.columns]

        edges_list     = []
        labels_list    = []
        severities_list = []
        feats_list     = []

        # ── Positive edges (known interactions) ───────────────────────────────
        for _, row in fm_df.iterrows():
            name_a = row.get("drug_a_name", "")
            name_b = row.get("drug_b_name", "")
            if name_a not in node_map or name_b not in node_map:
                continue
            sev = float(row.get("severity", 0))
            edges_list.append([node_map[name_a], node_map[name_b]])
            labels_list.append(1 if sev > 0 else 0)
            severities_list.append(sev)
            feats_list.append(row[feat_cols].values.astype(np.float32))

        n_pos = len(edges_list)
        logger.info(f"  Positive edges: {n_pos:,}")

        # ── Negative edges (random pairs with no known interaction) ───────────
        # We sample neg_pos_ratio * n_pos random pairs
        node_names  = list(node_map.keys())
        known_pairs = set(
            (min(e[0], e[1]), max(e[0], e[1])) for e in edges_list
        )
        n_neg_target = min(n_pos * DEFAULT_CONFIG["neg_pos_ratio"], 200_000)
        rng = np.random.default_rng(42)
        neg_added = 0

        # Build zero feature vector for unknown pairs
        zero_feats = np.zeros(len(feat_cols), dtype=np.float32)

        attempts = 0
        while neg_added < n_neg_target and attempts < n_neg_target * 10:
            attempts += 1
            i = int(rng.integers(0, len(node_names)))
            j = int(rng.integers(0, len(node_names)))
            if i == j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in known_pairs:
                continue
            edges_list.append([i, j])
            labels_list.append(0)
            severities_list.append(0.0)
            feats_list.append(zero_feats)
            known_pairs.add(pair_key)
            neg_added += 1

        logger.info(f"  Negative edges added: {neg_added:,}")

        edges        = torch.tensor(edges_list,      dtype=torch.long)
        labels       = torch.tensor(labels_list,     dtype=torch.long)
        severities   = torch.tensor(severities_list, dtype=torch.float32)
        pair_features = torch.tensor(
            np.array(feats_list), dtype=torch.float32
        )

        return edges, labels, severities, pair_features


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GNNTrainer:

    def __init__(
        self,
        base_dir:     str | Path = ".",
        artifact_dir: str | Path = "ml/artifacts/",
        config:       dict | None = None,
    ) -> None:
        self.base_dir     = Path(base_dir)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self._data:  dict | None = None
        self._model: nn.Module | None = None

    # ── Data ──────────────────────────────────────────────────────────────────

    def prepare_data(
        self,
        feature_matrix_path: str | Path,
        drug_pairs_path:     str | Path,
        feature_names_path:  str | Path,
    ) -> "GNNTrainer":
        prep = GNNDataPrep(self.base_dir)
        self._data = prep.prepare(
            feature_matrix_path = feature_matrix_path,
            drug_pairs_path     = drug_pairs_path,
            feature_names_path  = feature_names_path,
        )
        return self

    # ── Model ─────────────────────────────────────────────────────────────────

    def build_model(self) -> "GNNTrainer":
        from ml.models.gnn_model import DrugInteractionGNN
        assert self._data is not None, "Call prepare_data() first"

        self._model = DrugInteractionGNN(
            input_dim  = self._data["n_features"],
            hidden_dim = 128,
            embed_dim  = 32,
            mlp_hidden = 64,
            heads_1    = 4,
            heads_2    = 4,
            dropout    = 0.3,
            use_pyg    = True,
        ).to(self.device)

        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")
        return self

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self) -> "GNNTrainer":
        assert self._model is not None, "Call build_model() first"
        assert self._data  is not None, "Call prepare_data() first"

        # Move graph to device
        x           = self._data["x"].to(self.device)
        edge_index  = self._data["edge_index"].to(self.device)
        edges       = self._data["edges"]
        labels      = self._data["labels"]
        severities  = self._data["severities"]
        pair_feats  = self._data["pair_features"]
        idx_train   = self._data["idx_train"]
        idx_val     = self._data["idx_val"]

        # Class weights for interaction loss
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)

        optimizer = Adam(
            self._model.parameters(),
            lr           = self.cfg["lr"],
            weight_decay = self.cfg["weight_decay"],
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss  = float("inf")
        patience_count = 0
        history        = []

        logger.info(f"Starting GNN training for {self.cfg['epochs']} epochs …")
        t0 = time.time()

        for epoch in range(1, self.cfg["epochs"] + 1):
            # ── Train epoch ───────────────────────────────────────────────────
            self._model.train()
            train_loss = self._run_epoch(
                x, edge_index, edges, labels, severities, pair_feats,
                idx_train, optimizer, training=True, pos_weight=pos_weight
            )

            # ── Validation epoch ──────────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(
                    x, edge_index, edges, labels, severities, pair_feats,
                    idx_val, optimizer=None, training=False, pos_weight=pos_weight
                )

            scheduler.step(val_loss)
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss
            })

            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:>3}/{self.cfg['epochs']} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"time: {elapsed:.0f}s"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss  = val_loss
                patience_count = 0
                # Save best checkpoint
                torch.save(
                    self._model.state_dict(),
                    self.artifact_dir / "gnn_best_checkpoint.pt"
                )
            else:
                patience_count += 1
                if patience_count >= self.cfg["patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best checkpoint
        self._model.load_state_dict(
            torch.load(self.artifact_dir / "gnn_best_checkpoint.pt",
                       map_location=self.device)
        )

        self._history = history
        total_time = time.time() - t0
        logger.info(
            f"✅ GNN training complete — "
            f"best val loss: {best_val_loss:.4f} | "
            f"total time: {total_time:.0f}s"
        )
        return self

    def _run_epoch(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edges:       torch.Tensor,
        labels:      torch.Tensor,
        severities:  torch.Tensor,
        pair_feats:  torch.Tensor,
        idx:         np.ndarray,
        optimizer:   Optional[torch.optim.Optimizer],
        training:    bool,
        pos_weight:  torch.Tensor,
    ) -> float:
        """Run one epoch (train or eval) over a set of edge indices."""
        batch_size  = self.cfg["batch_size"]
        total_loss  = 0.0
        n_batches   = 0

        # Shuffle training indices
        if training:
            perm = np.random.permutation(idx)
        else:
            perm = idx

        for start in range(0, len(perm), batch_size):
            batch_idx   = perm[start: start + batch_size]
            batch_edges = edges[batch_idx].to(self.device)
            batch_labels = labels[batch_idx].to(self.device)
            batch_sev   = severities[batch_idx].to(self.device)
            batch_feats = pair_feats[batch_idx].to(self.device)

            if training:
                optimizer.zero_grad()

            logits, sev_pred = self._model(
                x, edge_index, batch_edges, batch_feats
            )

            # Loss 1: binary interaction classification
            interaction_loss = F.cross_entropy(
                logits, batch_labels,
                weight=torch.tensor(
                    [1.0, float(pos_weight.item())], device=self.device
                )
            )

            # Loss 2: severity regression (only on positive pairs)
            pos_mask = batch_labels == 1
            if pos_mask.sum() > 0:
                sev_loss = F.mse_loss(
                    sev_pred[pos_mask].squeeze(),
                    batch_sev[pos_mask]
                )
            else:
                sev_loss = torch.tensor(0.0, device=self.device)

            loss = (
                self.cfg["interaction_weight"] * interaction_loss
                + self.cfg["severity_weight"]  * sev_loss
            )

            if training:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Evaluate on test set. Returns metrics dict."""
        from sklearn.metrics import (
            classification_report, roc_auc_score, average_precision_score
        )

        assert self._data  is not None
        assert self._model is not None

        x          = self._data["x"].to(self.device)
        edge_index = self._data["edge_index"].to(self.device)

        # 🔥 Move once to GPU (avoid repeated transfers)
        edges      = self._data["edges"].to(self.device)
        pair_feats = self._data["pair_features"].to(self.device)

        # Keep labels/severity on CPU for metrics
        labels     = self._data["labels"].cpu().numpy()
        severities = self._data["severities"].cpu().numpy()
        idx_test   = self._data["idx_test"]

        self._model.eval()
        all_proba  = []
        all_sev    = []

        with torch.no_grad():
            for start in range(0, len(idx_test), self.cfg["batch_size"]):
                batch_idx   = idx_test[start: start + self.cfg["batch_size"]]

                # Already on GPU → just index
                batch_edges = edges[batch_idx]
                batch_feats = pair_feats[batch_idx]

                logits, sev_pred = self._model(
                    x, edge_index, batch_edges, batch_feats
                )

                # Move outputs to CPU for numpy conversion
                proba = F.softmax(logits, dim=-1).cpu().numpy()
                all_proba.append(proba)

                all_sev.append(sev_pred.squeeze().cpu().numpy())

        all_proba  = np.vstack(all_proba)
        all_sev    = np.concatenate(all_sev)

        y_true   = labels[idx_test]
        y_pred   = np.argmax(all_proba, axis=1)
        sev_true = severities[idx_test]

        # Metrics
        report = classification_report(
            y_true, y_pred,
            target_names = ["no_interaction", "interaction"],
            output_dict  = True,
            zero_division = 0,
        )

        auc = ap = None
        try:
            auc = roc_auc_score(y_true, all_proba[:, 1])
            ap  = average_precision_score(y_true, all_proba[:, 1])
        except Exception as e:
            logger.warning(f"AUC/AP skipped: {e}")

        # Severity MAE (on positive pairs only)
        pos_mask  = y_true == 1
        sev_mae   = float(np.mean(np.abs(all_sev[pos_mask] - sev_true[pos_mask]))) \
                    if pos_mask.sum() > 0 else None

        results = {
            "accuracy":     report["accuracy"],
            "macro_f1":     report["macro avg"]["f1-score"],
            "interaction_f1": report.get("interaction", {}).get("f1-score"),
            "auc":          auc,
            "ap":           ap,
            "severity_mae": sev_mae,
        }

        logger.info("\n── GNN TEST Results ────────────────────────────────────")
        logger.info(f"  Accuracy:         {results['accuracy']:.4f}")
        logger.info(f"  Macro F1:         {results['macro_f1']:.4f}")
        logger.info(f"  AUC:              {auc:.4f}" if auc else "  AUC: N/A")
        logger.info(f"  Avg Precision:    {ap:.4f}"  if ap  else "  AP: N/A")
        logger.info(f"  Severity MAE:     {sev_mae:.3f}" if sev_mae else "  Severity MAE: N/A")
        print(classification_report(
            y_true, y_pred,
            target_names=["no_interaction", "interaction"],
            zero_division=0,
        ))

        # Save results
        results_path = self.artifact_dir / "gnn_eval_results.json"
        with open(results_path, "w") as f:
            json.dump({k: float(v) if v is not None else None
                       for k, v in results.items()}, f, indent=2)
        logger.info(f"💾 GNN eval results saved → {results_path}")
        return results

    # ── Plot training curve ───────────────────────────────────────────────────

    def plot_training_curve(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs     = [h["epoch"]      for h in self._history]
        train_loss = [h["train_loss"] for h in self._history]
        val_loss   = [h["val_loss"]   for h in self._history]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label="train")
        plt.plot(epochs, val_loss,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GNN Training Curve")
        plt.legend()
        plt.tight_layout()
        curve_path = self.artifact_dir / "gnn_training_curve.png"
        plt.savefig(curve_path, dpi=150)
        logger.info(f"💾 GNN training curve saved → {curve_path}")
        plt.show()

    # ── Save model ────────────────────────────────────────────────────────────

    def save_model(self) -> None:
        assert self._model is not None
        # Move to CPU before saving for portability
        self._model.cpu()
        self._model.save(self.artifact_dir / "gnn_model.pt")
        self._model.to(self.device)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Colab entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("  GNN Training — Drug Interaction Prediction")
    logger.info("=" * 60)

    base_dir = Path(args.base_dir)

    trainer = GNNTrainer(
        base_dir     = base_dir,
        artifact_dir = args.artifact_dir,
        config       = {
            "epochs":             args.epochs,
            "lr":                 args.lr,
            "batch_size":         args.batch_size,
            "patience":           args.patience,
            "interaction_weight": args.interaction_weight,
            "severity_weight":    args.severity_weight,
        },
    )

    # Step 1: Prepare data
    trainer.prepare_data(
        feature_matrix_path = base_dir / "data/processed/feature_matrix.parquet",
        drug_pairs_path     = base_dir / "data/processed/drug_pairs.parquet",
        feature_names_path  = base_dir / "data/processed/feature_names.json",
    )

    # Step 2: Build model
    trainer.build_model()

    # Step 3: Train
    trainer.train()

    # Step 4: Evaluate
    trainer.evaluate()

    # Step 5: Plot
    if args.plot:
        trainer.plot_training_curve()

    # Step 6: Save
    trainer.save_model()

    logger.info("=" * 60)
    logger.info("  GNN training complete!")
    logger.info(f"  Model → {args.artifact_dir}/gnn_model.pt")
    logger.info(f"  Graph → {args.artifact_dir}/drug_graph.pkl")
    logger.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train GNN drug interaction model")
    ap.add_argument("--base-dir",          default=".")
    ap.add_argument("--artifact-dir",      default="ml/artifacts/")
    ap.add_argument("--epochs",            type=int,   default=100)
    ap.add_argument("--lr",                type=float, default=1e-3)
    ap.add_argument("--batch-size",        type=int,   default=2048)
    ap.add_argument("--patience",          type=int,   default=15)
    ap.add_argument("--interaction-weight",type=float, default=1.0)
    ap.add_argument("--severity-weight",   type=float, default=0.5)
    ap.add_argument("--plot",              action="store_true")
    args = ap.parse_args()
    main(args)