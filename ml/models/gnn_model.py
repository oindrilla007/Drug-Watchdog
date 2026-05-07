"""
ml/models/gnn_model.py
────────────────────────────────────────────────────────────────────────────────
Graph Neural Network for drug-drug interaction severity prediction.

Architecture:
  - Nodes  = drugs (one per unique drug in the dataset)
  - Edges  = known drug-drug interactions (from DrugBank)
  - Node features = 42-dim molecular + CYP450 feature vector (from feature_pipeline)
  - Edge labels   = severity 0/1/3 (binary interaction + severity)

Model: Graph Attention Network (GAT)
  Layer 1: GATConv(42 → 128, heads=4)   → 512-dim
  Layer 2: GATConv(512 → 64, heads=4)   → 256-dim
  Layer 3: GATConv(256 → 32, heads=1)   → 32-dim
  Link prediction head: MLP(32+32 → 64 → 2)
    Output 1: interaction probability  (binary)
    Output 2: severity score           (regression 0–3)

Why GAT over GCN?
  Attention heads learn WHICH neighbouring drugs matter most for
  predicting an interaction — e.g. a drug's CYP3A4 inhibitor
  neighbours get higher attention than unrelated drugs.

Why link prediction (edge-level) not node-level?
  We want to predict properties of a PAIR of drugs, not a single drug.
  The GNN encodes each drug's neighbourhood context, then the MLP
  combines both node embeddings to predict the edge (pair) outcome.

Run on: Google Colab (notebook 04) — needs GPU for reasonable training time
  This file defines the model architecture only.
  Training is in ml/training/train_gnn.py
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Architecture constants ────────────────────────────────────────────────────
INPUT_DIM    = 42       # matches feature_pipeline output
HIDDEN_DIM   = 128
GAT_HEADS_1  = 4
GAT_HEADS_2  = 4
GAT_HEADS_3  = 1
EMBED_DIM    = 32       # final node embedding size
MLP_HIDDEN   = 64
DROPOUT      = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# GAT Convolution (pure PyTorch — no PyG dependency for portability)
# ─────────────────────────────────────────────────────────────────────────────

class GATConv(nn.Module):
    """
    Multi-head Graph Attention Convolution layer.
    Pure PyTorch implementation — works without torch_geometric installed.
    Falls back to PyG's GATConv if available for better performance.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        heads:        int = 1,
        dropout:      float = 0.0,
        concat:       bool = True,
    ) -> None:
        super().__init__()
        self.heads       = heads
        self.out_per_head = out_features
        self.concat      = concat
        self.dropout     = dropout

        # Linear projection per head
        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        # Attention coefficients per head
        self.a = nn.Parameter(torch.empty(1, heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.reshape(1, -1).unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x:         torch.Tensor,   # (N, in_features)
        edge_index: torch.Tensor,   # (2, E)  source→target
    ) -> torch.Tensor:
        N = x.size(0)
        # Project all nodes: (N, heads * out_per_head)
        h = self.W(x).view(N, self.heads, self.out_per_head)

        src, tgt = edge_index[0], edge_index[1]

        # Attention: concat source + target features per head
        # e_ij = LeakyReLU(a^T [h_i || h_j])
        h_src = h[src]   # (E, heads, out_per_head)
        h_tgt = h[tgt]   # (E, heads, out_per_head)
        e = self.leaky_relu(
            (self.a * torch.cat([h_src, h_tgt], dim=-1)).sum(dim=-1)
        )   # (E, heads)

        # Softmax over incoming edges per target node
        alpha = self._sparse_softmax(e, tgt, N)   # (E, heads)
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)

        # Aggregate: weighted sum of source embeddings
        out = torch.zeros(N, self.heads, self.out_per_head, device=x.device)
        idx = tgt.unsqueeze(-1).unsqueeze(-1).expand_as(h_src)
        out.scatter_add_(0, idx, alpha.unsqueeze(-1) * h_src)

        if self.concat:
            return out.view(N, self.heads * self.out_per_head)   # (N, heads*out)
        else:
            return out.mean(dim=1)   # (N, out_per_head)

    @staticmethod
    def _sparse_softmax(
        e:   torch.Tensor,   # (E, heads)
        idx: torch.Tensor,   # (E,) target node indices
        N:   int,
    ) -> torch.Tensor:
        """Compute softmax over edges grouped by target node."""
        # Shift for numerical stability
        e_max = torch.zeros(N, e.size(1), device=e.device)
        e_max.scatter_reduce_(0, idx.unsqueeze(1).expand_as(e), e,
                               reduce="amax", include_self=True)
        e_exp  = torch.exp(e - e_max[idx])
        e_sum  = torch.zeros(N, e.size(1), device=e.device)
        e_sum.scatter_add_(0, idx.unsqueeze(1).expand_as(e_exp), e_exp)
        return e_exp / (e_sum[idx] + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Full GNN model
# ─────────────────────────────────────────────────────────────────────────────

class DrugInteractionGNN(nn.Module):
    """
    3-layer GAT encoder + link prediction MLP.

    Forward pass:
      1. Encode all drug nodes → embeddings
      2. For each (drug_a, drug_b) pair: concatenate embeddings
      3. MLP → [interaction_prob, severity_score]
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        hidden_dim:  int   = HIDDEN_DIM,
        embed_dim:   int   = EMBED_DIM,
        mlp_hidden:  int   = MLP_HIDDEN,
        heads_1:     int   = GAT_HEADS_1,
        heads_2:     int   = GAT_HEADS_2,
        dropout:     float = DROPOUT,
        use_pyg:     bool  = True,
    ) -> None:
        super().__init__()
        self.use_pyg = use_pyg and self._check_pyg()

        # ── GAT layers ────────────────────────────────────────────────────────
        if self.use_pyg:
            from torch_geometric.nn import GATConv as PyGGATConv
            self.gat1 = PyGGATConv(input_dim,  hidden_dim, heads=heads_1, dropout=dropout, concat=True)
            self.gat2 = PyGGATConv(hidden_dim * heads_1, hidden_dim // 2, heads=heads_2, dropout=dropout, concat=True)
            self.gat3 = PyGGATConv(hidden_dim // 2 * heads_2, embed_dim, heads=1, dropout=dropout, concat=False)
        else:
            logger.info("torch_geometric not found — using pure PyTorch GATConv")
            self.gat1 = GATConv(input_dim,  hidden_dim,          heads=heads_1, dropout=dropout, concat=True)
            self.gat2 = GATConv(hidden_dim * heads_1, hidden_dim // 2, heads=heads_2, dropout=dropout, concat=True)
            self.gat3 = GATConv(hidden_dim // 2 * heads_2, embed_dim, heads=1, dropout=dropout, concat=False)

        # Batch normalisation after each GAT layer
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads_1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2 * heads_2)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        # ── Link prediction MLP ───────────────────────────────────────────────
        # Input: concatenated node embeddings of drug_a and drug_b (embed_dim * 2)
        # + 42 raw features for the pair
        mlp_input = embed_dim * 2 + input_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, mlp_hidden * 2),
            nn.BatchNorm1d(mlp_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Two output heads
        self.interaction_head = nn.Linear(mlp_hidden, 2)    # binary: interaction or not
        self.severity_head    = nn.Linear(mlp_hidden, 1)    # regression: severity 0–3

        self._init_weights()

    def forward(
        self,
        x:          torch.Tensor,    # (N, input_dim) — all drug node features
        edge_index: torch.Tensor,    # (2, E) — graph edges for message passing
        pair_idx:   torch.Tensor,    # (B, 2) — drug pairs to predict
        pair_feats: torch.Tensor,    # (B, input_dim) — raw pair features
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          interaction_logits: (B, 2)  — for CrossEntropyLoss
          severity_scores:    (B, 1)  — for MSELoss (clipped 0–3 at inference)
        """
        # ── Encode all drug nodes ─────────────────────────────────────────────
        h = self._encode(x, edge_index)   # (N, embed_dim)

        # ── Extract embeddings for each pair ──────────────────────────────────
        idx_a = pair_idx[:, 0]
        idx_b = pair_idx[:, 1]
        h_a   = h[idx_a]   # (B, embed_dim)
        h_b   = h[idx_b]   # (B, embed_dim)

        # ── Concatenate: graph embeddings + raw features ──────────────────────
        pair_repr = torch.cat([h_a, h_b, pair_feats], dim=-1)   # (B, embed*2 + input_dim)

        # ── MLP ───────────────────────────────────────────────────────────────
        z = self.mlp(pair_repr)

        interaction_logits = self.interaction_head(z)    # (B, 2)
        severity_scores    = self.severity_head(z)       # (B, 1)

        return interaction_logits, severity_scores

    def _encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run 3-layer GAT to get node embeddings."""
        h = F.dropout(x, p=0.1, training=self.training)

        h = self.gat1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)

        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)

        h = self.gat3(h, edge_index)
        h = self.bn3(h)
        h = F.elu(h)

        return h   # (N, embed_dim)

    def get_node_embeddings(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return drug node embeddings without running the prediction head.
        Useful for drug similarity analysis and visualisation (t-SNE/UMAP).
        """
        self.eval()
        with torch.no_grad():
            return self._encode(x, edge_index)

    def predict_pair(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        idx_a:      int,
        idx_b:      int,
        pair_feats: torch.Tensor,
    ) -> dict:
        """
        Single-pair inference interface matching predictor.py expectations.
        Returns {severity, confidence, probabilities, model}.
        """
        self.eval()
        with torch.no_grad():
            pair_idx = torch.tensor([[idx_a, idx_b]], dtype=torch.long,
                                    device=x.device)
            logits, sev = self.forward(x, edge_index, pair_idx, pair_feats)
            proba       = F.softmax(logits, dim=-1)[0].cpu().numpy()
            sev_val     = float(sev[0, 0].cpu().clamp(0, 3))
            severity    = round(sev_val)

        return {
            "severity":      severity,
            "severity_raw":  sev_val,
            "confidence":    float(proba[1]),   # P(interaction)
            "probabilities": proba.tolist(),
            "model":         "gnn",
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = "ml/artifacts/gnn_model.pt") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "config": {
                "input_dim":  INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "embed_dim":  EMBED_DIM,
                "mlp_hidden": MLP_HIDDEN,
                "heads_1":    GAT_HEADS_1,
                "heads_2":    GAT_HEADS_2,
                "dropout":    DROPOUT,
                "use_pyg":    self.use_pyg,
            }
        }, path)
        logger.info(f"💾 GNN model saved → {path}")

    @classmethod
    def load(cls, path: str | Path = "ml/artifacts/gnn_model.pt") -> "DrugInteractionGNN":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)

        cfg = checkpoint["config"]

        model = cls(**cfg)

        state = cls._adapt_gat_state_dict(checkpoint["model_state"], model)
        model.load_state_dict(state)

        model.to(device)
        model.eval()

        logger.info(f"✅ GNN model loaded from {path} on {device}")

        return model

    @staticmethod
    def _adapt_gat_state_dict(
        state_dict: dict[str, torch.Tensor],
        model: "DrugInteractionGNN",
    ) -> dict[str, torch.Tensor]:
        """
        Make GAT checkpoints portable between PyG and the local fallback layer.

        PyG checkpoints use gat*.lin.weight, gat*.att_src, gat*.att_dst, and
        gat*.bias. The fallback layer uses gat*.W.weight and gat*.a.
        """
        adapted = dict(state_dict)
        target_keys = model.state_dict().keys()

        for prefix in ("gat1", "gat2", "gat3"):
            pyg_lin = f"{prefix}.lin.weight"
            local_w = f"{prefix}.W.weight"
            pyg_src = f"{prefix}.att_src"
            pyg_dst = f"{prefix}.att_dst"
            local_a = f"{prefix}.a"

            if local_w in target_keys and local_w not in adapted and pyg_lin in adapted:
                adapted[local_w] = adapted[pyg_lin]
            if local_a in target_keys and local_a not in adapted and pyg_src in adapted and pyg_dst in adapted:
                adapted[local_a] = torch.cat([adapted[pyg_src], adapted[pyg_dst]], dim=-1)

            if pyg_lin in target_keys and pyg_lin not in adapted and local_w in adapted:
                adapted[pyg_lin] = adapted[local_w]
            if pyg_src in target_keys and pyg_dst in target_keys and pyg_src not in adapted and local_a in adapted:
                out_per_head = adapted[local_a].shape[-1] // 2
                adapted[pyg_src] = adapted[local_a][..., :out_per_head]
                adapted[pyg_dst] = adapted[local_a][..., out_per_head:]

        return {k: v for k, v in adapted.items() if k in target_keys}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _check_pyg() -> bool:
        try:
            import torch_geometric
            return True
        except ImportError:
            return False

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder — converts drug pair data into PyTorch graph tensors
# ─────────────────────────────────────────────────────────────────────────────

class DrugGraph:
    """
    Builds the drug-drug interaction graph from processed data.
    Converts drug names → integer node IDs and interaction pairs → edge_index.

    Usage:
      graph = DrugGraph()

        graph_path = self.base_dir / "ml/artifacts/drug_graph.pkl"

        if graph_path.exists():
            logger.info("Loading cached drug graph …")
            graph = DrugGraph.load(graph_path)

        else:
            logger.info("Building drug interaction graph …")
            graph = DrugGraph()
            graph.build(
                feature_matrix_path = feature_matrix_path,
                drug_pairs_path     = drug_pairs_path,
                feature_names_path  = feature_names_path,
            )
            graph.save(graph_path)

        # 🔥 IMPORTANT: DO NOT REASSIGN graph ANYWHERE BELOW THIS
        logger.info(f"GRAPH BUILT FLAG: {getattr(graph, '_built', None)}")
        logger.info(f"GRAPH TYPE: {type(graph)}")

        x, edge_index, node_map = graph.get_tensors()
    """

    def __init__(self) -> None:
        self._node_map:   dict[str, int] = {}   # drug_name → node_id
        self._x:          Optional[torch.Tensor] = None
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_labels: Optional[torch.Tensor] = None
        self._built = False

    def build(
        self,
        feature_matrix_path: str | Path,
        drug_pairs_path:     str | Path,
        feature_names_path:  str | Path = "data/processed/feature_names.json",
    ) -> "DrugGraph":
        import json
        import pandas as pd

        fm_path = Path(feature_matrix_path)
        dp_path = Path(drug_pairs_path)

        logger.info("Building drug interaction graph …")

        # Load feature matrix (has per-drug-pair features)
        fm_df = pd.read_parquet(fm_path)

        # Load feature names
        with open(feature_names_path) as f:
            feature_names = json.load(f)

        # ── Build node map ────────────────────────────────────────────────────
        all_drugs = pd.concat([
            fm_df["drug_a_name"].rename("name"),
            fm_df["drug_b_name"].rename("name"),
        ]).drop_duplicates().dropna()

        self._node_map = {name: idx for idx, name in enumerate(all_drugs)}
        N = len(self._node_map)
        logger.info(f"  Drug nodes: {N:,}")

        # ── Build node feature matrix X ───────────────────────────────────────
        # Each drug gets features = MEAN of all pair features where it appears
        feat_cols = [c for c in feature_names if c in fm_df.columns]
        node_feats = np.zeros((N, len(feat_cols)), dtype=np.float32)
        node_counts = np.zeros(N, dtype=np.int32)

        for _, row in fm_df.iterrows():
            name_a = row["drug_a_name"]
            name_b = row["drug_b_name"]
            feats  = row[feat_cols].values.astype(np.float32)

            if name_a in self._node_map:
                idx = self._node_map[name_a]
                node_feats[idx]  += feats
                node_counts[idx] += 1
            if name_b in self._node_map:
                idx = self._node_map[name_b]
                node_feats[idx]  += feats
                node_counts[idx] += 1

        # Average
        counts = np.maximum(node_counts, 1).reshape(-1, 1)
        node_feats = node_feats / counts

        self._x = torch.tensor(node_feats, dtype=torch.float32)

        # ── Build edge_index from ALL interaction pairs ────────────────────────
        dp_df = pd.read_parquet(dp_path)
        edges_src, edges_tgt, edge_labels = [], [], []

        for _, row in dp_df.iterrows():
            name_a = row.get("drug_a_name", "")
            name_b = row.get("drug_b_name", "")
            sev    = int(row.get("severity", 0))

            if name_a in self._node_map and name_b in self._node_map:
                idx_a = self._node_map[name_a]
                idx_b = self._node_map[name_b]
                # Add both directions (undirected graph)
                edges_src += [idx_a, idx_b]
                edges_tgt += [idx_b, idx_a]
                edge_labels += [sev, sev]

        self._edge_index = torch.tensor(
            [edges_src, edges_tgt], dtype=torch.long
        )
        self._edge_labels = torch.tensor(edge_labels, dtype=torch.float32)

        self._built = True
        logger.info(
            f"✅ Graph built — "
            f"nodes: {N:,} | "
            f"edges: {self._edge_index.shape[1]:,} | "
            f"node features: {self._x.shape[1]}"
        )
        return self

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Return (x, edge_index, node_map)."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        return self._x, self._edge_index, self._node_map

    def get_pair_tensors(
        self,
        drug_a: str,
        drug_b: str,
        pair_feats: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get pair_idx and pair_feats tensors for a single drug pair.
        Used at inference time in predictor.py.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")
        idx_a = self._node_map.get(drug_a, 0)
        idx_b = self._node_map.get(drug_b, 0)
        pair_idx  = torch.tensor([[idx_a, idx_b]], dtype=torch.long)
        pair_feat_t = torch.tensor(pair_feats.reshape(1, -1), dtype=torch.float32)
        return pair_idx, pair_feat_t

    def save(self, path: str | Path = "ml/artifacts/drug_graph.pkl") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "node_map":   self._node_map,
                "x":          self._x,
                "edge_index": self._edge_index,
            }, f)
        logger.info(f"💾 Drug graph saved → {path}")

    @classmethod
    def load(cls, path: str | Path = "ml/artifacts/drug_graph.pkl") -> "DrugGraph":
        with open(path, "rb") as f:
            data = pickle.load(f)
        g = cls()
        g._node_map   = data["node_map"]
        g._x          = data["x"]
        g._edge_index = data["edge_index"]
        g._built      = True
        logger.info(f"✅ Drug graph loaded — {len(g._node_map):,} nodes")
        return g


# ─────────────────────────────────────────────────────────────────────────────
# CLI — architecture sanity check (VS Code)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("GNN architecture sanity check …")

    # Create a tiny synthetic graph
    N = 100    # drugs
    E = 500    # interactions
    B = 16     # batch pairs

    x          = torch.randn(N, INPUT_DIM)
    edge_index = torch.randint(0, N, (2, E))
    pair_idx   = torch.randint(0, N, (B, 2))
    pair_feats = torch.randn(B, INPUT_DIM)

    model = DrugInteractionGNN(use_pyg=False)   # pure PyTorch mode
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Forward pass
    logits, severity = model(x, edge_index, pair_idx, pair_feats)
    logger.info(f"interaction_logits: {logits.shape}")   # (B, 2)
    logger.info(f"severity_scores:    {severity.shape}") # (B, 1)

    # Single pair prediction
    result = model.predict_pair(x, edge_index, idx_a=0, idx_b=1,
                                pair_feats=pair_feats[:1])
    logger.info(f"Single pair prediction: {result}")

    # Save / load test
    model.save("ml/artifacts/gnn_model.pt")
    model2 = DrugInteractionGNN.load("ml/artifacts/gnn_model.pt")
    logger.info("Save/load cycle: OK")

    print("\n── GNN Architecture ────────────────────────────────────────")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print("\n── Graph builder test ──────────────────────────────────────")
    print("DrugGraph ready — call build() with feature_matrix.parquet")
    print("\n✅ GNN sanity check passed. Ready for Colab training (notebook 04).")
