"""
ml/models/xgboost_classifier.py
────────────────────────────────────────────────────────────────────────────────
XGBoost-based drug interaction severity classifier.

This is the BASELINE model (before GNN). It:
  - Classifies drug pairs into severity 0 / 1 / 2 / 3
  - Handles class imbalance via sample weights
  - Supports hyperparameter tuning via Optuna
  - Outputs calibrated probability scores per class
  - Generates SHAP feature importance values

Why XGBoost as baseline?
  - Handles tabular data extremely well
  - Interpretable via SHAP
  - Fast to train (seconds on CPU, no GPU needed)
  - Strong AUC on imbalanced classification tasks
  - Production-reliable fallback when GNN is unavailable

Run on: Google Colab (notebook 03) OR VS Code
  python -m ml.models.xgboost_classifier   ← quick sanity check
  The full training is in ml/training/train_xgboost.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default hyperparameters ───────────────────────────────────────────────────
# These are good starting values; Optuna will search around them.
DEFAULT_PARAMS = {
    "objective":        "multi:softprob",   # outputs probability per class
    "num_class":        4,                  # severity 0, 1, 2, 3
    "eval_metric":      ["mlogloss", "merror"],
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,               # L1
    "reg_lambda":       1.0,               # L2
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# ── Optuna search space ───────────────────────────────────────────────────────
OPTUNA_SPACE = {
    "max_depth":        (3, 10),
    "learning_rate":    (0.01, 0.3),
    "n_estimators":     (200, 1000),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 20),
    "gamma":            (0.0, 1.0),
    "reg_alpha":        (0.0, 2.0),
    "reg_lambda":       (0.5, 5.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DrugInteractionXGB:
    """
    XGBoost wrapper for drug interaction severity classification.
    Provides fit / predict / predict_proba / explain interface
    consistent with the GNN model for easy swapping in predictor.py.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params  = {**DEFAULT_PARAMS, **(params or {})}
        self._model  = None
        self._fitted = False
        self._feature_names: Optional[list[str]] = None
        self._class_weights: Optional[dict] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names:  Optional[list[str]] = None,
        class_weights:  Optional[dict]      = None,
        early_stopping: int = 50,
    ) -> "DrugInteractionXGB":
        """
        Train XGBoost with early stopping on validation loss.

        class_weights: {class_int: weight_float} from compute_class_weight
        early_stopping: rounds without improvement before stopping
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Run: pip install xgboost")

        self._feature_names = feature_names
        self._class_weights = class_weights

        # Build sample weights from class weights
        sample_weights = None
        if class_weights:
            sample_weights = np.array([
                class_weights.get(int(label), 1.0) for label in y_train
            ])

        logger.info(
            f"Training XGBoost — "
            f"train={len(y_train):,} | val={len(y_val):,} | "
            f"features={X_train.shape[1]}"
        )

        # Build DMatrix objects
        dtrain = xgb.DMatrix(
            X_train,
            label         = y_train,
            weight        = sample_weights,
            feature_names = feature_names,
        )
        dval = xgb.DMatrix(
            X_val,
            label         = y_val,
            feature_names = feature_names,
        )

        # Extract booster params (remove sklearn-style keys)
        booster_params = {
            k: v for k, v in self.params.items()
            if k not in ("n_estimators", "random_state", "n_jobs", "verbosity")
        }
        booster_params["seed"] = self.params.get("random_state", 42)
        booster_params["nthread"] = self.params.get("n_jobs", -1)

        evals_result = {}
        self._model = xgb.train(
            params            = booster_params,
            dtrain            = dtrain,
            num_boost_round   = self.params["n_estimators"],
            evals             = [(dtrain, "train"), (dval, "val")],
            early_stopping_rounds = early_stopping,
            evals_result      = evals_result,
            verbose_eval      = 50,
        )

        self._fitted    = True
        self._evals     = evals_result
        best_round      = self._model.best_iteration
        best_val_loss   = evals_result["val"]["mlogloss"][best_round]

        logger.info(
            f"✅ XGBoost training complete — "
            f"best round: {best_round} | "
            f"val mlogloss: {best_val_loss:.4f}"
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted severity class (0-3) for each sample."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability distribution over severity classes.
        Shape: (n_samples, 4) — columns = [P(0), P(1), P(2), P(3)]
        """
        self._check_fitted()
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Run: pip install xgboost")

        dmat = xgb.DMatrix(X, feature_names=self._feature_names)
        proba = self._model.predict(dmat)
        # XGBoost multi:softprob returns flat array of shape (n*n_classes,)
        # reshape to (n, n_classes)
        n_classes = self.params["num_class"]
        if proba.ndim == 1:
            proba = proba.reshape(-1, n_classes)
        return proba

    def predict_severity(self, X: np.ndarray) -> dict:
        """
        High-level inference interface matching predictor.py expectations.
        Returns {severity, confidence, probabilities, model}.
        """
        proba    = self.predict_proba(X)
        severity = int(np.argmax(proba[0]))
        return {
            "severity":      severity,
            "confidence":    float(proba[0][severity]),
            "probabilities": proba[0].tolist(),
            "model":         "xgboost",
        }

    # ── Explainability ────────────────────────────────────────────────────────

    def explain(
        self,
        X: np.ndarray,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for input samples.
        Returns DataFrame with feature importances.
        Requires: pip install shap
        """
        self._check_fitted()
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")

        explainer   = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)

        # Handle both old (list) and new (3D array) SHAP output formats
        if isinstance(shap_values, list):
            # Old format: list of (n_samples, n_features) per class
            mean_abs = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        elif shap_values.ndim == 3:
            # New format: (n_samples, n_features, n_classes)
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        mean_abs = np.array(mean_abs).flatten()
        names = self._feature_names or [f"f{i}" for i in range(len(mean_abs))]
        importance_df = pd.DataFrame({
            "feature":    names,
            "importance": mean_abs.tolist(),
        }).sort_values("importance", ascending=False).head(top_n)

        return importance_df

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        XGBoost built-in feature importance (faster than SHAP, less accurate).
        importance_type: 'gain', 'weight', 'cover'
        """
        self._check_fitted()
        scores = self._model.get_score(importance_type=importance_type)
        df = pd.DataFrame(
            list(scores.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
        return df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = "ml/artifacts/xgb_model.pkl") -> None:
        """Save the full model wrapper (model + params + feature names)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model":         self._model,
            "params":        self.params,
            "feature_names": self._feature_names,
            "class_weights": self._class_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"💾 XGBoost model saved → {path}")

        # Also save native XGBoost format (lighter, version-safe)
        xgb_path = path.with_suffix(".json")
        self._model.save_model(str(xgb_path))
        logger.info(f"💾 XGBoost model (native) saved → {xgb_path}")

    @classmethod
    def load(cls, path: str | Path = "ml/artifacts/xgb_model.pkl") -> "DrugInteractionXGB":
        """Load a saved model wrapper."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(params=payload["params"])
        instance._model         = payload["model"]
        instance._feature_names = payload["feature_names"]
        instance._class_weights = payload["class_weights"]
        instance._fitted        = True
        logger.info(f"✅ XGBoost model loaded from {path}")
        return instance

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    @staticmethod
    def tune(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        n_trials: int = 50,
        class_weights: Optional[dict] = None,
    ) -> dict:
        """
        Run Optuna hyperparameter search.
        Returns best params dict — pass to DrugInteractionXGB(params=best_params).
        Requires: pip install optuna
        """
        try:
            import optuna
            import xgboost as xgb
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Run: pip install optuna xgboost")

        sample_weights = None
        if class_weights:
            sample_weights = np.array([
                class_weights.get(int(lbl), 1.0) for lbl in y_train
            ])

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        def objective(trial):
            params = {
                "objective":        "multi:softprob",
                "num_class":        4,
                "eval_metric":      "mlogloss",
                "verbosity":        0,
                "max_depth":        trial.suggest_int(   "max_depth",        *OPTUNA_SPACE["max_depth"]),
                "learning_rate":    trial.suggest_float( "learning_rate",    *OPTUNA_SPACE["learning_rate"], log=True),
                "subsample":        trial.suggest_float( "subsample",        *OPTUNA_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float( "colsample_bytree", *OPTUNA_SPACE["colsample_bytree"]),
                "min_child_weight": trial.suggest_int(   "min_child_weight", *OPTUNA_SPACE["min_child_weight"]),
                "gamma":            trial.suggest_float( "gamma",            *OPTUNA_SPACE["gamma"]),
                "reg_alpha":        trial.suggest_float( "reg_alpha",        *OPTUNA_SPACE["reg_alpha"]),
                "reg_lambda":       trial.suggest_float( "reg_lambda",       *OPTUNA_SPACE["reg_lambda"]),
            }
            n_estimators = trial.suggest_int("n_estimators", *OPTUNA_SPACE["n_estimators"])

            model = xgb.train(
                params          = params,
                dtrain          = dtrain,
                num_boost_round = n_estimators,
                evals           = [(dval, "val")],
                early_stopping_rounds = 30,
                verbose_eval    = False,
            )
            return model.best_score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best["objective"]   = "multi:softprob"
        best["num_class"]   = 4
        best["eval_metric"] = ["mlogloss", "merror"]
        best["random_state"] = 42
        best["n_jobs"]       = -1
        best["verbosity"]    = 0

        logger.info(f"✅ Optuna tuning complete — best val mlogloss: {study.best_value:.4f}")
        logger.info(f"   Best params: {best}")
        return best

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

    def training_history(self) -> pd.DataFrame:
        """Return training history as a DataFrame (train + val loss per round)."""
        if not hasattr(self, "_evals"):
            raise RuntimeError("No training history available.")
        rows = []
        for split, metrics in self._evals.items():
            for metric, values in metrics.items():
                for round_num, value in enumerate(values):
                    rows.append({
                        "round":  round_num,
                        "split":  split,
                        "metric": metric,
                        "value":  value,
                    })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="XGBoost drug interaction classifier - sanity check")
    ap.add_argument("--X-train", default="data/processed/X_train.npy")
    ap.add_argument("--y-train", default="data/processed/y_train.npy")
    ap.add_argument("--X-val",   default="data/processed/X_val.npy")
    ap.add_argument("--y-val",   default="data/processed/y_val.npy")
    ap.add_argument("--features",default="data/processed/feature_names.json")
    ap.add_argument("--weights", default="data/processed/class_weights.json")
    ap.add_argument("--out",     default="ml/artifacts/xgb_model.pkl")
    ap.add_argument("--quick",   action="store_true",
                    help="Quick sanity check with 100 estimators")
    args = ap.parse_args()

    # Load data
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train)
    X_val   = np.load(args.X_val)
    y_val   = np.load(args.y_val)

    with open(args.features) as f:
        feature_names = json.load(f)
    with open(args.weights) as f:
        class_weights = {int(k): v for k, v in json.load(f).items()}

    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"Class weights: {class_weights}")

    # Quick check: fewer estimators
    params = {**DEFAULT_PARAMS}
    if args.quick:
        params["n_estimators"] = 100
        logger.info("Quick mode: 100 estimators")

    model = DrugInteractionXGB(params=params)
    model.fit(
        X_train, y_train,
        X_val,   y_val,
        feature_names = feature_names,
        class_weights = class_weights,
    )

    # Quick eval
    from sklearn.metrics import classification_report, roc_auc_score
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    severity_labels = [0, 1, 2, 3]
    severity_names = ["none", "minor", "moderate", "major"]

    print("\n── Classification Report (Validation) ──────────────────────")
    print(classification_report(
        y_val,
        y_pred,
        labels=severity_labels,
        target_names=severity_names,
        zero_division=0,
    ))

    # One-vs-rest AUC
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_val, classes=severity_labels)
    try:
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        print(f"Macro AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"AUC calculation skipped: {e}")

    print("\n── Top 10 Feature Importances (gain) ───────────────────────")
    print(model.feature_importance("gain").head(10).to_string(index=False))

    model.save(args.out)
