"""
IntelliPipe Ensemble Anomaly Detection Engine
==============================================
Three-model ensemble with configurable weights:
1. Isolation Forest  — fast, tree-based; great for tabular outliers
2. Autoencoder       — deep learning; captures complex multivariate patterns
3. Z-Score baseline  — statistical fallback; robust for single-feature spikes

Each model is independently tracked in MLflow.
Shadow evaluation mode allows new models to run without affecting production scoring.
Feature importance attribution for explainable anomaly explanations.
"""

from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.telemetry import ANOMALIES_DETECTED_TOTAL, ANOMALY_SCORE_HISTOGRAM

logger = get_logger(__name__, component="anomaly_engine")
settings = get_settings()


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Unified anomaly detection result with explainability metadata.
    """
    is_anomaly: bool
    ensemble_score: float          # 0-1: higher = more anomalous
    if_score: float                # Isolation Forest normalised score
    ae_score: float                # Autoencoder reconstruction error
    zscore_max: float              # Maximum Z-score across features
    severity: str                  # critical / high / medium / low
    anomaly_type: str              # statistical / null_spike / outlier / etc.
    affected_features: List[str]   # Top features driving the anomaly
    feature_contributions: Dict[str, float]  # Feature → contribution score
    explanation: str               # Human-readable explanation
    model_versions: Dict[str, str] = field(default_factory=dict)
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


SEVERITY_THRESHOLDS = {
    "critical": 0.85,
    "high": 0.70,
    "medium": 0.50,
    "low": 0.30,
}


def score_to_severity(score: float) -> str:
    for level, threshold in SEVERITY_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "info"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ORDER_NUMERIC_FEATURES = [
    "total_amount",
    "subtotal",
    "tax_amount",
    "shipping_amount",
    "item_count",
    "avg_item_price",
    "max_item_quantity",
    "total_quantity",
    "discount_variance",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw order events into numeric feature matrix.
    Handles missing values with median imputation.
    """
    features = pd.DataFrame()

    if "total_amount" in df.columns:
        features["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)
    else:
        features["total_amount"] = 0.0

    if "subtotal" in df.columns:
        features["subtotal"] = pd.to_numeric(df["subtotal"], errors="coerce").fillna(0)
    else:
        features["subtotal"] = 0.0

    for col in ["tax_amount", "shipping_amount"]:
        features[col] = pd.to_numeric(df.get(col, pd.Series([0] * len(df))), errors="coerce").fillna(0)

    # Derived features from items array (pre-exploded)
    if "item_count" in df.columns:
        features["item_count"] = pd.to_numeric(df["item_count"], errors="coerce").fillna(1)
    else:
        features["item_count"] = 1.0

    if "avg_item_price" in df.columns:
        features["avg_item_price"] = pd.to_numeric(df["avg_item_price"], errors="coerce").fillna(0)
    else:
        features["avg_item_price"] = features["total_amount"] / features["item_count"].clip(lower=1)

    if "max_item_quantity" in df.columns:
        features["max_item_quantity"] = pd.to_numeric(df["max_item_quantity"], errors="coerce").fillna(1)
    else:
        features["max_item_quantity"] = 1.0

    if "total_quantity" in df.columns:
        features["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce").fillna(1)
    else:
        features["total_quantity"] = features["item_count"]

    # Discount variance (spread of discounts across items)
    if "discount_variance" in df.columns:
        features["discount_variance"] = pd.to_numeric(df["discount_variance"], errors="coerce").fillna(0)
    else:
        features["discount_variance"] = 0.0

    # Fill remaining NaN with column median
    for col in features.columns:
        median = features[col].median()
        features[col] = features[col].fillna(median if not pd.isna(median) else 0)

    return features


# ---------------------------------------------------------------------------
# Isolation Forest detector
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Isolation Forest wrapper with MLflow experiment tracking.
    Uses sklearn's IsolationForest with hyperparameter logging.
    """

    MODEL_NAME = "isolation_forest"

    def __init__(self) -> None:
        cfg = settings.anomaly
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._version: str = "untrained"
        self._contamination = cfg.isolation_forest_contamination
        self._n_estimators = cfg.isolation_forest_n_estimators

    def train(
        self,
        df: pd.DataFrame,
        mlflow_experiment: str,
        table_name: str = "orders",
    ) -> str:
        """Train model and log to MLflow. Returns MLflow run ID."""
        X = extract_features(df)

        with mlflow.start_run(
            experiment_id=mlflow.set_experiment(mlflow_experiment).experiment_id,
            run_name=f"IF_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ) as run:
            mlflow.log_params({
                "contamination": self._contamination,
                "n_estimators": self._n_estimators,
                "random_state": 42,
                "features": ",".join(X.columns.tolist()),
                "training_rows": len(X),
                "table_name": table_name,
            })

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._model = IsolationForest(
                contamination=self._contamination,
                n_estimators=self._n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(X_scaled)

            # Log model metrics
            anomaly_pct = (self._model.predict(X_scaled) == -1).mean()
            mlflow.log_metrics({
                "training_anomaly_pct": float(anomaly_pct),
                "avg_path_length": float(np.mean(self._model.score_samples(X_scaled))),
            })
            mlflow.sklearn.log_model(self._model, "isolation_forest_model")

            self._version = run.info.run_id[:8]
            logger.info(
                "Isolation Forest trained",
                run_id=run.info.run_id,
                training_rows=len(X),
                anomaly_pct=anomaly_pct,
            )
            return run.info.run_id

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return normalised anomaly scores [0, 1] for each row.
        Higher = more anomalous.
        """
        if self._model is None or self._scaler is None:
            return np.zeros(len(df))

        X = extract_features(df)
        X_scaled = self._scaler.transform(X)

        # sklearn returns negative values; more negative = more anomalous
        raw_scores = self._model.score_samples(X_scaled)
        # Normalise to [0, 1] — invert and rescale
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            return np.zeros(len(df))
        normalised = 1 - (raw_scores - min_s) / (max_s - min_s)
        return np.clip(normalised, 0, 1)

    def feature_contributions(
        self, row: pd.Series
    ) -> Dict[str, float]:
        """Approximate per-feature contribution via perturbation analysis."""
        if self._model is None or self._scaler is None:
            return {}

        df = pd.DataFrame([row])
        X = extract_features(df)
        X_scaled = self._scaler.transform(X)
        base_score = float(self._model.score_samples(X_scaled)[0])

        contributions = {}
        for col in X.columns:
            perturbed = X.copy()
            perturbed[col] = 0  # Zero-out feature
            ps = self._scaler.transform(perturbed)
            perturbed_score = float(self._model.score_samples(ps)[0])
            contributions[col] = abs(base_score - perturbed_score)

        # Normalise contributions
        total = sum(contributions.values()) or 1
        return {k: round(v / total, 4) for k, v in contributions.items()}


# ---------------------------------------------------------------------------
# Autoencoder detector (PyTorch-based)
# ---------------------------------------------------------------------------

class AutoencoderDetector:
    """
    Simple feed-forward autoencoder for reconstruction-error anomaly detection.
    Falls back gracefully if PyTorch is not available.
    """

    MODEL_NAME = "autoencoder"

    def __init__(self) -> None:
        cfg = settings.anomaly
        self._latent_dim = cfg.autoencoder_latent_dim
        self._epochs = cfg.autoencoder_epochs
        self._batch_size = cfg.autoencoder_batch_size
        self._scaler: Optional[StandardScaler] = None
        self._threshold: float = 0.0
        self._version: str = "untrained"

        try:
            import torch
            import torch.nn as nn
            self._torch_available = True
            self._torch = torch
            self._nn = nn
        except ImportError:
            self._torch_available = False
            logger.warning("PyTorch not available; Autoencoder detector disabled")

    def _build_model(self, input_dim: int) -> Any:
        """Construct encoder-decoder architecture."""
        nn = self._nn
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def train(self, df: pd.DataFrame, mlflow_experiment: str, table_name: str = "orders") -> str:
        """Train autoencoder on clean training data."""
        if not self._torch_available:
            return "disabled"

        torch = self._torch
        X = extract_features(df).values.astype(np.float32)

        with mlflow.start_run(
            experiment_id=mlflow.set_experiment(mlflow_experiment).experiment_id,
            run_name=f"AE_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ) as run:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X).astype(np.float32)

            input_dim = X_scaled.shape[1]
            model = self._build_model(input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = self._nn.MSELoss()

            X_tensor = torch.FloatTensor(X_scaled)
            dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self._batch_size, shuffle=True
            )

            mlflow.log_params({
                "latent_dim": self._latent_dim,
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "input_dim": input_dim,
                "training_rows": len(X),
            })

            losses = []
            for epoch in range(self._epochs):
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(loader)
                losses.append(avg_loss)
                if epoch % 10 == 0:
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Set anomaly threshold at 95th percentile of training reconstruction errors
            model.eval()
            with torch.no_grad():
                reconstructed = model(X_tensor)
                errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            self._threshold = float(np.percentile(errors, 95))
            self._model = model

            mlflow.log_metrics({
                "final_train_loss": losses[-1],
                "anomaly_threshold": self._threshold,
            })

            self._version = run.info.run_id[:8]
            logger.info(
                "Autoencoder trained",
                run_id=run.info.run_id,
                final_loss=losses[-1],
                threshold=self._threshold,
            )
            return run.info.run_id

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return normalised anomaly scores based on reconstruction error."""
        if not self._torch_available or self._scaler is None or not hasattr(self, "_model"):
            return np.zeros(len(df))

        torch = self._torch
        X = extract_features(df).values.astype(np.float32)
        X_scaled = self._scaler.transform(X).astype(np.float32)
        X_tensor = torch.FloatTensor(X_scaled)

        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

        if self._threshold == 0:
            return np.zeros(len(df))

        # Normalise: errors above threshold → closer to 1
        normalised = np.clip(errors / (self._threshold * 3), 0, 1)
        return normalised


# ---------------------------------------------------------------------------
# Z-Score fallback detector
# ---------------------------------------------------------------------------

class ZScoreDetector:
    """
    Simple Z-score anomaly detection for numeric features.
    Robust to distribution shifts via Modified Z-Score (Median Absolute Deviation).
    """

    MODEL_NAME = "zscore"

    def __init__(self) -> None:
        self._medians: Dict[str, float] = {}
        self._mads: Dict[str, float] = {}
        self._threshold = settings.anomaly.zscore_threshold
        self._version = "statistical"

    def fit(self, df: pd.DataFrame) -> None:
        """Compute per-column medians and MADs from training data."""
        X = extract_features(df)
        for col in X.columns:
            self._medians[col] = float(X[col].median())
            self._mads[col] = float(stats.median_abs_deviation(X[col].dropna()))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Modified Z-Score per row.
        Returns max modified Z-score across all features, normalised to [0,1].
        """
        X = extract_features(df)
        max_zscores = np.zeros(len(X))

        for col in X.columns:
            if col not in self._medians or self._mads.get(col, 0) == 0:
                continue
            # Modified Z-Score: 0.6745 * (x - median) / MAD
            modified_z = 0.6745 * np.abs(X[col] - self._medians[col]) / max(self._mads[col], 1e-10)
            max_zscores = np.maximum(max_zscores, modified_z.values)

        # Normalise: z >= threshold → anomaly
        normalised = np.clip(max_zscores / (self._threshold * 2), 0, 1)
        return normalised

    def max_zscores_per_row(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Return max Z-scores and the feature name responsible."""
        X = extract_features(df)
        max_scores = np.zeros(len(X))
        responsible_features = [""] * len(X)

        for col in X.columns:
            if col not in self._medians or self._mads.get(col, 0) == 0:
                continue
            z = 0.6745 * np.abs(X[col] - self._medians[col]) / max(self._mads[col], 1e-10)
            for i, (cur_max, cur_z) in enumerate(zip(max_scores, z.values)):
                if cur_z > cur_max:
                    max_scores[i] = cur_z
                    responsible_features[i] = col

        return max_scores, responsible_features


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

class EnsembleAnomalyScorer:
    """
    Weighted ensemble of all three detectors.
    Produces AnomalyResult with explainability metadata.
    Supports shadow model evaluation for model governance.
    """

    def __init__(
        self,
        if_detector: IsolationForestDetector,
        ae_detector: AutoencoderDetector,
        zscore_detector: ZScoreDetector,
    ) -> None:
        self._if = if_detector
        self._ae = ae_detector
        self._zscore = zscore_detector

        cfg = settings.anomaly
        self._weights = {
            "isolation_forest": cfg.ensemble_weights_if,
            "autoencoder": cfg.ensemble_weights_ae,
            "zscore": cfg.ensemble_weights_zscore,
        }

    def score_batch(
        self,
        df: pd.DataFrame,
        table_name: str = "orders",
        shadow_only: bool = False,
    ) -> List[AnomalyResult]:
        """
        Score a batch of events. Returns one AnomalyResult per row.

        Args:
            shadow_only: If True, scores are computed but not used for alerts.
        """
        if_scores = self._if.score(df)
        ae_scores = self._ae.score(df)
        zscore_raw, responsible_features = self._zscore.max_zscores_per_row(df)
        zscore_scores = self._zscore.score(df)

        ensemble = (
            self._weights["isolation_forest"] * if_scores +
            self._weights["autoencoder"] * ae_scores +
            self._weights["zscore"] * zscore_scores
        )

        results = []
        for i, (row_idx, row) in enumerate(df.iterrows()):
            score = float(ensemble[i])
            is_anomaly = score >= SEVERITY_THRESHOLDS["low"]
            severity = score_to_severity(score)

            # Feature contribution analysis (Isolation Forest)
            contribs = self._if.feature_contributions(row) if is_anomaly else {}
            top_features = sorted(contribs, key=contribs.get, reverse=True)[:3]

            explanation = self._generate_explanation(
                score=score,
                if_score=float(if_scores[i]),
                ae_score=float(ae_scores[i]),
                zscore_max=float(zscore_raw[i]),
                top_features=top_features,
                responsible_feature=responsible_features[i],
            )

            result = AnomalyResult(
                is_anomaly=is_anomaly,
                ensemble_score=round(score, 4),
                if_score=round(float(if_scores[i]), 4),
                ae_score=round(float(ae_scores[i]), 4),
                zscore_max=round(float(zscore_raw[i]), 4),
                severity=severity,
                anomaly_type=self._classify_type(row, score, responsible_features[i]),
                affected_features=top_features,
                feature_contributions=contribs,
                explanation=explanation,
                model_versions={
                    "isolation_forest": self._if._version,
                    "autoencoder": self._ae._version,
                    "zscore": self._zscore._version,
                },
            )

            if is_anomaly and not shadow_only:
                ANOMALIES_DETECTED_TOTAL.labels(
                    table=table_name,
                    anomaly_type=result.anomaly_type,
                    severity=severity,
                ).inc()
                ANOMALY_SCORE_HISTOGRAM.labels(
                    table=table_name,
                    model_type="ensemble",
                ).observe(score)

            results.append(result)

        return results

    @staticmethod
    def _classify_type(row: pd.Series, score: float, top_feature: str) -> str:
        """Heuristic-based anomaly type classification."""
        if top_feature in ["total_amount", "avg_item_price", "subtotal"]:
            return "statistical_price"
        if top_feature in ["max_item_quantity", "total_quantity"]:
            return "statistical_quantity"
        if top_feature in ["tax_amount", "shipping_amount"]:
            return "statistical_fees"
        return "statistical"

    @staticmethod
    def _generate_explanation(
        score: float,
        if_score: float,
        ae_score: float,
        zscore_max: float,
        top_features: List[str],
        responsible_feature: str,
    ) -> str:
        """Generate a human-readable anomaly explanation."""
        parts = [f"Ensemble anomaly score: {score:.2%}."]

        if if_score > 0.6:
            parts.append(
                f"Isolation Forest (score={if_score:.2f}) indicates "
                "this record is structurally isolated from normal patterns."
            )
        if ae_score > 0.5:
            parts.append(
                f"Autoencoder reconstruction error (score={ae_score:.2f}) "
                "suggests unusual feature combinations."
            )
        if zscore_max > 3.0:
            parts.append(
                f"Statistical Z-score of {zscore_max:.1f} on '{responsible_feature}' "
                "exceeds the 3-sigma threshold."
            )
        if top_features:
            parts.append(f"Top contributing features: {', '.join(top_features)}.")

        return " ".join(parts)
