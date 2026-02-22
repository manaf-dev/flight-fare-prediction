"""
Loads the trained pipeline artifact and exposes a clean ``predict`` function.

Design decisions:
- The pipeline is loaded once and cached (LRU cache with maxsize=1).
- The same preprocessing logic used at training time is applied at inference
  via ``src.features.build.build_inference`` — no duplication, no skew.
- Returns a structured dict so callers (API, app) get consistent output.
"""

import json
from functools import lru_cache

import joblib
import numpy as np

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    PIPELINE_PATH,
)
from src.features.build import build_inference
from src.utils import get_logger

logger = get_logger(__name__)


# Cached artifact loaders
@lru_cache(maxsize=1)
def _load_pipeline():
    """Load and cache the trained sklearn Pipeline from disk."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Model pipeline not found at '{PIPELINE_PATH}'. "
            "Run the training pipeline first: python pipeline/main.py"
        )
    logger.info("Loading pipeline from: %s", PIPELINE_PATH)
    return joblib.load(PIPELINE_PATH)


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    """Load and cache the model metadata JSON."""
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"feature_cols": ALL_FEATURES, "route_freq_map": {}}


# Public API
def predict(payload: dict) -> dict:
    """
    Predict the total fare (BDT) for a single flight.

    Args:
        payload: Dict containing flight attributes. Required keys mirror the
                 leakage-free feature set defined in ``config.py``.
                 At minimum: ``airline``, ``source``, ``destination``,
                 ``departure_datetime``, ``duration_hrs``,
                 ``days_before_departure``.

    Returns:
        Dict with:
        - ``predicted_total_fare_bdt`` (float): Point estimate.
        - ``prediction_std_bdt`` (float, optional): Std-dev across
          individual trees — only present for tree ensemble models.

    Raises:
        FileNotFoundError: If the model pipeline has not been trained yet.
        ValueError: If the payload is missing critical fields.
    """
    pipeline = _load_pipeline()
    metadata = _load_metadata()

    feature_cols = metadata.get("feature_cols", ALL_FEATURES)
    route_freq_map = metadata.get("route_freq_map", {})

    # Build a single-row DataFrame with all engineered features.
    row_df = build_inference(payload, route_freq_map=route_freq_map)

    # Align to the exact feature set the model was trained on.
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = "Unknown" if col in CATEGORICAL_FEATURES else 0.0

    X = row_df[feature_cols]

    prediction = float(pipeline.predict(X)[0])
    result = {"predicted_total_fare_bdt": round(prediction, 2)}

    # Optional uncertainty estimate from tree ensembles.
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if hasattr(model, "estimators_") and preprocessor is not None:
        X_transformed = preprocessor.transform(X)
        tree_preds = np.array(
            [est.predict(X_transformed)[0] for est in model.estimators_],
            dtype=float,
        )
        result["prediction_std_bdt"] = round(float(np.std(tree_preds)), 2)

    return result
