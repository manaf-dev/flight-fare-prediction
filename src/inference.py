"""Inference module for leakage-free flight fare prediction."""

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
    NUMERICAL_FEATURES,
    PIPELINE_PATH,
)
from src.data_preprocessing import preprocess_dataframe


@lru_cache(maxsize=1)
def _load_pipeline():
    return joblib.load(PIPELINE_PATH)


@lru_cache(maxsize=1)
def _load_metadata():
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "expected_input_schema": [
            c for c in (CATEGORICAL_FEATURES + NUMERICAL_FEATURES) if c != "total_fare_bdt"
        ]
    }



def _align_features(df: pd.DataFrame, expected_cols):
    aligned = df.copy()
    for col in expected_cols:
        if col not in aligned.columns:
            if col in CATEGORICAL_FEATURES:
                aligned[col] = "Unknown"
            else:
                aligned[col] = 0.0
    return aligned[expected_cols]



def predict(payload: dict) -> dict:
    """Predict total fare from a single payload dictionary."""
    pipeline = _load_pipeline()
    metadata = _load_metadata()

    raw = pd.DataFrame([payload])
    processed = preprocess_dataframe(raw, inference_mode=True)

    expected = metadata.get("expected_input_schema", [])
    X = _align_features(processed, expected)

    pred_value = float(pipeline.predict(X)[0])
    result = {"predicted_total_fare_bdt": pred_value}

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if hasattr(model, "estimators_") and preprocessor is not None:
        transformed = preprocessor.transform(X)
        tree_preds = np.array([est.predict(transformed)[0] for est in model.estimators_], dtype=float)
        result["prediction_std_bdt"] = float(np.std(tree_preds))

    return result



def predict_fare(input_data):
    """Backward-compatible fare-only method for existing callers."""
    return predict(input_data)["predicted_total_fare_bdt"]
