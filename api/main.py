"""
api/main.py
===========
FastAPI REST API for flight fare predictions.

Endpoints
---------
GET  /health     nfig import FEAT_IMPORTANCE_PLOT, MODEL_METADATA_PATH
from src.modeling.inference import predict         — liveness check
GET  /metadata            — model metadata (features, best model name, etc.)
POST /predict             — predict total fare for a single flight
POST /predict/batch       — predict total fare for multiple flights

Run
---
    uvicorn api.main:app --reload --port 8000

Then call:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"airline":"Emirates","source":"DAC","destination":"DXB",
              "stopovers":"Direct","aircraft_type":"Boeing 777",
              "class":"Economy","booking_source":"Online Website",
              "seasonality":"Regular","departure_datetime":"2025-06-15 14:00:00",
              "duration_hrs":4.5,"days_before_departure":30}'
"""

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_METADATA_PATH
from src.modeling.inference import predict
from src.utils import get_logger

logger = get_logger(__name__)

# App setup
app = FastAPI(
    title="Flight Fare Prediction API",
    description=(
        "Predict flight fares for Bangladesh routes using a leakage-free "
        "machine learning model. Excludes base fare and tax inputs to ensure "
        "realistic, deployable predictions."
    ),
    version="1.0.0",
)


# Request / response schemas
class FlightInput(BaseModel):
    """Schema for a single flight prediction request."""

    airline: str = Field(..., example="Emirates")
    source: str = Field(..., example="DAC")
    destination: str = Field(..., example="DXB")
    stopovers: str = Field("Direct", example="Direct")
    aircraft_type: str = Field("Boeing 737", example="Boeing 777")
    flight_class: str = Field("Economy", alias="class", example="Economy")
    booking_source: str = Field("Online Website", example="Online Website")
    seasonality: str = Field("Regular", example="Regular")
    departure_datetime: str = Field(..., example="2025-06-15 14:00:00")
    duration_hrs: float = Field(..., ge=0.1, le=48.0, example=4.5)
    days_before_departure: int = Field(..., ge=0, le=365, example=30)

    model_config = {"populate_by_name": True}

    @field_validator("duration_hrs")
    @classmethod
    def duration_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("duration_hrs must be positive.")
        return v


class PredictionResponse(BaseModel):
    """Schema for a prediction response."""

    predicted_total_fare_bdt: float
    prediction_std_bdt: Optional[float] = None
    currency: str = "BDT"
    model: Optional[str] = None


class BatchInput(BaseModel):
    """Schema for batch prediction requests."""

    flights: list[FlightInput] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# Helpers
def _flight_input_to_payload(flight: FlightInput) -> dict:
    """Convert a FlightInput model to the dict expected by ``predict()``."""
    return {
        "airline": flight.airline,
        "source": flight.source,
        "destination": flight.destination,
        "stopovers": flight.stopovers,
        "aircraft_type": flight.aircraft_type,
        "class": flight.flight_class,
        "booking_source": flight.booking_source,
        "seasonality": flight.seasonality,
        "departure_datetime": flight.departure_datetime,
        "duration_hrs": flight.duration_hrs,
        "days_before_departure": flight.days_before_departure,
    }


def _load_best_model_name() -> Optional[str]:
    try:
        with open(MODEL_METADATA_PATH, "r") as fh:
            return json.load(fh).get("best_model")
    except Exception:
        return None


# Endpoints
@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness check.

    Returns 200 if the API is running. Useful for container health probes.
    """
    return {"status": "ok"}


@app.get("/metadata", tags=["System"])
def get_metadata():
    """
    Return model metadata: best model name, features, training timestamp, etc.
    """
    if not MODEL_METADATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python pipeline/main.py` first.",
        )
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(flight: FlightInput):
    """
    Predict the total fare (BDT) for a single flight.

    - Does **not** require base fare or tax inputs (leakage-free).
    - Returns a point estimate and, for tree models, a standard deviation
      across individual tree predictions as an uncertainty proxy.
    """
    payload = _flight_input_to_payload(flight)
    try:
        result = predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        predicted_total_fare_bdt=result["predicted_total_fare_bdt"],
        prediction_std_bdt=result.get("prediction_std_bdt"),
        model=_load_best_model_name(),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict total fares for up to 100 flights in one request.

    Useful for bulk pricing lookups or comparison tools.
    """
    predictions = []
    best_model = _load_best_model_name()

    for flight in batch.flights:
        payload = _flight_input_to_payload(flight)
        try:
            result = predict(payload)
            predictions.append(
                PredictionResponse(
                    predicted_total_fare_bdt=result["predicted_total_fare_bdt"],
                    prediction_std_bdt=result.get("prediction_std_bdt"),
                    model=best_model,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            logger.exception("Batch prediction item failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return BatchResponse(predictions=predictions)
