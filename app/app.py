"""
app/app.py
==========
Streamlit web application for flight fare prediction.

Features:
- Sidebar form with all leakage-free inputs
- Predicted fare display with uncertainty range (if available)
- Contextual pricing insights
- Top 15 feature importance table + chart

Run
---
    streamlit run app/app.py
"""

import json
import sys
from datetime import datetime, time
from pathlib import Path

import streamlit as st

# Ensure project root is on path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEAT_IMPORTANCE_PLOT, MODEL_METADATA_PATH
from src.modeling.inference import predict

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Flight Fare Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

AIRLINES = sorted(
    [
        "Biman Bangladesh Airlines",
        "US-Bangla Airlines",
        "NovoAir",
        "Air Astra",
        "Emirates",
        "Qatar Airways",
        "Etihad Airways",
        "Turkish Airlines",
        "Air Arabia",
        "FlyDubai",
        "Gulf Air",
        "Kuwait Airways",
        "Saudia",
        "Lufthansa",
        "British Airways",
        "Singapore Airlines",
        "Cathay Pacific",
        "Thai Airways",
        "Malaysian Airlines",
        "AirAsia",
        "Air India",
        "IndiGo",
        "Vistara",
        "SriLankan Airlines",
    ]
)

SOURCES = ["DAC", "CGP", "SPD", "BZL", "ZYL", "RJH", "JSR", "CXB"]

DESTINATIONS = sorted(
    [
        "DXB",
        "SIN",
        "BKK",
        "LHR",
        "YYZ",
        "DEL",
        "JFK",
        "KUL",
        "DOH",
        "IST",
        "CCU",
        "JSR",
        "BZL",
        "DAC",
        "SPD",
        "CGP",
        "RJH",
        "ZYL",
        "CXB",
    ]
)

STOPOVERS = ["Direct", "1 Stop", "2 Stops"]
AIRCRAFT_TYPES = [
    "Boeing 737",
    "Boeing 777",
    "Boeing 787",
    "Airbus A320",
    "Airbus A350",
]
CLASSES = ["Economy", "Business", "First Class"]
BOOKING_SRCS = ["Online Website", "Travel Agency", "Direct Booking"]
SEASONALITIES = ["Regular", "Winter Holidays", "Eid", "Hajj"]


def _load_metadata() -> dict:
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _format_bdt(value: float) -> str:
    return f"BDT {value:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------


def render_sidebar() -> dict | None:
    """Render the sidebar form and return payload dict on submit, else None."""
    st.sidebar.title("Flight Details")

    with st.sidebar.form("prediction_form"):
        st.subheader("Route & Airline")
        airline = st.selectbox("Airline", AIRLINES)
        source = st.selectbox("Source", SOURCES)
        destination = st.selectbox("Destination", DESTINATIONS)
        stopovers = st.selectbox("Stopovers", STOPOVERS)

        st.subheader("Flight Specs")
        aircraft = st.selectbox("Aircraft Type", AIRCRAFT_TYPES)
        cls = st.selectbox("Class", CLASSES)
        duration = st.slider(
            "Duration (hours)", min_value=0.5, max_value=20.0, value=3.0, step=0.5
        )

        st.subheader("Booking Info")
        booking_src = st.selectbox("Booking Source", BOOKING_SRCS)
        seasonality = st.selectbox("Seasonality", SEASONALITIES)
        days_before = st.slider(
            "Days Before Departure", min_value=0, max_value=120, value=30
        )

        st.subheader("Departure")
        dep_date = st.date_input("Departure Date", value=datetime.now().date())
        dep_time = st.time_input("Departure Time", value=time(12, 0))

        submitted = st.form_submit_button(
            "Predict Fare", type="primary", use_container_width=True
        )

    if not submitted:
        return None

    dep_dt = datetime.combine(dep_date, dep_time)

    return {
        "airline": airline,
        "source": source,
        "destination": destination,
        "stopovers": stopovers,
        "aircraft_type": aircraft,
        "class": cls,
        "booking_source": booking_src,
        "seasonality": seasonality,
        "departure_datetime": dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_hrs": duration,
        "days_before_departure": days_before,
    }


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


def render_insights(days_before: int, seasonality: str, flight_class: str) -> None:
    """Show contextual pricing tips based on the user's inputs."""
    st.subheader("Pricing Insights")

    tips = []

    if days_before <= 7:
        tips.append(
            "⚠️ **Last-minute booking** — fares are typically 20–40% higher than booking 30+ days out."
        )
    elif days_before >= 60:
        tips.append(
            "✅ **Booking well in advance** — you're likely to get a better price."
        )

    if seasonality in {"Eid", "Hajj"}:
        tips.append(
            "🌙 **Peak season** (Eid/Hajj) — demand is high, fares are elevated."
        )
    elif seasonality == "Winter Holidays":
        tips.append(
            "❄️ **Holiday season** — expect higher fares on international routes."
        )
    else:
        tips.append("📅 **Regular season** — fares are closer to baseline pricing.")

    if flight_class == "Business":
        tips.append(
            "💼 **Business Class** — typically 3–5× the Economy fare on the same route."
        )
    elif flight_class == "First Class":
        tips.append("🥇 **First Class** — premium fares, often 6–10× Economy.")

    for tip in tips:
        st.info(tip)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main():
    # --- Header ---
    st.title("Flight Fare Predictor — Bangladesh")
    st.markdown(
        "Predict total flight fares using a machine learning model trained on "
        "Bangladesh route data. Inputs are **leakage-free**: no base fare or "
        "tax inputs are required."
    )

    col_main, col_info = st.columns([3, 1])

    with col_info:
        meta = _load_metadata()
        if meta:
            st.metric("Best Model", meta.get("best_model", "—"))
            st.metric("Training Features", len(meta.get("feature_cols", [])))
        else:
            st.warning("Model not trained yet. Run `python pipeline/main.py` first.")

    # --- Sidebar form ---
    payload = render_sidebar()

    if payload is None:
        with col_main:
            st.info(
                "👈 Fill in the flight details on the left and click **Predict Fare**."
            )

            # Show feature importance chart even before prediction.
            if FEAT_IMPORTANCE_PLOT.exists():
                st.subheader("📊 Top 15 Model Drivers")
                st.image(str(FEAT_IMPORTANCE_PLOT), use_container_width=True)
        return

    # --- Run prediction ---
    with st.spinner("Predicting..."):
        try:
            result = predict(payload)
        except FileNotFoundError:
            st.error(
                "Model not found. Please run the training pipeline first:\n\n"
                "```bash\npython pipeline/main.py\n```"
            )
            return
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

    # --- Display result ---
    with col_main:
        st.success("Prediction complete!")

        fare = result["predicted_total_fare_bdt"]
        std = result.get("prediction_std_bdt")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Estimated Total Fare",
                value=_format_bdt(fare),
            )
        with col_b:
            if std:
                st.metric(
                    label="Uncertainty (±1 std)",
                    value=_format_bdt(std),
                    help="Standard deviation across individual tree predictions. "
                    "A lower value means the model is more confident.",
                )

        if std:
            lower = max(0, fare - std)
            upper = fare + std
            st.caption(f"Estimated range: {_format_bdt(lower)} — {_format_bdt(upper)}")

        st.divider()

        # Insights
        render_insights(
            days_before=payload["days_before_departure"],
            seasonality=payload["seasonality"],
            flight_class=payload["class"],
        )

        st.divider()

        # Feature importance
        st.subheader("📊 Top 15 Model Drivers")

        meta = _load_metadata()
        top_features = meta.get("top_features", [])

        if top_features:
            import pandas as pd

            feat_df = pd.DataFrame(top_features[:15]).rename(
                columns={"feature": "Feature", "importance": "Importance"}
            )
            feat_df["Feature"] = (
                feat_df["Feature"].str.replace("cat__", "").str.replace("num__", "")
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        if FEAT_IMPORTANCE_PLOT.exists():
            st.image(str(FEAT_IMPORTANCE_PLOT), use_container_width=True)

        # Raw payload expander for transparency
        with st.expander("View prediction payload"):
            st.json(payload)

        with st.expander("View raw model output"):
            st.json(result)


if __name__ == "__main__":
    main()