"""Streamlit app for flight fare prediction."""

import json
import os
import sys
from datetime import datetime, time

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import MODEL_METADATA_PATH, TOP_FEATURES_PLOT_PATH
from src.inference import predict

st.set_page_config(page_title="Flight Fare Predictor", layout="wide")


def _load_top_features():
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("top_features", [])
    return []


def _render_insights(days_before_departure, seasonality):
    notes = []
    if days_before_departure <= 7:
        notes.append("Last-minute bookings tend to be more expensive.")
    if seasonality in {"Hajj", "Eid", "Winter Holidays"}:
        notes.append("Holiday and pilgrimage seasons often increase fares.")
    if not notes:
        notes.append("Booking farther in advance during regular seasons often improves prices.")

    for note in notes:
        st.info(note)


def main():
    st.title("Flight Fare Predictor")
    st.caption("This app predicts the total fare of a flight based on various inputs.")

    st.sidebar.header("Flight Inputs")

    airline = st.sidebar.selectbox(
        "Airline",
        [
            "Turkish Airlines",
            "AirAsia",
            "Cathay Pacific",
            "Thai Airways",
            "Malaysian Airlines",
            "IndiGo",
            "Air India",
            "US-Bangla Airlines",
            "Kuwait Airways",
            "Etihad Airways",
            "Gulf Air",
            "SriLankan Airlines",
            "British Airways",
            "Biman Bangladesh Airlines",
            "Emirates",
            "Air Arabia",
            "Qatar Airways",
            "Lufthansa",
            "Saudia",
            "FlyDubai",
            "Air Astra",
            "NovoAir",
            "Singapore Airlines",
            "Vistara",
        ],
    )

    source = st.sidebar.selectbox("Source", ["DAC", "CGP", "SPD", "BZL", "ZYL", "RJH", "JSR", "CXB"])
    destination = st.sidebar.selectbox(
        "Destination",
        ["DXB", "SIN", "BKK", "LHR", "YYZ", "DEL", "JFK", "KUL", "DOH", "IST", "CCU", "JSR", "BZL", "DAC", "SPD", "CGP", "RJH", "ZYL", "CXB"],
    )
    stopovers = st.sidebar.selectbox("Stopovers", ["Direct", "1 Stop", "2 Stops"])
    aircraft_type = st.sidebar.selectbox(
        "Aircraft Type", ["Boeing 737", "Boeing 777", "Boeing 787", "Airbus A320", "Airbus A350"]
    )
    flight_class = st.sidebar.selectbox("Class", ["Economy", "Business", "First Class"])
    booking_source = st.sidebar.selectbox(
        "Booking Source", ["Online Website", "Travel Agency", "Direct Booking"]
    )
    seasonality = st.sidebar.selectbox(
        "Seasonality", ["Regular", "Winter Holidays", "Eid", "Hajj"]
    )

    departure_date = st.sidebar.date_input("Departure Date", datetime.now().date())
    departure_time = st.sidebar.time_input("Departure Time", time(12, 0))
    duration_hrs = st.sidebar.slider("Duration (hrs)", 0.5, 20.0, 2.0, 0.1)
    days_before_departure = st.sidebar.slider("Days Before Departure", 0, 120, 30)

    if st.sidebar.button("Predict Fare", type="primary"):
        departure_datetime = datetime.combine(departure_date, departure_time)
        payload = {
            "airline": airline,
            "source": source,
            "destination": destination,
            "stopovers": stopovers,
            "aircraft_type": aircraft_type,
            "class": flight_class,
            "booking_source": booking_source,
            "seasonality": seasonality,
            "departure_datetime": departure_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hrs": duration_hrs,
            "days_before_departure": days_before_departure,
        }

        try:
            prediction = predict(payload)
            value = prediction["predicted_total_fare_bdt"]

            st.subheader("Predicted Total Fare (BDT)")
            st.metric("Estimated Fare", f"BDT {value:,.0f}")

            if "prediction_std_bdt" in prediction:
                st.caption(f"Uncertainty proxy (std across trees): BDT {prediction['prediction_std_bdt']:,.0f}")

            st.subheader("Insights")
            _render_insights(days_before_departure, seasonality)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    st.subheader("Top 15 Model Drivers")
    top_features = _load_top_features()

    if top_features:
        table = [{"feature": f["feature"].replace("cat__", "").replace("num__", ""), "importance": round(f["importance"], 5)} for f in top_features[:15]]
        st.dataframe(table, use_container_width=True)

    if TOP_FEATURES_PLOT_PATH.exists():
        st.image(str(TOP_FEATURES_PLOT_PATH), caption="Top 15 Feature Importances", use_container_width=True)


if __name__ == "__main__":
    main()
