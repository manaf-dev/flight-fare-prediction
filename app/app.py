"""
Streamlit App for Flight Fare Prediction
Interactive web application for predicting flight fares using the trained Gradient Boosting model.
"""

import importlib
import os
import sys
from datetime import datetime, time

import streamlit as st

from src import inference as inference_module
from src.config import VISUALIZATIONS_PATH

# Add src to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

inference_module = importlib.reload(inference_module)

# Page configuration
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        '<div class="main-header">Flight Fare Predictor</div>', unsafe_allow_html=True
    )
    st.markdown("### Predict flight fares using advanced machine learning")

    # Sidebar for inputs
    st.sidebar.markdown(
        '<div class="sidebar-header">Flight Details</div>', unsafe_allow_html=True
    )

    # Airline selection
    airlines = [
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
    ]
    airline = st.sidebar.selectbox("Airline", airlines)

    # Source and Destination
    sources = ["DAC", "CGP", "SPD", "BZL", "ZYL", "RJH", "JSR", "CXB"]
    destinations = [
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

    col1, col2 = st.sidebar.columns(2)
    with col1:
        source = st.selectbox("Source Airport", sources)
    with col2:
        destination = st.selectbox("Destination Airport", destinations)

    # Flight details
    col3, col4 = st.sidebar.columns(2)
    with col3:
        stopovers = st.selectbox("Stopovers", ["Direct", "1 Stop", "2 Stops"])
    with col4:
        aircraft_type = st.selectbox(
            "Aircraft Type",
            ["Boeing 737", "Boeing 777", "Boeing 787", "Airbus A320", "Airbus A350"],
        )

    # Class and Booking Source
    col5, col6 = st.sidebar.columns(2)
    with col5:
        flight_class = st.selectbox("Class", ["Economy", "Business", "First Class"])
    with col6:
        booking_source = st.selectbox(
            "Booking Source", ["Online Website", "Travel Agency", "Direct Booking"]
        )

    # Date and time
    st.sidebar.markdown("### Departure Details")
    departure_date = st.sidebar.date_input("Departure Date", datetime.now().date())
    departure_time = st.sidebar.time_input("Departure Time", time(12, 0))

    # Combine date and time
    departure_datetime = datetime.combine(departure_date, departure_time)

    # Duration and fares
    st.sidebar.markdown("### Flight Details")
    duration_hrs = st.sidebar.slider("Duration (hours)", 0.5, 16.0, 2.0, 0.1)

    col7, col8 = st.sidebar.columns(2)
    with col7:
        base_fare = st.sidebar.number_input(
            "Base Fare (BDT)", min_value=1000, max_value=500000, value=30000
        )
    with col8:
        tax_surcharge = st.sidebar.number_input(
            "Tax & Surcharge (BDT)", min_value=0, max_value=100000, value=5000
        )

    # Days before departure
    days_before = st.sidebar.slider("Days Before Departure", 1, 90, 30)

    # Prediction button
    if st.sidebar.button("Predict Fare", type="primary"):
        # Prepare input data
        input_data = {
            "airline": airline,
            "source": source,
            "destination": destination,
            "departure_date_and_time": departure_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hrs": duration_hrs,
            "stopovers": stopovers,
            "aircraft_type": aircraft_type,
            "class": flight_class,
            "booking_source": booking_source,
            "base_fare_bdt": base_fare,
            "tax_and_surcharge_bdt": tax_surcharge,
            "days_before_departure": days_before,
        }

        try:
            # Make prediction
            with st.spinner("Predicting fare..."):
                predicted_fare = inference_module.predict_fare(input_data)

            # Display results
            st.markdown("## Prediction Results")

            # Main prediction box
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Predicted Total Fare")
            st.markdown(f"## BDT {predicted_fare:,.0f}")
            st.markdown("*Bangladeshi Taka (BDT)*")
            st.markdown("</div>", unsafe_allow_html=True)

            # Additional metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Base Fare", f"BDT {base_fare:,.0f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Tax & Surcharge", f"BDT {tax_surcharge:,.0f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                actual_total = base_fare + tax_surcharge
                difference = predicted_fare - actual_total
                st.metric(
                    "Difference",
                    f"BDT {difference:,.0f}",
                    delta=f"{difference / actual_total * 100:.1f}%"
                    if actual_total > 0
                    else "N/A",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Insights
            st.markdown("### Insights")

            # Route popularity
            route_popularity = {
                ("RJH", "SIN"): "Very Popular",
                ("DAC", "DXB"): "Very Popular",
                ("BZL", "YYZ"): "Very Popular",
                ("CGP", "BKK"): "Popular",
                ("CXB", "DEL"): "Popular",
            }

            route_key = (source, destination)
            if route_key in route_popularity:
                st.info(
                    f"This route is **{route_popularity[route_key]}** - expect higher demand and potentially higher fares."
                )

            # Seasonal insights
            month = departure_date.month
            if month in [11, 12, 1]:  # Winter holidays
                st.info(
                    "**Winter Holiday Season** - Expect higher fares due to increased travel demand."
                )
            elif month in [4, 5]:  # Eid
                st.info(
                    "**Eid Season** - Religious holidays may cause fare fluctuations."
                )
            elif month in [6, 7, 8]:  # Hajj
                st.info("**Hajj Season** - Peak season with highest fares.")
            else:
                st.info(
                    "**Regular Season** - Generally lower fares with better deals available."
                )

            # Airline insights
            airline_insights = {
                "Turkish Airlines": "Premium carrier with high fares but excellent service",
                "AirAsia": "Budget airline offering competitive low-cost fares",
                "Cathay Pacific": "Full-service carrier with premium pricing",
                "Thai Airways": "Regional premium carrier with good value",
            }

            if airline in airline_insights:
                st.info(f"**{airline}**: {airline_insights[airline]}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")

    # Model Information
    st.markdown("---")
    st.markdown("## Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "Gradient Boosting")
        st.metric("Training Accuracy", "99.998%")

    with col2:
        st.metric("Average Error", "BDT 185")
        st.metric("Dataset Size", "57,000 flights")

    with col3:
        st.metric("Features Used", "74")
        st.metric("Last Updated", "Feb 2026")

    # Feature Importance Visualization
    st.markdown("### Key Factors Influencing Fare")
    try:
        st.image(
            f"{VISUALIZATIONS_PATH}/feature_importance_Gradient_Boosting.png",
            caption="Feature Importance - Gradient Boosting Model",
            width="content",
        )
    except Exception:
        st.info("Feature importance visualization will be displayed here.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | Flight Fare Prediction Project</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
