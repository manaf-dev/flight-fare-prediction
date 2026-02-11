"""
Inference module for Flight Fare Prediction.
Handles loading preprocessors and making predictions with the trained model.
"""

import os

import joblib
import pandas as pd

from src.config import MODELS_PATH
from src.data_preprocessing import encode_and_scale


def load_preprocessors():
    """
    Load the saved scaler and encoders.

    Returns:
        tuple: (scaler, encoders_dict)
    """
    try:
        scaler_path = os.path.join(MODELS_PATH, 'scaler.pkl')
        encoders_path = os.path.join(MODELS_PATH, 'encoders.pkl')

        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)

        return scaler, encoders
    except Exception as e:
        raise Exception(f"Error loading preprocessors: {e}")

def preprocess_input(input_data, scaler, encoders):
    """
    Preprocess input data for prediction.

    Args:
        input_data (dict): Raw input data from user
        scaler: Trained StandardScaler
        encoders (dict): Trained encoders

    Returns:
        pd.DataFrame: Processed input ready for prediction
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Feature engineering (similar to training)
        df = feature_engineering(df)

        # Encode and scale
        df_processed, _ = encode_and_scale(df)

        return df_processed

    except Exception as e:
        raise Exception(f"Error preprocessing input: {e}")

def predict_fare(input_data):
    """
    Make fare prediction for given input data.

    Args:
        input_data (dict): Flight details

    Returns:
        float: Predicted fare
    """
    try:
        # Load model and preprocessors
        model_path = os.path.join(MODELS_PATH, 'Gradient_Boosting.pkl')
        model = joblib.load(model_path)
        scaler, encoders = load_preprocessors()

        # Preprocess input
        processed_input = preprocess_input(input_data, scaler, encoders)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        return prediction

    except Exception as e:
        raise Exception(f"Error making prediction: {e}")

def feature_engineering(df):
    """
    Apply feature engineering to input data.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()

    # Convert date to datetime
    df['departure_date_and_time'] = pd.to_datetime(df['departure_date_and_time'])

    # Extract date features
    df['departure_month'] = df['departure_date_and_time'].dt.month
    df['departure_day'] = df['departure_date_and_time'].dt.day
    df['departure_hour'] = df['departure_date_and_time'].dt.hour
    df['departure_weekday'] = df['departure_date_and_time'].dt.weekday

    # Create season feature
    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'

    df['season'] = df['departure_month'].apply(get_season)

    # Calculate total fare (if not provided)
    if 'total_fare_bdt' not in df.columns:
        df['total_fare_bdt'] = df['base_fare_bdt'] + df['tax_and_surcharge_bdt']

    return df
