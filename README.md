# Flight Fare Prediction

A production-style machine learning project for predicting flight fares in Bangladesh using a time-aware regression pipeline.


## Project overview

### Business problem
Estimate flight fares from observable booking/flight attributes to support pricing analysis and user-facing fare estimation.

### ML task
- Type: Supervised regression
- Target: `total_fare_bdt`
- Modeling features:
  - Categorical: `airline`, `source`, `destination`, `stopovers`, `aircraft_type`, `class`, `booking_source`, `seasonality`, `route`
  - Numerical: `duration_hrs`, `days_before_departure`, `departure_month`, `departure_day`, `departure_hour`, `departure_weekday`, `is_weekend`, `is_peak_hour`, `route_frequency`

## Dataset

- Source file: `data/Flight_Price_Dataset_of_Bangladesh.csv`
- Rows: 57,000
- Original columns: 17


## Pipeline flow

Run end-to-end:

```bash
python pipeline/main.py
```

Optional flags:

```bash
python pipeline/main.py --test-size 0.2
python pipeline/main.py --no-tuning
```

Pipeline stages:

1. Load raw CSV and normalize column names
2. Canonical preprocessing + feature engineering
3. EDA visualizations and KPIs
4. Baseline model training with time-aware CV
5. Tune best tree candidate with `RandomizedSearchCV`
6. Evaluate holdout test set (R2, MAE, RMSE)
7. Save model/artifacts/reports

## Main artifacts

- Best pipeline: `models/best_model_pipeline.pkl`
- Metadata/schema: `models/model_metadata.json`
- Holdout metrics: `reports/model_metrics.csv`
- CV summary: `reports/cv_results.csv`
- Interpretation report: `reports/model_interpretation_and_insights.md`
- Feature-importance plot: `visualizations/feature_importance_top15.png`

## Current model results

From `reports/model_metrics.csv`:

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| HistGradientBoosting_tuned | 0.6780 | 30,618.47 | 50,605.68 |
| HistGradientBoosting | 0.6756 | 30,707.98 | 50,787.26 |
| RandomForest | 0.6519 | 32,200.92 | 52,612.18 |
| Lasso | 0.5700 | 42,812.36 | 58,474.33 |
| Ridge | 0.5699 | 42,808.50 | 58,480.43 |

Best model: `HistGradientBoosting_tuned`

## Inference API

`src/inference.py` exposes:

- `predict(payload: dict) -> dict`
  - Returns `predicted_total_fare_bdt`
  - Returns `prediction_std_bdt` only if estimator-level uncertainty is available

The inference path reuses the same preprocessing logic as training to avoid train/serve skew.

## Streamlit app

Launch:

```bash
streamlit run app/app.py
```

UI now uses inputs:

- `duration_hrs`, `days_before_departure`
- `departure_date`, `departure_time`
- `airline`, `source`, `destination`, `stopovers`, `aircraft_type`, `class`, `booking_source`, `seasonality`

Outputs:

- Predicted Total Fare (BDT)
- Simple heuristics-based insights
- Top 15 feature drivers

## Repository structure

```text
flight_fare_prediction/
|-- app/
|   `-- app.py
|-- data/
|   `-- Flight_Price_Dataset_of_Bangladesh.csv
|-- logs/
|   `-- pipeline.log
|-- models/
|   |-- best_model_pipeline.pkl
|   `-- model_metadata.json
|-- pipeline/
|   `-- main.py
|-- reports/
|   |-- cv_results.csv
|   |-- model_interpretation_and_insights.md
|   `-- model_metrics.csv
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- data_preprocessing.py
|   |-- eda.py
|   |-- inference.py
|   |-- modeling.py
|   `-- utils.py
`-- visualizations/
    |-- correlation_heatmap.png
    |-- fare_by_airline.png
    |-- fare_by_month.png
    |-- fare_distribution.png
    |-- fare_components_vs_total.png
    `-- feature_importance_top15.png
```

## Notes

- Logs are written to `logs/pipeline.log`
- Time-aware evaluation is intentionally stricter than random split and may yield lower but more realistic scores
- Component plots in EDA are for business explanation only, not for predictor selection
