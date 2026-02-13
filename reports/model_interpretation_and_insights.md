# Model Interpretation & Insights

## Executive Summary

A leakage-free, time-aware modeling workflow was implemented for flight fare prediction.

- Target: `total_fare_bdt`
- Leakage predictors excluded from modeling: `base_fare_bdt`, `tax_and_surcharge_bdt`
- Chronological holdout: first 80% train, last 20% test
- Cross-validation: `TimeSeriesSplit(n_splits=5)` on training window
- Best model: `HistGradientBoosting_tuned`

## Why leakage mattered

In this dataset, `total_fare_bdt` is largely composed of base fare plus surcharge components. Using those components as predictors creates target leakage and can make metrics look unrealistically perfect.

To fix this:

- Base and surcharge were removed from predictors.
- They are only allowed for optional target filling when target is missing.
- Final feature matrix is asserted to exclude leakage columns and target column.

## Modeling setup

### Candidate models

- Ridge
- Lasso
- RandomForestRegressor
- HistGradientBoostingRegressor

### Preprocessing and features

- Column normalization and canonical mapping
- Missing value handling:
  - Categorical -> `Unknown`
  - Numerical -> median
- Engineered features:
  - `departure_month`, `departure_day`, `departure_hour`, `departure_weekday`
  - `is_weekend`, `is_peak_hour`
  - `route`, `route_frequency`

### Validation strategy

- Holdout split is chronological by `departure_datetime`
- CV uses `TimeSeriesSplit` to avoid future leakage across folds
- Metrics: `R2`, `MAE`, `RMSE`

## Final model comparison (holdout)

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| HistGradientBoosting_tuned | 0.6780 | 30,618.47 | 50,605.68 |
| HistGradientBoosting | 0.6756 | 30,707.98 | 50,787.26 |
| RandomForest | 0.6519 | 32,200.92 | 52,612.18 |
| Lasso | 0.5700 | 42,812.36 | 58,474.33 |
| Ridge | 0.5699 | 42,808.50 | 58,480.43 |

Best model selection rationale:

- Highest holdout R2 under leakage-free and time-aware evaluation
- Stable performance relative to untuned baseline

## Cross-validation summary

| Model | CV R2 mean | CV MAE mean | CV RMSE mean |
|---|---:|---:|---:|
| HistGradientBoosting_tuned | 0.6694 | - | - |
| HistGradientBoosting | 0.6675 | 27,512.66 | 45,466.88 |
| RandomForest | 0.6504 | 28,899.33 | 46,631.16 |
| Ridge | 0.5624 | 40,036.06 | 52,158.38 |
| Lasso | 0.5605 | 40,106.36 | 52,279.11 |

## Top model drivers

Top 10 drivers from `models/model_metadata.json`:

1. `aircraft_type`
2. `class`
3. `destination`
4. `days_before_departure`
5. `seasonality`
6. `stopovers`
7. `route_frequency`
8. `airline`
9. `departure_month`
10. `departure_weekday`

Interpretation notes:

- `days_before_departure` confirms booking timing impact.
- `seasonality`, route/destination, and stopovers capture demand and itinerary effects.
- Cabin class and airline/aircraft variables encode service tier and pricing strategy.

## Inference and deployment alignment

- A single artifact (`models/best_model_pipeline.pkl`) is used for serving.
- Inference reuses shared preprocessing logic to maintain feature compatibility.
- Expected inference schema and training timestamp are tracked in `models/model_metadata.json`.

## EDA and reporting perspective

- Component plots (base + tax vs total) are retained for business explanation only.
- Modeling excludes these components to preserve correctness.

## Practical implications

- The updated pipeline provides realistic out-of-time performance estimates.
- Streamlit now accepts only leakage-free inputs.
- Reported metrics are suitable for comparing future model iterations consistently.
