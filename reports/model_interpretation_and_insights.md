# Model Interpretation and Insights

## Run Context

- Problem: Predict `total_fare_bdt` from booking, route, airline, and time features
- Best model selected: `HistGradientBoosting_tuned`
- Evaluation setup:
  - Chronological holdout split (80% train / 20% test)
  - Time-series cross-validation (5 folds)
  - Metrics: R2, MAE, RMSE

## 1. Model Performance Summary

### Holdout test set results

| Model | R2 | MAE (BDT) | RMSE (BDT) |
|---|---:|---:|---:|
| HistGradientBoosting_tuned | 0.6759 | 30,674 | 50,766 |
| HistGradientBoosting | 0.6756 | 30,617 | 50,793 |
| RandomForest | 0.6521 | 32,260 | 52,599 |
| LinearRegression (baseline) | 0.5695 | 42,940 | 58,510 |
| Lasso | 0.5695 | 42,959 | 58,510 |
| Ridge | 0.5695 | 42,941 | 58,511 |
| DecisionTree | 0.5189 | 35,270 | 61,853 |

### What this means

- `HistGradientBoosting_tuned` is the best overall model by R2 and RMSE.
- Compared with baseline Linear Regression:
  - R2 improved by **+0.1064** (0.5695 -> 0.6759)
  - MAE improved by about **12,266 BDT** (42,940 -> 30,674)
  - RMSE improved by about **7,744 BDT** (58,510 -> 50,766)

This is a meaningful accuracy gain, not just a marginal improvement.

## 2. Why This Model Won

From `reports/cv_results.csv`:

- `HistGradientBoosting_tuned` CV R2 = **0.6694**
- `HistGradientBoosting` CV R2 = **0.6672**
- `RandomForest` CV R2 = **0.6478**

The tuned gradient boosting model performs best in both CV and holdout, which indicates better generalization than the alternatives for this dataset.

## 3. What the Diagnostics Show

### Actual vs Predicted (`visualizations/actual_vs_predicted.png`)

- Predictions follow the overall upward direction, but points are widely dispersed.
- The model compresses many high-fare cases into lower predicted bands.
- Practical implication: the model is directionally useful but less precise for expensive tickets.

### Residual Analysis (`visualizations/residual_analysis.png`)

- Residual spread increases at higher predicted fares (heteroscedasticity).
- Residual distribution is centered near zero but with wide tails.
- Practical implication: average performance is good, but error risk grows for high-fare segments.

### Bias-Variance Tradeoff (`visualizations/bias_variance_tradeoff.png`)

- Tree ensembles clearly outperform linear models in CV R2.
- Error bars for the best-performing models are moderate and stable.
- Practical implication: the winning model balances fit and generalization better than simpler baselines.

## 4. Fare Behavior Insights From EDA

### Distribution patterns

- `visualizations/fare_distribution_total.png`: Total fares are strongly right-skewed; most fares are low-to-mid, with a long high-price tail.
- `visualizations/fare_distribution_base.png` and `visualizations/fare_distribution_tax.png`: both components are right-skewed and contribute to occasional high total fares.

### Seasonality (`visualizations/fare_by_season_boxplot.png`)

- Median and upper-range fares are higher in peak periods, especially **Hajj**.
- **Regular** season shows lower central tendency than peak seasons.
- Practical implication: seasonality is a pricing lever and should be retained in pricing/forecast workflows.

### Airline variation (`visualizations/fare_by_airline.png`)

- Airlines show different fare distributions and spread, indicating distinct pricing strategies.
- Practical implication: airline-specific behavior should remain a core predictor.

## 5. What Drives Fares (Model Interpretation)

For this run, the best model (`HistGradientBoosting_tuned`) did not produce a native feature-importance vector in the saved output, so interpretation uses the linear coefficient reports:

- `reports/linear_coefficients_linearregression.csv`
- `reports/linear_coefficients_ridge.csv`
- `reports/linear_coefficients_lasso.csv`

Consistent high-impact signals across linear models:

1. **Cabin class**  
   - `class_First Class` strongly increases fares  
   - `class_Economy` strongly decreases fares relative to baseline categories

2. **Destination / route effects**  
   - Some destinations (e.g., `destination_CCU`) have large positive fare impact
   - Route-specific terms in Lasso highlight expensive corridors

3. **Aircraft and airline attributes**  
   - Aircraft-type terms carry large coefficients, indicating equipment/operational pricing effects

4. **Booking lead time**  
   - `days_before_departure` is negative in Lasso top coefficients, supporting the business pattern that earlier booking is generally cheaper

## 6. Business Recommendations

1. Use `HistGradientBoosting_tuned` as the default pricing prediction model.  
Reason: best holdout R2 and lowest RMSE among tested models.

2. Add confidence bands or caution flags for expensive itineraries (high-fare range).  
Reason: residual spread increases for higher predicted fares.

3. Include season-aware pricing policy controls, especially for Hajj and holiday periods.  
Reason: season boxplots show higher medians and wider upper ranges in peak seasons.

4. Keep airline, class, and route-level segmentation in reporting and forecasting.  
Reason: both EDA and coefficients show these variables are strong price differentiators.

5. Monitor drift monthly using the same chronological test strategy.  
Reason: stable offline performance today does not guarantee future seasonal stability.

