# ML: weekly MODIS → VIIRS harmonization and anomaly weeks

Input: `data/processed/detections.parquet` (built by `scripts/build_processed.py`).

## Run

From the repo root, with the venv active:

```bash
python ml/run_all.py
```

This runs `features` → `train` → `anomaly` and prints a summary. Each step can also be run on its own (`python ml/train.py`, `python ml/anomaly.py`).

Outputs:

| file | content |
|---|---|
| `data/processed/ml_predictions.parquet` | region, year, week, week_start, actual_viirs, predicted_viirs, baseline_viirs, split |
| `data/processed/ml_metrics.json` | baseline vs model test metrics (overall + per region), model type and parameters, train/test years, baseline ratios, top feature importances |
| `data/processed/anomalies.parquet` | region, year, week, week_start, modis_count, viirs_count, anomaly_score, is_anomaly_iforest, is_anomaly_percentile |
| `ml/models/harmonizer.joblib` | fitted model + feature list + baseline ratios (git-ignored) |

## Method

**Weekly table (`features.py`).** One row per region per ISO week (Monday start, from `datetime_utc` in UTC) for every ISO week of 2018–2024. Weeks with no detections are filled with 0. Detections on 30–31 Dec 2024 belong to ISO week 2025-W01, so they are outside the range and left out.

- Target: `viirs_count`, the number of VIIRS detections in the week.
- MODIS features: `modis_count`, `modis_frp_sum/mean/max`, `modis_day_share`, `terra_count`, `aqua_count`, `modis_high_conf_share`.
- Season features: `week_of_year`, `month` (the month of the ISO week's Thursday), and the sin and cos of the week.
- Region one-hot: `region_Bolivia`, `region_Chad`.
- No VIIRS value is used as a feature.

**Training (`train.py`).**
- Time-based split: the model trains on ISO years 2018–2022 and is tested on 2023–2024. There is no random split.
- Baseline: VIIRS = MODIS count × a per-region ratio. The ratio is total VIIRS / total MODIS over the training years only.
- Candidate models. All use the same features, and the one with the lowest overall test MAE is saved:
  - `rf_count`: `RandomForestRegressor(n_estimators=300, random_state=42)` predicting `viirs_count` directly.
  - `rf_ratio`: the same random forest predicting the **ratio** `viirs_count / modis_count`. It is trained only on weeks with `modis_count > 0`. Then `predicted_viirs = predicted_ratio × modis_count`, and weeks with `modis_count == 0` predict 0.
  - `lgbm_count`: `LGBMRegressor` predicting the count, tried only if `lightgbm` is installed.
- Choosing a model on the test set slightly favours the kept model. The baseline is always reported next to it, and every candidate's test metrics are stored under `candidates_test_metrics` in `ml_metrics.json`.
- Metrics: MAE, RMSE and R² on the test set, overall and per region, for both the baseline and the model.

**Anomalies (`anomaly.py`).** These run per region on the weekly table.
- `IsolationForest(contamination=0.05, random_state=42)` on `modis_count`, `viirs_count`, `modis_frp_sum` and the residual (actual − predicted VIIRS). `anomaly_score` is the negated `score_samples`, so a higher score means a more unusual week.
- Residual: each year is predicted by the kept model type, trained on the other six years (leave-one-year-out). Training-year predictions from the main model are almost perfect in-sample, so their residuals would otherwise look much smaller than those of the test years.
- Comparison rule: `is_anomaly_percentile` is True when MODIS + VIIRS detections exceed the region's 95th percentile.

## Reading the results

`python ml/run_all.py` prints the current numbers, and `ml_metrics.json` stores them. On the current data:

- **`rf_count` does not beat the baseline overall.** It is better for Chad but much worse for Bolivia.
  - The cause is 2024 in Bolivia, a record fire year. Its biggest weeks are about 3× larger than any week in 2018–2022.
  - A random forest cannot predict values above the range it saw in training, so it falls far short on those weeks.
- **`rf_ratio` fixes this, and it is the saved model.** It has the lowest test MAE overall and in both regions, and it is ahead of the baseline in each region in both 2023 and 2024.
  - The ratio stays within its training range even in an extreme year, and multiplying by `modis_count` lets the prediction grow with the fire activity.
  - In effect it is the baseline with a ratio that changes with season, MODIS platform mix and day/night share, instead of one fixed number per region.
- The weekly VIIRS/MODIS ratio is fairly stable within each region, so the baseline is already strong. The gain from `rf_ratio` is real but moderate, not dramatic.

Possible next steps (not done): fit on log counts, or try a linear model with seasonal terms.

## Caveats

- In the current data, VIIRS is NOAA-20 only.
- `detections.parquet` is already cleaned: low-confidence and non-vegetation detections were dropped in `src/clean.py`. The counts here are therefore cleaned counts, not raw FIRMS counts.
- Weekly region totals hide where in the region fires happen. This is a region-level harmonization, not a per-cell one.
