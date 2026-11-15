"""Flag unusual fire weeks per region.

Two methods, run separately for each region:
- IsolationForest on modis_count, viirs_count, modis_frp_sum and the model
  residual (actual - predicted VIIRS).
- A simple rule: combined MODIS + VIIRS detections above the region's
  95th percentile.

The residual uses leave-one-year-out predictions from the kept model type
(each year predicted by a model trained on the other years). In-sample random-forest predictions on the
training years would be close to perfect, so their residuals would look
artificially small next to the test years.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from ml.features import TARGET  # noqa: E402
from ml.train import METRICS_PATH, available_candidates  # noqa: E402

CONTAMINATION = 0.05
RANDOM_STATE = 42
PERCENTILE = 0.95
IFOREST_COLUMNS = ["modis_count", "viirs_count", "modis_frp_sum", "residual"]
ANOMALIES_PATH = config.PROCESSED_DIR / "anomalies.parquet"


def kept_model_name():
    """Name of the model kept by train.py (from ml_metrics.json), default rf_count."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text()).get("model_name", "rf_count")
    return "rf_count"


def leave_one_year_out_residuals(weekly, model_name):
    """Actual minus predicted VIIRS, each year predicted by a model fit on the other years."""
    fit_fn, predict_fn, _ = available_candidates()[model_name]
    pred = pd.Series(0.0, index=weekly.index)
    for year in sorted(weekly["year"].unique()):
        held_out = weekly["year"] == year
        model = fit_fn(weekly[~held_out])
        pred[held_out] = predict_fn(model, weekly[held_out])
    return weekly[TARGET] - pred


def isolation_forest_flags(region_df):
    """Return (anomaly_score, is_anomaly) for one region. Higher score = more unusual."""
    forest = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    X = region_df[IFOREST_COLUMNS]
    is_anomaly = forest.fit_predict(X) == -1
    score = -forest.score_samples(X)
    return score, is_anomaly


def percentile_flags(region_df):
    """True where MODIS + VIIRS detections exceed the region's 95th percentile."""
    combined = region_df["modis_count"] + region_df["viirs_count"]
    return combined > combined.quantile(PERCENTILE)


def detect_anomalies(weekly, model_name):
    """Run both methods per region and return the anomalies table."""
    df = weekly.assign(residual=leave_one_year_out_residuals(weekly, model_name))
    parts = []
    for _, region_df in df.groupby("region"):
        region_df = region_df.copy()
        score, flag = isolation_forest_flags(region_df)
        region_df["anomaly_score"] = score
        region_df["is_anomaly_iforest"] = flag
        region_df["is_anomaly_percentile"] = percentile_flags(region_df)
        parts.append(region_df)
    cols = ["region", "year", "week", "week_start", "modis_count", "viirs_count",
            "anomaly_score", "is_anomaly_iforest", "is_anomaly_percentile"]
    return pd.concat(parts)[cols].reset_index(drop=True)


def main(weekly, model_name=None):
    """Detect anomalies, save anomalies.parquet and return the table."""
    model_name = model_name or kept_model_name()
    print(f"[anomaly] running IsolationForest (residual from {model_name}) "
          "and percentile rule ...")
    anomalies = detect_anomalies(weekly, model_name)
    anomalies.to_parquet(ANOMALIES_PATH, index=False)
    print(f"  wrote {ANOMALIES_PATH.relative_to(config.ROOT)}")
    return anomalies


if __name__ == "__main__":
    from ml import features
    main(features.main())
