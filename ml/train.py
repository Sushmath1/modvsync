"""Train and evaluate the weekly MODIS -> VIIRS harmonization model.

Time-based split: train on ISO years 2018-2022, test on 2023-2024.
Baseline: VIIRS = MODIS count x per-region ratio (ratio from training data only).
Candidate models (the one with the lowest overall test MAE is kept):
- rf_count:   RandomForestRegressor predicting viirs_count directly.
- rf_ratio:   RandomForestRegressor predicting viirs_count / modis_count
              (trained on weeks with modis_count > 0); prediction = ratio x
              modis_count, and 0 when modis_count == 0.
- lgbm_count: LGBMRegressor predicting viirs_count (only if lightgbm is installed).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from ml.features import FEATURE_COLUMNS, TARGET  # noqa: E402

TRAIN_YEARS = list(range(2018, 2023))
TEST_YEARS = [2023, 2024]
RF_PARAMS = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
LGBM_PARAMS = {"n_estimators": 300, "random_state": 42, "verbose": -1}
TOP_N_IMPORTANCES = 10

PREDICTIONS_PATH = config.PROCESSED_DIR / "ml_predictions.parquet"
METRICS_PATH = config.PROCESSED_DIR / "ml_metrics.json"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "harmonizer.joblib"


def split_by_year(weekly):
    """Return (train, test) DataFrames using the fixed year split. No shuffling."""
    train = weekly[weekly["year"].isin(TRAIN_YEARS)]
    test = weekly[weekly["year"].isin(TEST_YEARS)]
    return train, test


def fit_baseline_ratios(train):
    """Per-region ratio = total VIIRS / total MODIS detections in the training years."""
    sums = train.groupby("region")[[TARGET, "modis_count"]].sum()
    return (sums[TARGET] / sums["modis_count"]).to_dict()


def predict_baseline(weekly, ratios):
    """Baseline prediction: MODIS count times the region's training ratio."""
    return weekly["modis_count"] * weekly["region"].map(ratios)


def make_random_forest():
    """Return an unfitted RandomForestRegressor with the project parameters."""
    return RandomForestRegressor(**RF_PARAMS)


def make_lightgbm():
    """Return an unfitted LGBMRegressor, or None if lightgbm is not installed."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return LGBMRegressor(**LGBM_PARAMS)


# --- Candidate models: each has a fit(train) and a predict(model, df) ----------

def fit_rf_count(train):
    """Random forest predicting the weekly VIIRS count directly."""
    return make_random_forest().fit(train[FEATURE_COLUMNS], train[TARGET])


def fit_rf_ratio(train):
    """Random forest predicting VIIRS / MODIS, trained on weeks with modis_count > 0."""
    rows = train[train["modis_count"] > 0]
    ratio = rows[TARGET] / rows["modis_count"]
    return make_random_forest().fit(rows[FEATURE_COLUMNS], ratio)


def fit_lgbm_count(train):
    """LightGBM predicting the weekly VIIRS count directly."""
    return make_lightgbm().fit(train[FEATURE_COLUMNS], train[TARGET])


def predict_count(model, df):
    """Predicted VIIRS count from a count model, clipped at 0."""
    return np.clip(model.predict(df[FEATURE_COLUMNS]), 0, None)


def predict_ratio(model, df):
    """Predicted VIIRS count = predicted ratio x modis_count (0 when modis_count is 0)."""
    pred = model.predict(df[FEATURE_COLUMNS]) * df["modis_count"].to_numpy()
    return np.clip(np.where(df["modis_count"] > 0, pred, 0.0), 0, None)


def available_candidates():
    """Return {name: (fit_fn, predict_fn, params)} for the candidates that can run."""
    cands = {
        "rf_count": (fit_rf_count, predict_count, RF_PARAMS),
        "rf_ratio": (fit_rf_ratio, predict_ratio, RF_PARAMS),
    }
    if make_lightgbm() is not None:
        cands["lgbm_count"] = (fit_lgbm_count, predict_count, LGBM_PARAMS)
    else:
        print("  lgbm_count: lightgbm not installed, skipped")
    return cands


# --- Metrics ------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R^2 as a plain dict of floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_by_region(test, pred):
    """Metrics for one prediction array on the test rows, overall and per region."""
    pred = pd.Series(np.asarray(pred), index=test.index)
    out = {"overall": regression_metrics(test[TARGET], pred)}
    for region, part in test.groupby("region"):
        out[region] = regression_metrics(part[TARGET], pred.loc[part.index])
    return out


def feature_importances(model):
    """Return feature importances sorted high to low, normalised to sum to 1."""
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS, dtype=float)
    return (imp / imp.sum()).sort_values(ascending=False)


def evaluate_candidates(train, test, ratios):
    """Fit every candidate on train and score it (and the baseline) on test."""
    results = {"baseline": {"metrics": metrics_by_region(test, predict_baseline(test, ratios))}}
    for name, (fit_fn, predict_fn, params) in available_candidates().items():
        model = fit_fn(train)
        metrics = metrics_by_region(test, predict_fn(model, test))
        print(f"  {name}: test MAE = {metrics['overall']['mae']:.1f}")
        results[name] = {"model": model, "predict": predict_fn, "params": params,
                         "metrics": metrics}
    return results


# --- Outputs ------------------------------------------------------------------

def build_predictions(weekly, model, predict_fn, ratios):
    """Prediction table for every week: actual, model, baseline and split label."""
    preds = weekly[["region", "year", "week", "week_start"]].copy()
    preds["actual_viirs"] = weekly[TARGET]
    preds["predicted_viirs"] = predict_fn(model, weekly)
    preds["baseline_viirs"] = predict_baseline(weekly, ratios)
    preds["split"] = np.where(weekly["year"].isin(TEST_YEARS), "test", "train")
    return preds


def save_outputs(preds, metrics, model_bundle):
    """Write predictions parquet, metrics JSON and the joblib model bundle."""
    preds.to_parquet(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"  wrote {PREDICTIONS_PATH.relative_to(config.ROOT)}, "
          f"{METRICS_PATH.relative_to(config.ROOT)}, {MODEL_PATH.relative_to(config.ROOT)}")


def main(weekly):
    """Train, evaluate and save. Returns (predictions DataFrame, metrics dict)."""
    print("[train] fitting models ...")
    train, test = split_by_year(weekly)
    ratios = fit_baseline_ratios(train)
    print("  baseline ratios (VIIRS/MODIS, train years): "
          + ", ".join(f"{r}={v:.3f}" for r, v in ratios.items()))

    results = evaluate_candidates(train, test, ratios)
    models = [n for n in results if n != "baseline"]
    best = min(models, key=lambda n: results[n]["metrics"]["overall"]["mae"])
    kept = results[best]
    print(f"  kept model: {best}")

    preds = build_predictions(weekly, kept["model"], kept["predict"], ratios)
    metrics = {
        "model_type": f"{type(kept['model']).__name__} ({best})",
        "model_name": best,
        "model_params": kept["params"],
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "n_train_weeks": int(len(train)),
        "n_test_weeks": int(len(test)),
        "features": FEATURE_COLUMNS,
        "baseline_ratios": {r: float(v) for r, v in ratios.items()},
        "test_metrics": {"baseline": results["baseline"]["metrics"],
                         "model": kept["metrics"]},
        "candidates_test_metrics": {n: r["metrics"] for n, r in results.items()},
        "top_feature_importances":
            feature_importances(kept["model"]).head(TOP_N_IMPORTANCES).round(4).to_dict(),
    }
    bundle = {"model": kept["model"], "model_name": best, "features": FEATURE_COLUMNS,
              "baseline_ratios": ratios, "train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
              "note": "rf_ratio predicts VIIRS/MODIS; multiply by modis_count (0 if modis_count == 0)"}
    save_outputs(preds, metrics, bundle)
    return preds, metrics


if __name__ == "__main__":
    from ml import features
    main(features.main())
