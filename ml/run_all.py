"""Run the full ML pipeline: features -> train -> anomaly, then print a summary.

Usage (from the repo root):  python ml/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import anomaly, features, train  # noqa: E402


def print_model_summary(metrics):
    """Print test MAE / R2 of the baseline and every candidate, overall and per region."""
    cands = metrics["candidates_test_metrics"]
    years = f"{metrics['test_years'][0]}-{metrics['test_years'][-1]}"
    print(f"\nTest set ({years}), weekly VIIRS detections. Kept model: {metrics['model_type']}")
    print(f"  {'scope':<10}" + "".join(f"{n + ' MAE':>16}" for n in cands)
          + "".join(f"{n + ' R2':>15}" for n in cands) + "  best MAE")
    for scope in cands["baseline"]:
        maes = {n: c[scope]["mae"] for n, c in cands.items()}
        print(f"  {scope:<10}" + "".join(f"{v:>16.1f}" for v in maes.values())
              + "".join(f"{c[scope]['r2']:>15.3f}" for c in cands.values())
              + f"  {min(maes, key=maes.get)}")
    kept_mae = metrics["test_metrics"]["model"]["overall"]["mae"]
    base_mae = metrics["test_metrics"]["baseline"]["overall"]["mae"]
    verdict = "beats" if kept_mae < base_mae else "does NOT beat"
    print(f"  -> Kept model ({metrics['model_name']}) {verdict} the ratio baseline "
          f"overall on the test set ({kept_mae:.1f} vs {base_mae:.1f} MAE).")
    print("  top features: " + ", ".join(
        f"{k} ({v:.2f})" for k, v in list(metrics["top_feature_importances"].items())[:5]))


def print_anomaly_summary(anomalies):
    """Print the number of anomaly weeks per region for both methods."""
    counts = anomalies.groupby("region")[["is_anomaly_iforest", "is_anomaly_percentile"]].sum()
    both = (anomalies["is_anomaly_iforest"] & anomalies["is_anomaly_percentile"])
    counts["both"] = both.groupby(anomalies["region"]).sum()
    counts["weeks"] = anomalies.groupby("region").size()
    print("\nAnomaly weeks per region")
    print(counts.rename(columns={"is_anomaly_iforest": "iforest",
                                 "is_anomaly_percentile": "p95_rule"}).to_string())


def main():
    """Run every step in order and print the summary."""
    weekly = features.main()
    _, metrics = train.main(weekly)
    anomalies = anomaly.main(weekly, metrics["model_name"])
    print_model_summary(metrics)
    print_anomaly_summary(anomalies)


if __name__ == "__main__":
    main()
