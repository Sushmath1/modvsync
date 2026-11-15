"""Build the weekly per-region table used by the ML models.

One row per (region, ISO year, ISO week) for every ISO week of 2018-2024,
including weeks with no detections (filled with 0). The target is the weekly
VIIRS detection count; all features come from MODIS or the calendar only.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

FIRST_YEAR = 2018
LAST_YEAR = 2024
REGIONS = sorted(config.REGION_BBOX)  # ["Bolivia", "Chad"]

# Features the model may use. No VIIRS values here: that would leak the target.
MODIS_FEATURES = [
    "modis_count", "modis_frp_sum", "modis_frp_mean", "modis_frp_max",
    "modis_day_share", "terra_count", "aqua_count", "modis_high_conf_share",
]
SEASON_FEATURES = ["week_of_year", "month", "week_sin", "week_cos"]
REGION_FEATURES = [f"region_{r}" for r in REGIONS]
FEATURE_COLUMNS = MODIS_FEATURES + SEASON_FEATURES + REGION_FEATURES
TARGET = "viirs_count"


def load_detections(path=config.DETECTIONS_PATH):
    """Read detections.parquet with only the columns the ML step needs."""
    cols = ["sensor", "platform", "datetime_utc", "confidence_norm", "frp",
            "daynight", "region"]
    return pd.read_parquet(path, columns=cols)


def add_iso_week(df):
    """Add ISO `year` and `week` columns (from datetime_utc, UTC) and keep 2018-2024."""
    df = df.copy()
    iso = df["datetime_utc"].dt.isocalendar()
    df["year"] = iso["year"].astype(int)
    df["week"] = iso["week"].astype(int)
    in_range = df["year"].between(FIRST_YEAR, LAST_YEAR)
    dropped = int((~in_range).sum())
    if dropped:
        print(f"  note: {dropped} detections fall in ISO years outside "
              f"{FIRST_YEAR}-{LAST_YEAR} and are left out")
    return df[in_range]


def full_week_index():
    """Return every (region, year, week, week_start) for ISO years 2018-2024."""
    rows = []
    for region in REGIONS:
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            n_weeks = date(year, 12, 28).isocalendar()[1]  # 52 or 53
            for week in range(1, n_weeks + 1):
                rows.append((region, year, week, date.fromisocalendar(year, week, 1)))
    return pd.DataFrame(rows, columns=["region", "year", "week", "week_start"])


def weekly_modis_features(df):
    """Aggregate MODIS detections to one row per (region, year, week)."""
    m = df[df["sensor"] == "MODIS"].assign(
        is_day=lambda d: (d["daynight"] == "D").astype(int),
        is_high=lambda d: (d["confidence_norm"] == "high").astype(int),
        is_terra=lambda d: (d["platform"] == "Terra").astype(int),
        is_aqua=lambda d: (d["platform"] == "Aqua").astype(int),
    )
    g = m.groupby(["region", "year", "week"])
    out = pd.DataFrame({
        "modis_count": g.size(),
        "modis_frp_sum": g["frp"].sum(),
        "modis_frp_mean": g["frp"].mean(),
        "modis_frp_max": g["frp"].max(),
        "modis_day_share": g["is_day"].mean(),
        "terra_count": g["is_terra"].sum(),
        "aqua_count": g["is_aqua"].sum(),
        "modis_high_conf_share": g["is_high"].mean(),
    })
    return out.reset_index()


def weekly_viirs_target(df):
    """Count VIIRS detections per (region, year, week): the prediction target."""
    v = df[df["sensor"] == "VIIRS"]
    return v.groupby(["region", "year", "week"]).size().rename(TARGET).reset_index()


def add_season_features(weekly):
    """Add week_of_year, month (of the ISO week's Thursday) and sin/cos of the week."""
    weekly = weekly.copy()
    weekly["week_of_year"] = weekly["week"]
    thursday = pd.to_datetime(weekly["week_start"]) + timedelta(days=3)
    weekly["month"] = thursday.dt.month
    angle = 2 * np.pi * weekly["week"] / 52.18  # mean ISO weeks per year
    weekly["week_sin"] = np.sin(angle)
    weekly["week_cos"] = np.cos(angle)
    return weekly


def add_region_onehot(weekly):
    """Add one 0/1 column per region (region_Bolivia, region_Chad)."""
    weekly = weekly.copy()
    for region in REGIONS:
        weekly[f"region_{region}"] = (weekly["region"] == region).astype(int)
    return weekly


def build_weekly_table(df):
    """Join MODIS features and VIIRS target onto the full week grid; fill gaps with 0."""
    df = add_iso_week(df)
    weekly = (full_week_index()
              .merge(weekly_modis_features(df), on=["region", "year", "week"], how="left")
              .merge(weekly_viirs_target(df), on=["region", "year", "week"], how="left"))
    fill_cols = MODIS_FEATURES + [TARGET]
    weekly[fill_cols] = weekly[fill_cols].fillna(0)
    for col in ["modis_count", "terra_count", "aqua_count", TARGET]:
        weekly[col] = weekly[col].astype(int)
    weekly = add_region_onehot(add_season_features(weekly))
    return weekly.sort_values(["region", "year", "week"]).reset_index(drop=True)


def main():
    """Build and return the weekly table, printing a short summary."""
    print("[features] loading detections ...")
    weekly = build_weekly_table(load_detections())
    empty = (weekly["modis_count"] + weekly[TARGET] == 0).sum()
    print(f"  weekly table: {len(weekly)} rows "
          f"({weekly.groupby('region').size().to_dict()}), {empty} weeks with no detections")
    return weekly


if __name__ == "__main__":
    main()
