"""Generate a FAKE data/processed/calendar.parquet that follows the CLAUDE.md section 5 contract.

Lets the app be built before the real calendar exists. Values are random but realistic.
Run from the project root:  python scripts/make_fake_calendar.py [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script import the 'src' package from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

CALENDAR_PATH = config.PROCESSED_DIR / "calendar.parquet"

SEED = 42
START_DATE, END_DATE = "2018-01-01", "2018-03-31"
N_CELLS = 20  # sub-area of N_CELLS x N_CELLS cells per region

# Lower-left corner of the fake sub-area in each region (must lie inside REGION_BBOX)
SUBAREA_ORIGIN = {
    "Chad":    (9.0, 17.0),     # southern Chad savanna
    "Bolivia": (-17.0, -63.0),  # Chiquitania, east of Santa Cruz
}

# Probabilities used to fake the data
P_NOT_OBSERVED = 0.15    # no satellite passed at all
P_FIRE_IF_OBSERVED = 0.08

REASONS = {
    "not_observed": "No satellite pass this day",
    "both": "Seen by both MODIS and VIIRS",
    "viirs_small": "Seen by VIIRS only: likely a small fire below MODIS resolution",
    "viirs_no_modis": "Seen by VIIRS only: no MODIS pass this day",
    "modis_missed": "Seen by MODIS only: VIIRS may have missed it (cloud or timing)",
    "modis_no_viirs": "Seen by MODIS only: no VIIRS pass this day",
}


def make_cells(region):
    """Return a DataFrame of the region's sub-area cells with cell_id and cell-centre lat/lon."""
    lat_origin, lon_origin = SUBAREA_ORIGIN[region]
    box = config.REGION_BBOX[region]
    size = config.GRID_SIZE_DEG
    assert box["lat_min"] <= lat_origin and lat_origin + N_CELLS * size <= box["lat_max"]
    assert box["lon_min"] <= lon_origin and lon_origin + N_CELLS * size <= box["lon_max"]

    i, j = np.meshgrid(np.arange(N_CELLS), np.arange(N_CELLS), indexing="ij")
    lat0 = (lat_origin + i.ravel() * size).round(4)
    lon0 = (lon_origin + j.ravel() * size).round(4)
    # Same cell_id format as src/grid.py: lower-left corner, one decimal
    cell_id = [f"{a:.1f}_{b:.1f}" for a, b in zip(lat0, lon0)]
    return pd.DataFrame({"region": region, "cell_id": cell_id,
                         "lat": (lat0 + size / 2).round(4), "lon": (lon0 + size / 2).round(4)})


def add_coverage(df, rng):
    """Fake modis_passed / viirs_passed: some days nobody passes, VIIRS passes more often than MODIS."""
    n = len(df)
    observed = rng.random(n) >= P_NOT_OBSERVED
    # Among observed cell-days: 70% both pass, 20% VIIRS only, 10% MODIS only
    kind = rng.choice(["both", "viirs", "modis"], size=n, p=[0.7, 0.2, 0.1])
    df["modis_passed"] = observed & (kind != "viirs")
    df["viirs_passed"] = observed & (kind != "modis")
    return df


def add_detections(df, rng):
    """Fake fires: clustered in some cells, only on days a sensor passed, with consistent seen_by."""
    n = len(df)
    # Some cells burn much more than others (mean weight = 1)
    cell_weight = rng.gamma(shape=0.8, scale=1 / 0.8, size=df["cell_id"].nunique())
    weight = pd.Series(cell_weight, index=df["cell_id"].unique()).loc[df["cell_id"]].to_numpy()
    observed = df["modis_passed"] | df["viirs_passed"]
    is_fire = observed.to_numpy() & (rng.random(n) < P_FIRE_IF_OBSERVED * weight)

    # Who saw the fire: when both passed, VIIRS (finer resolution) catches more
    choice = rng.choice(["both", "viirs", "modis"], size=n, p=[0.45, 0.40, 0.15])
    seen_by = np.where(df["modis_passed"] & df["viirs_passed"], choice,
                       np.where(df["viirs_passed"], "viirs", "modis"))
    seen_by = np.where(is_fire, seen_by, "none")

    modis_hit = np.isin(seen_by, ["both", "modis"])
    viirs_hit = np.isin(seen_by, ["both", "viirs"])
    df["modis_count"] = np.where(modis_hit, rng.poisson(1.0, n) + 1, 0).astype("int64")
    df["viirs_count"] = np.where(viirs_hit, rng.poisson(2.5, n) + 1, 0).astype("int64")
    df["seen_by"] = seen_by
    df["max_frp"] = np.where(is_fire, rng.lognormal(mean=2.5, sigma=0.9, size=n).round(1), np.nan)
    return df


def add_state_and_reason(df):
    """Apply the section 5 state rules and attach a short reason string."""
    has_fire = (df["modis_count"] + df["viirs_count"]) > 0
    passed = df["modis_passed"] | df["viirs_passed"]
    df["state"] = np.select([has_fire, passed], ["fire", "clear"], default="not_observed")

    conditions = [
        df["state"] == "not_observed",
        df["seen_by"] == "both",
        (df["seen_by"] == "viirs") & df["modis_passed"],
        df["seen_by"] == "viirs",
        (df["seen_by"] == "modis") & df["viirs_passed"],
        df["seen_by"] == "modis",
    ]
    choices = [REASONS[k] for k in ["not_observed", "both", "viirs_small", "viirs_no_modis",
                                    "modis_missed", "modis_no_viirs"]]
    df["reason"] = np.select(conditions, choices, default="")  # clear days have no reason
    return df


def make_region(region, dates, rng):
    """Build the full cell x date calendar for one region."""
    cells = make_cells(region)
    df = cells.merge(pd.DataFrame({"date": dates}), how="cross")
    df = add_coverage(df, rng)
    df = add_detections(df, rng)
    df = add_state_and_reason(df)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing calendar.parquet")
    args = parser.parse_args()

    if CALENDAR_PATH.exists() and not args.force:
        print(f"{CALENDAR_PATH.relative_to(config.ROOT)} already exists (it may be the real one).")
        print("Not overwriting. Re-run with --force to replace it with fake data.")
        return

    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    df = pd.concat([make_region(r, dates, rng) for r in SUBAREA_ORIGIN], ignore_index=True)

    columns = ["region", "cell_id", "lat", "lon", "date", "modis_count", "viirs_count", "seen_by",
               "max_frp", "modis_passed", "viirs_passed", "state", "reason"]
    df = df[columns]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CALENDAR_PATH, index=False)
    print(f"Saved FAKE calendar: {len(df):,} rows to {CALENDAR_PATH.relative_to(config.ROOT)}")

    print("\nRows per region:")
    print(df["region"].value_counts().to_string())
    print("\nCounts per state:")
    counts = df["state"].value_counts()
    print(pd.DataFrame({"rows": counts, "share": (counts / len(df)).round(3)}).to_string())


if __name__ == "__main__":
    main()
