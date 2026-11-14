"""Build data/processed/detections.parquet from the raw FIRMS CSVs.

Run from the project root:  python scripts/build_processed.py
"""
import sys
from pathlib import Path

# Let this script import the 'src' package from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingest, clean, grid

FINAL_COLUMNS = ["sensor", "platform", "latitude", "longitude", "datetime_utc", "date",
                 "confidence_raw", "confidence_norm", "frp", "daynight", "region", "cell_id"]


def print_summary(df):
    """Print row counts per sensor, region and year."""
    print("\nRows per sensor:")
    print(df["sensor"].value_counts().to_string())
    print("\nRows per region:")
    print(df["region"].value_counts().to_string())
    print("\nRows per year and sensor:")
    year = df["datetime_utc"].dt.year.rename("year")
    print(df.groupby([year, "sensor"]).size().unstack(fill_value=0).to_string())


def main():
    df = ingest.load_all()
    print(f"Loaded {len(df):,} rows")

    print("Cleaning:")
    df = clean.clean(df)

    df = grid.add_grid(df)
    df = grid.add_date(df)
    df = df[FINAL_COLUMNS]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DETECTIONS_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {config.DETECTIONS_PATH.relative_to(config.ROOT)}")

    print_summary(df)


if __name__ == "__main__":
    main()
