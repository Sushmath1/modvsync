"""Remove bad, duplicate and unwanted detections."""
from src import config


def drop_invalid_coords(df):
    """Drop rows with missing/out-of-range coordinates or outside their region's bounding box."""
    ok = df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    for region, box in config.REGION_BBOX.items():
        in_region = df["region"] == region
        in_box = (df["latitude"].between(box["lat_min"], box["lat_max"])
                  & df["longitude"].between(box["lon_min"], box["lon_max"]))
        ok &= ~in_region | in_box
    return df[ok]


def drop_duplicates(df):
    """Same sensor + platform + location + time counts as one detection."""
    return df.drop_duplicates(subset=["sensor", "platform", "latitude", "longitude", "datetime_utc"])


def filter_confidence(df):
    """Drop low-confidence detections (and any we could not classify)."""
    return df[df["confidence_norm"].notna() & ~df["confidence_norm"].isin(config.DROP_CONFIDENCE)]


def filter_type(df):
    """Keep only presumed vegetation fires, if the 'type' column exists."""
    if "type" in df.columns:
        df = df[df["type"].isin(config.KEEP_TYPES)]
    return df


def clean(df):
    """Run all cleaning steps and print how many rows each one removes."""
    steps = [drop_invalid_coords, drop_duplicates, filter_confidence, filter_type]
    for step in steps:
        before = len(df)
        df = step(df)
        print(f"  {step.__name__:<20} removed {before - len(df):>9,} rows")
    return df.reset_index(drop=True)
