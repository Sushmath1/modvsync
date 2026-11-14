"""Load raw FIRMS CSVs and put MODIS and VIIRS into one common format."""
import pandas as pd

from src import config

# Columns we read from each CSV (the rest are sensor-specific and not needed)
USECOLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite",
           "confidence", "frp", "daynight", "type"]


def parse_filename(path):
    """Get sensor and region from names like 'modis_2018_Chad.csv' or 'viirs-jpss1_2018_Chad.csv'."""
    parts = path.stem.split("_")
    sensor = parts[0].split("-")[0].upper()  # 'modis' -> MODIS, 'viirs-jpss1' -> VIIRS
    region = parts[-1]
    return sensor, region


def load_csv(path):
    """Read one CSV and tag it with sensor, platform and region."""
    # acq_time and confidence are read as text so '0035' and 'n' stay intact
    df = pd.read_csv(path, usecols=USECOLS, dtype={"acq_time": str, "confidence": str})
    sensor, region = parse_filename(path)
    df["sensor"] = sensor
    df["region"] = region
    # Platform comes from the 'satellite' column (MODIS files mix Terra and Aqua)
    df["platform"] = df["satellite"].map(config.PLATFORM_NAMES).fillna(df["satellite"])
    return df


def build_datetime(df):
    """Combine acq_date (YYYY-MM-DD) and acq_time (HHMM, UTC) into datetime_utc."""
    hhmm = df["acq_time"].str.zfill(4)
    df["datetime_utc"] = pd.to_datetime(df["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
    return df


def unify_confidence(df):
    """Keep the original value in confidence_raw and add a shared class in confidence_norm.

    MODIS gives 0-100, VIIRS gives l/n/h. Both become 'low' / 'nominal' / 'high'.
    """
    df["confidence_raw"] = df["confidence"]
    is_modis = df["sensor"] == "MODIS"

    # MODIS: bin the 0-100 number into classes
    modis_num = pd.to_numeric(df["confidence"].where(is_modis), errors="coerce")
    modis_cls = pd.cut(modis_num, bins=config.MODIS_CONF_BINS, labels=config.CONF_CLASSES)

    # VIIRS: map the l/n/h letters to the same classes
    viirs_cls = df["confidence"].where(~is_modis).str.lower().map(config.VIIRS_CONF_MAP)

    norm = modis_cls.astype(object).where(is_modis, viirs_cls)
    df["confidence_norm"] = pd.Categorical(norm, categories=config.CONF_CLASSES, ordered=True)
    return df


def load_all():
    """Load every CSV under the MODIS and VIIRS raw folders into one DataFrame."""
    files = sorted(config.RAW_MODIS_DIR.rglob("*.csv")) + sorted(config.RAW_VIIRS_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSVs found in data/raw/modis or data/raw/viirs")
    print(f"Loading {len(files)} CSV files...")

    df = pd.concat([load_csv(f) for f in files], ignore_index=True)
    df = build_datetime(df)
    df = unify_confidence(df)
    return df
