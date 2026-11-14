"""Load calendar.parquet (or raw detections as a fallback) and derive per-sensor views."""
import numpy as np
import pandas as pd
import streamlit as st

from src import config

CALENDAR_PATH = config.PROCESSED_DIR / "calendar.parquet"

VIEWS = ["Harmonized", "MODIS only", "VIIRS only"]

# Only the detection columns the fallback map needs (keeps memory down)
DETECTION_COLUMNS = ["sensor", "platform", "latitude", "longitude", "datetime_utc",
                     "date", "confidence_norm", "frp", "region"]


@st.cache_data
def load_calendar():
    """Read calendar.parquet, or return None if it doesn't exist yet."""
    if not CALENDAR_PATH.exists():
        return None
    df = pd.read_parquet(CALENDAR_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@st.cache_data
def load_detections():
    """Read detections.parquet (one row per raw detection), or return None if missing."""
    if not config.DETECTIONS_PATH.exists():
        return None
    df = pd.read_parquet(config.DETECTIONS_PATH, columns=DETECTION_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def apply_view(df, view):
    """Return a copy of df whose 'state' reflects the chosen sensor view.

    Harmonized keeps the contract's state. 'MODIS only' / 'VIIRS only' recompute it from
    that sensor alone: fire if its count > 0, else clear if it passed, else not_observed.
    """
    if view == "Harmonized":
        return df
    prefix = "modis" if view == "MODIS only" else "viirs"
    df = df.copy()
    df["state"] = np.select([df[f"{prefix}_count"] > 0, df[f"{prefix}_passed"]],
                            ["fire", "clear"], default="not_observed")
    return df
