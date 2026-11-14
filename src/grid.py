"""Assign each detection to a grid cell and a calendar date."""
import numpy as np

from src import config


def add_grid(df, size=config.GRID_SIZE_DEG):
    """Add cell_id = 'lat_lon' of the cell's lower-left corner, e.g. '10.8_16.8'.

    Detections from both sensors that fall in the same cell share a cell_id,
    which is what lets us compare MODIS and VIIRS later.
    """
    # Small epsilon avoids float errors like 10.8 / 0.1 = 107.999...
    lat0 = (np.floor(df["latitude"] / size + 1e-9) * size).round(4)
    lon0 = (np.floor(df["longitude"] / size + 1e-9) * size).round(4)
    df["cell_id"] = lat0.map("{:.1f}".format) + "_" + lon0.map("{:.1f}".format)
    return df


def add_date(df):
    """Add the UTC calendar date of each detection."""
    df["date"] = df["datetime_utc"].dt.date
    return df
