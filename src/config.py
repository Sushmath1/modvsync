"""Project-wide settings: paths, regions and thresholds."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_MODIS_DIR = ROOT / "data" / "raw" / "modis"
RAW_VIIRS_DIR = ROOT / "data" / "raw" / "viirs"
PROCESSED_DIR = ROOT / "data" / "processed"
DETECTIONS_PATH = PROCESSED_DIR / "detections.parquet"

# --- Regions -----------------------------------------------------------
# Rough bounding boxes (degrees). Used as a sanity check on coordinates.
REGION_BBOX = {
    "Chad":    {"lat_min": 7.0,   "lat_max": 23.6, "lon_min": 13.3,  "lon_max": 24.1},
    "Bolivia": {"lat_min": -23.0, "lat_max": -9.5, "lon_min": -69.8, "lon_max": -57.3},
}

# --- Platforms ---------------------------------------------------------
# Maps the FIRMS 'satellite' column to a readable platform name.
PLATFORM_NAMES = {"T": "Terra", "Terra": "Terra", "A": "Aqua", "Aqua": "Aqua",
                  "N": "Suomi-NPP", "N20": "NOAA-20", "N21": "NOAA-21"}

# --- Thresholds --------------------------------------------------------
GRID_SIZE_DEG = 0.1  # grid cell size for cell_id

# MODIS confidence (0-100) is binned into the same classes VIIRS uses,
# following the FIRMS convention: low < 30 <= nominal < 80 <= high.
MODIS_CONF_BINS = [-1, 29, 79, 100]
CONF_CLASSES = ["low", "nominal", "high"]
VIIRS_CONF_MAP = {"l": "low", "n": "nominal", "h": "high"}

# Detections with these confidence classes are dropped during cleaning.
DROP_CONFIDENCE = ["low"]

# FIRMS 'type': 0 = presumed vegetation fire (1 volcano, 2 static land source, 3 offshore)
KEEP_TYPES = [0]
