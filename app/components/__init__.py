"""UI building blocks for the Streamlit app."""

# Shared state colours (used by the map, the calendar and the cell history)
STATE_COLORS = {"fire": "#E4572E", "clear": "#DDE3EA", "not_observed": "#8A94A6"}
STATE_ORDER = ["fire", "clear", "not_observed"]
STATE_LABELS = {"fire": "Fire", "clear": "Clear", "not_observed": "Not observed"}

# Sensor colours for the raw-detections fallback map
SENSOR_COLORS = {"MODIS": "#E4572E", "VIIRS": "#2E86AB"}


def first_selected_point(event):
    """Return the first selected point (a dict) from a plotly_chart selection event, or None."""
    if not event:
        return None
    points = event.get("selection", {}).get("points", [])
    return points[0] if points else None
