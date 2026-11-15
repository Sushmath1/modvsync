"""Shared UI building blocks: page setup, stat cards, states, colours, map helpers."""
import math

import pandas as pd
import streamlit as st

from services import api

SENSOR_COLORS = {"MODIS": "#E4572E", "VIIRS": "#2E86AB"}
STATUS_COLORS = {
    "SPATIAL_CANDIDATE": "#F3A712",
    "SPATIOTEMPORAL_CANDIDATE": "#4CAF8A",
    "TEMPORAL_UNAVAILABLE": "#8A94A6",
    "AMBIGUOUS": "#B565A7",
    "UNRESOLVED": "#5C6B80",
}
ASSOCIATION_LABELS = list(STATUS_COLORS)
ACCENT = "#F3A712"
MAP_STYLE = "carto-darkmatter"
PLOT_TEMPLATE = "plotly_dark"


def page_setup(title, subtitle):
    """First call on every page: config, sidebar data-source settings, title, mock banner."""
    st.set_page_config(page_title=f"{title} · FireHarmonize", page_icon="🛰️", layout="wide")
    api.sidebar_settings()
    st.title(title)
    st.caption(subtitle)
    if api.is_mock():
        st.warning("**Mock data mode.** All values are illustrative and are not research results. "
                   "Switch to *Live API* in the sidebar to use the backend.", icon="⚠️")


def load(fn, what="data"):
    """Run an API call and show an error state instead of crashing. Returns None on failure."""
    try:
        with st.spinner(f"Loading {what}..."):
            return fn()
    except api.ApiError as exc:
        st.error(f"Could not load {what} from the backend.\n\n{exc}", icon="🚫")
        if not api.is_mock():
            st.caption("Check that the FastAPI backend is running and the API base URL in the sidebar is correct.")
        return None


def show_warnings(warnings):
    """Show backend warnings (the mock warning is already shown as a banner)."""
    for w in dict.fromkeys(warnings or []):
        if not str(w).startswith("MOCK DATA"):
            st.warning(w, icon="⚠️")


def fmt(value, unit=""):
    """Format a value for display. None becomes 'Not available'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}{unit}"
    if isinstance(value, float):
        return f"{value:,.2f}{unit}"
    return f"{value}{unit}"


def stat(col, label, value, unit="", help=None):
    col.metric(label, fmt(value, unit), help=help)


def status_badge(label, status):
    """Small coloured status line, e.g. Temporal analysis: UNAVAILABLE."""
    good = {"PROCESSED", "AVAILABLE", "COMPLETED", "OK"}
    colour = "#4CAF8A" if str(status).upper() in good else "#F3A712"
    st.markdown(f"{label}: <span style='color:{colour};font-weight:600'>{status or 'Not available'}</span>",
                unsafe_allow_html=True)


def unavailable(message):
    st.info(message, icon="ℹ️")


def empty(message="No records match the current filters."):
    st.info(message, icon="🔍")


def map_view(df, lat="latitude", lon="longitude"):
    """Centre and zoom that fit the given points."""
    if df is None or df.empty:
        return {"lat": 0, "lon": 0}, 1
    lat_span = max(df[lat].max() - df[lat].min(), 0.05)
    lon_span = max(df[lon].max() - df[lon].min(), 0.05)
    span = max(lat_span, lon_span)
    zoom = max(1, min(12, math.log2(360 / span) - 1.2))
    return {"lat": float(df[lat].mean()), "lon": float(df[lon].mean())}, zoom


def to_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def style_fig(fig, height=380):
    fig.update_layout(template=PLOT_TEMPLATE, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.15))
    return fig
