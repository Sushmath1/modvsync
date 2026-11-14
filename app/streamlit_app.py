"""Mod-V Sync Streamlit app: map (where) linked to a fire calendar (when).

Run from the repo root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Let the app import 'src' and 'app' from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data import VIEWS, apply_view, load_calendar, load_detections
from app.components.calendar_view import clicked_date, render_calendar
from app.components.cell_history import render_cell_history
from app.components.explain import render_explain, render_region_stats
from app.components.map_view import render_map
from app.components.raw_view import render_raw_map
from app.components.stats import render_stats

RAW_INFO = "Showing raw detections. Run the data pipeline to see the harmonized calendar."


def sync_selected_date(min_date, max_date):
    """Apply a new calendar-bar click to st.session_state['selected_date'] and keep it in range.

    Runs before the sidebar slider is drawn, because Streamlit only lets us set a
    widget's value (via its key) before the widget is created in this run.
    """
    click = clicked_date()
    if click != st.session_state.get("_last_calendar_click"):
        st.session_state["_last_calendar_click"] = click
        if click is not None:
            st.session_state["selected_date"] = click

    current = st.session_state.get("selected_date")
    if current is None or not (min_date <= current <= max_date):
        st.session_state["selected_date"] = min_date


def render_sidebar(df, show_view=True):
    """Draw the sidebar controls and return (region, selected_date, view)."""
    st.sidebar.header("Controls")
    region = st.sidebar.selectbox("Region", sorted(df["region"].unique()))
    dates = df.loc[df["region"] == region, "date"]
    sync_selected_date(dates.min(), dates.max())
    # The slider and the calendar chart share st.session_state["selected_date"]
    selected_date = st.sidebar.slider("Date", min_value=dates.min(), max_value=dates.max(),
                                      key="selected_date", format="YYYY-MM-DD")
    view = st.sidebar.radio("View", VIEWS) if show_view else "Harmonized"
    return region, selected_date, view


def selected_cell_row(df_day):
    """Return the calendar row for the selected cell on the selected day, or None."""
    cell_id = st.session_state.get("selected_cell")
    rows = df_day[df_day["cell_id"] == cell_id]
    return rows.iloc[0] if len(rows) else None


def run_calendar_mode(df):
    """Full app: stats, clickable map, explanation, cell history and clickable calendar."""
    region, selected_date, view = render_sidebar(df)
    df_region = df[df["region"] == region]           # harmonized (used for seen_by stats)
    df_view = apply_view(df_region, view)             # state follows the chosen view
    df_day = df_view[df_view["date"] == selected_date]

    st.subheader(f"{region} · {selected_date:%Y-%m-%d} · {view}")
    render_stats(df_day)

    col_map, col_explain = st.columns([3, 2])
    with col_map:
        clicked = render_map(df_day)
        if clicked:
            st.session_state["selected_cell"] = clicked
    with col_explain:
        render_explain(selected_cell_row(df_day))
        render_region_stats(df_region)

    render_cell_history(df_view, st.session_state.get("selected_cell"), selected_date)

    st.subheader("Fire cells per day")
    st.caption("Click a bar to jump to that day.")
    render_calendar(df_view, selected_date)


def run_raw_mode(detections):
    """Fallback when calendar.parquet is missing: raw detections map coloured by sensor."""
    st.info(RAW_INFO)
    region, selected_date, _ = render_sidebar(detections, show_view=False)
    df_day = detections[(detections["region"] == region) & (detections["date"] == selected_date)]
    st.subheader(f"{region} · {selected_date:%Y-%m-%d} · {len(df_day):,} raw detections")
    render_raw_map(df_day)


def main():
    st.set_page_config(page_title="Mod-V Sync", layout="wide")
    st.title("Mod-V Sync")
    st.caption("One honest fire calendar from two NASA satellites")

    calendar = load_calendar()
    if calendar is not None:
        run_calendar_mode(calendar)
        return

    detections = load_detections()
    if detections is not None:
        run_raw_mode(detections)
        return

    st.error("No data found in data/processed/. Run `python scripts/make_fake_calendar.py` "
             "(fake calendar) or `python scripts/build_processed.py` (raw detections) from the repo root.")


main()
