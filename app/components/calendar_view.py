"""Clickable bar chart of fire cells per day across the full date range."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import STATE_COLORS, first_selected_point

SELECTED_COLOR = "#1F3A5F"
CHART_KEY = "calendar_chart"


def fire_cells_per_day(df_region):
    """Count fire cells per date (dates with no fire get 0)."""
    is_fire = df_region["state"] == "fire"
    return is_fire.groupby(df_region["date"]).sum().sort_index()


def clicked_date():
    """Return the date of the clicked bar from the chart's stored selection, or None.

    Read from st.session_state at the start of a rerun, so it can update the sidebar
    date slider before that slider is drawn.
    """
    point = first_selected_point(st.session_state.get(CHART_KEY))
    if point is None or "x" not in point:
        return None
    return pd.to_datetime(point["x"]).date()


def render_calendar(df_region, selected_date):
    """Draw fire cells per day, with the selected day highlighted. Clicking a bar selects that day."""
    daily = fire_cells_per_day(df_region)
    colors = [SELECTED_COLOR if d == selected_date else STATE_COLORS["fire"] for d in daily.index]

    fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=colors,
                           hovertemplate="%{x|%Y-%m-%d}: %{y} fire cells<extra></extra>"))
    fig.update_layout(height=260, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      yaxis_title="Fire cells", xaxis_title=None, bargap=0.15)
    # Keep every bar fully opaque when one is selected (the colour shows the selected day)
    fig.update_traces(unselected={"marker": {"opacity": 1}})
    st.plotly_chart(fig, width="stretch", key=CHART_KEY, on_select="rerun", selection_mode="points")
