"""Map of grid cells for one day, coloured by state. Clicking a cell selects it."""
import numpy as np
import plotly.express as px
import streamlit as st

from app.components import STATE_COLORS, STATE_ORDER, first_selected_point

HOVER_COLUMNS = ["state", "modis_count", "viirs_count", "seen_by", "reason"]


def auto_zoom(df, lat="lat", lon="lon"):
    """Pick a map zoom level that fits the points' lat/lon extent."""
    span = max(df[lat].max() - df[lat].min(), df[lon].max() - df[lon].min(), 0.1)
    return float(np.clip(np.log2(360 / span) - 1, 1, 12))


def clicked_cell(event):
    """Return the cell_id of the clicked map point, or None if nothing was clicked."""
    point = first_selected_point(event)
    if point is None or not point.get("customdata"):
        return None
    return point["customdata"][0]


def render_map(df_day):
    """Draw the day's cells on an OpenStreetMap basemap and return the clicked cell_id (or None)."""
    fig = px.scatter_map(
        df_day, lat="lat", lon="lon", color="state",
        color_discrete_map=STATE_COLORS, category_orders={"state": STATE_ORDER},
        hover_name="cell_id", hover_data={**{c: True for c in HOVER_COLUMNS}, "lat": False, "lon": False},
        custom_data=["cell_id"],
        center={"lat": df_day["lat"].mean(), "lon": df_day["lon"].mean()},
        zoom=auto_zoom(df_day), map_style="open-street-map", height=520,
    )
    fig.update_traces(marker={"size": 11, "opacity": 0.85})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend_title_text="State")
    event = st.plotly_chart(fig, width="stretch", key="map_chart",
                            on_select="rerun", selection_mode="points")
    return clicked_cell(event)
