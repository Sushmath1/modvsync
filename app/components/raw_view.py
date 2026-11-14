"""Fallback map of raw detections, used when calendar.parquet doesn't exist yet."""
import plotly.express as px
import streamlit as st

from app.components import SENSOR_COLORS
from app.components.map_view import auto_zoom


def render_raw_map(df_day):
    """Draw one day's raw detections, coloured by sensor."""
    if df_day.empty:
        st.caption("No detections on this day.")
        return
    fig = px.scatter_map(
        df_day, lat="latitude", lon="longitude", color="sensor",
        color_discrete_map=SENSOR_COLORS,
        hover_data={"platform": True, "frp": True, "confidence_norm": True,
                    "datetime_utc": True, "latitude": False, "longitude": False},
        center={"lat": df_day["latitude"].mean(), "lon": df_day["longitude"].mean()},
        zoom=auto_zoom(df_day, lat="latitude", lon="longitude"),
        map_style="open-street-map", height=520,
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.8})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend_title_text="Sensor")
    st.plotly_chart(fig, width="stretch")
