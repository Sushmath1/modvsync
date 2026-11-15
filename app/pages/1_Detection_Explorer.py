"""Page 02: Detection Explorer."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import ui
from services import api

ui.page_setup("Detection Explorer", "Individual MODIS and VIIRS detections as returned by the backend")

with st.sidebar:
    st.subheader("Filters")
    sensor = st.selectbox("Sensor", ["ALL", "MODIS", "VIIRS"])
    satellite = st.selectbox("Satellite", ["ALL", "AQUA", "TERRA", "N20"])
    daynight = st.selectbox("Day / night", ["ALL", "D", "N"], format_func=lambda x: {"ALL": "All", "D": "Day", "N": "Night"}[x])
    min_frp = st.number_input("Min FRP (MW)", min_value=0.0, value=0.0, step=1.0)
    max_frp = st.number_input("Max FRP (MW, 0 = no limit)", min_value=0.0, value=0.0, step=10.0)
    st.subheader("Map layers")
    show_modis = st.checkbox("MODIS layer", value=True)
    show_viirs = st.checkbox("VIIRS layer", value=True)
    size_by_frp = st.checkbox("Marker size by FRP", value=False)
    show_clusters = st.checkbox("Cluster overlay", value=False)

params = dict(sensor=sensor, satellite=satellite, daynight=daynight,
              min_frp=min_frp or None, max_frp=max_frp or None)
result = ui.load(lambda: api.get_all("/detections", **params), "detections")
if result is None:
    st.stop()
rows, warnings, _ = result
ui.show_warnings(warnings)
df = ui.to_df(rows)
if df.empty:
    ui.empty()
    st.stop()

layers = [s for s, on in (("MODIS", show_modis), ("VIIRS", show_viirs)) if on]
map_df = df[df["sensor"].isin(layers)]

c = st.columns(4)
ui.stat(c[0], "Detections shown", len(df))
ui.stat(c[1], "MODIS", int((df["sensor"] == "MODIS").sum()))
ui.stat(c[2], "VIIRS", int((df["sensor"] == "VIIRS").sum()))
ui.stat(c[3], "With acquisition date", int(df["acq_date"].notna().sum()) if "acq_date" in df else None)

if map_df.empty:
    ui.empty("All map layers are turned off.")
else:
    hover = [c for c in ["detection_id", "satellite", "frp", "confidence", "daynight", "scan_km", "track_km",
                         "footprint_area_km2", "acq_time_utc"] if c in map_df]
    center, zoom = ui.map_view(map_df)
    plot_df = map_df.copy()
    if size_by_frp and "frp" in plot_df:
        plot_df["_size"] = plot_df["frp"].clip(lower=0.1)
    fig = px.scatter_map(plot_df, lat="latitude", lon="longitude", color="sensor",
                         color_discrete_map=ui.SENSOR_COLORS, hover_name="detection_id" if "detection_id" in plot_df else None,
                         hover_data=hover, size="_size" if size_by_frp else None, size_max=18,
                         opacity=0.85, center=center, zoom=zoom, map_style=ui.MAP_STYLE)
    if not size_by_frp:
        fig.update_traces(marker=dict(size=9))
    if show_clusters:
        cl = ui.load(lambda: api.get_all("/clusters"), "clusters")
        if cl and cl[0]:
            cdf = ui.to_df(cl[0])
            fig.add_trace(go.Scattermap(
                lat=cdf["latitude"], lon=cdf["longitude"], mode="markers",
                marker=dict(size=(cdf["detection_count"] * 3 + 10).clip(upper=60), color=ui.ACCENT, opacity=0.25),
                text=cdf["cluster_id"] + " · " + cdf["detection_count"].astype(str) + " detections",
                hoverinfo="text", name="Clusters"))
        elif cl is not None:
            ui.unavailable("No cluster results available from the backend.")
    st.plotly_chart(ui.style_fig(fig, height=560), width="stretch")

st.subheader("Detection details")
if "detection_id" in df:
    chosen = st.selectbox("Select a detection", df["detection_id"].tolist())
    d = df[df["detection_id"] == chosen].iloc[0]
    fields = [("Sensor", "sensor"), ("Satellite / platform", "satellite"), ("Latitude", "latitude"),
              ("Longitude", "longitude"), ("FRP (MW)", "frp"), ("Confidence (sensor-specific)", "confidence"),
              ("Day / night", "daynight"), ("Scan (km)", "scan_km"), ("Track (km)", "track_km"),
              ("Footprint area (km²)", "footprint_area_km2"), ("Acquisition time (UTC)", "acq_time_utc"),
              ("Acquisition date", "acq_date"), ("Timestamp source", "timestamp_source")]
    cols = st.columns(4)
    for i, (label, key) in enumerate(fields):
        cols[i % 4].markdown(f"**{label}**  \n{ui.fmt(d.get(key)) if key in d else 'Not available'}")

st.subheader("Detection table")
table = df.copy()
if "confidence" in table:
    # MODIS confidence is numeric and VIIRS is categorical, so show both as text
    table["confidence"] = table["confidence"].astype(str)
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download filtered detections (CSV)", df.to_csv(index=False).encode(),
                   "detections.csv", "text/csv")
