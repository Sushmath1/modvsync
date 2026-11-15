"""Page 05: Cluster & Density Analysis."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import ui
from services import api

ui.page_setup("Cluster & Density Analysis", "Backend-calculated spatial groupings of detections")

st.info("Without acquisition dates, clusters are **spatial groupings of detections**, "
        "not confirmed individual fire events.", icon="🔬")

result = ui.load(lambda: api.get_all("/clusters"), "clusters")
if result is None:
    st.stop()
rows, warnings, extra = result
ui.show_warnings(warnings)
cdf = ui.to_df(rows)

c = st.columns(4)
ui.stat(c[0], "Spatial clusters", len(cdf) if not cdf.empty else None)
ui.stat(c[1], "Noise / unclustered", extra.get("noise_count"))
ui.stat(c[2], "Clustered detections", int(cdf["detection_count"].sum()) if not cdf.empty else None)
ui.stat(c[3], "Algorithm", extra.get("algorithm"))
if extra.get("parameters"):
    st.caption("Parameters: " + ", ".join(f"{k} = {v}" for k, v in extra["parameters"].items()))

det = ui.load(lambda: api.get_all("/detections"), "detections")
dets = ui.to_df(det[0]) if det else None

left, right = st.columns(2)
with left:
    st.subheader("Cluster map")
    if cdf.empty:
        ui.unavailable("No cluster results available from the backend.")
    else:
        fig = go.Figure()
        if dets is not None and not dets.empty:
            for sensor, g in dets.groupby("sensor"):
                fig.add_trace(go.Scattermap(lat=g["latitude"], lon=g["longitude"], mode="markers", name=sensor,
                                            marker=dict(size=6, color=ui.SENSOR_COLORS.get(sensor), opacity=0.6),
                                            hoverinfo="skip"))
        fig.add_trace(go.Scattermap(
            lat=cdf["latitude"], lon=cdf["longitude"], mode="markers", name="Cluster centres",
            marker=dict(size=(cdf["detection_count"] * 2 + 10).clip(upper=50), color=ui.ACCENT, opacity=0.55),
            text=cdf["cluster_id"] + "<br>" + cdf["detection_count"].astype(str) + " detections", hoverinfo="text"))
        center, zoom = ui.map_view(cdf)
        fig.update_layout(map=dict(style=ui.MAP_STYLE, center=center, zoom=zoom))
        st.plotly_chart(ui.style_fig(fig, height=460), width="stretch")
with right:
    st.subheader("Density heatmap")
    if dets is None or dets.empty:
        ui.unavailable("Detections are needed for the density view.")
    else:
        center, zoom = ui.map_view(dets)
        fig = px.density_map(dets, lat="latitude", lon="longitude", radius=18, center=center, zoom=zoom,
                             map_style=ui.MAP_STYLE, color_continuous_scale="YlOrRd")
        st.plotly_chart(ui.style_fig(fig, height=460), width="stretch")
        st.caption("Visual density of backend detection coordinates (display only).")

if not cdf.empty:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(cdf, x="detection_count", nbins=20, title="Cluster size distribution")
        fig.update_traces(marker_color=ui.ACCENT)
        fig.update_layout(xaxis_title="Detections per cluster", yaxis_title="Clusters")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    with right:
        long = cdf.melt(id_vars="cluster_id", value_vars=["modis_count", "viirs_count"],
                        var_name="sensor", value_name="detections")
        long["sensor"] = long["sensor"].map({"modis_count": "MODIS", "viirs_count": "VIIRS"})
        fig = px.bar(long, x="cluster_id", y="detections", color="sensor", color_discrete_map=ui.SENSOR_COLORS,
                     title="Sensor distribution within clusters")
        st.plotly_chart(ui.style_fig(fig), width="stretch")

    st.subheader("Cluster details")
    st.dataframe(cdf, width="stretch", hide_index=True)
    st.download_button("Download clusters (CSV)", cdf.to_csv(index=False).encode(), "clusters.csv", "text/csv")
