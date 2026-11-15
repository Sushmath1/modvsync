"""Page 01: Overview dashboard."""
import plotly.express as px
import streamlit as st

from components import ui
from services import api

ui.page_setup("Overview", "FireHarmonize (Mod-V Sync) · Sensor-aware harmonization of MODIS and VIIRS active-fire detections")

result = ui.load(lambda: api.get("/dashboard/summary"), "dashboard summary")
if result is None:
    st.stop()
summary, warnings = result
ui.show_warnings(warnings)

c = st.columns(4)
ui.stat(c[0], "Total detections", summary.get("total_detections"))
ui.stat(c[1], "MODIS detections", summary.get("modis_detections"))
ui.stat(c[2], "VIIRS detections", summary.get("viirs_detections"))
ui.stat(c[3], "Spatial clusters", summary.get("spatial_clusters"))
c = st.columns(4)
ui.stat(c[0], "Spatial candidate pairs", summary.get("spatial_candidate_pairs"),
        help="Pairs whose coordinates satisfy the spatial criterion. Not confirmed same-fire events.")
ui.stat(c[1], "Spatiotemporal candidate pairs", summary.get("spatiotemporal_candidate_pairs"),
        help="Requires valid acquisition dates.")
with c[2]:
    ui.status_badge("Dataset status", summary.get("dataset_status"))
    ui.status_badge("Temporal analysis", summary.get("temporal_analysis_status"))
with c[3]:
    dq = ui.load(lambda: api.get("/data-quality"), "data quality")
    if dq:
        n_warn = len(dq[0].get("warnings") or [])
        ui.status_badge("Data quality", "OK" if n_warn == 0 else f"{n_warn} warning(s)")
        st.page_link("pages/7_Data_Quality.py", label="View data quality →")

st.divider()

det = ui.load(lambda: api.get_all("/detections"), "detections")
if det is None:
    st.stop()
df = ui.to_df(det[0])
if df.empty:
    ui.empty("The backend returned no detections.")
    st.stop()

left, right = st.columns(2)
with left:
    counts = df.groupby("sensor").size().reset_index(name="detections")
    fig = px.bar(counts, x="sensor", y="detections", color="sensor", color_discrete_map=ui.SENSOR_COLORS,
                 title="Detections by sensor", text="detections")
    st.plotly_chart(ui.style_fig(fig), width="stretch")
with right:
    if "daynight" in df:
        dn = df.groupby(["sensor", "daynight"]).size().reset_index(name="detections")
        dn["daynight"] = dn["daynight"].map({"D": "Day", "N": "Night"}).fillna(dn["daynight"])
        fig = px.bar(dn, x="daynight", y="detections", color="sensor", barmode="group",
                     color_discrete_map=ui.SENSOR_COLORS, title="Day / night distribution")
        st.plotly_chart(ui.style_fig(fig), width="stretch")

left, right = st.columns(2)
with left:
    if "frp" in df:
        fig = px.box(df, x="sensor", y="frp", color="sensor", color_discrete_map=ui.SENSOR_COLORS,
                     points="outliers", title="FRP summary by sensor (MW, original sensor values)")
        fig.update_yaxes(type="log", title="FRP (MW, log scale)")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
with right:
    center, zoom = ui.map_view(df)
    fig = px.scatter_map(df, lat="latitude", lon="longitude", color="sensor", color_discrete_map=ui.SENSOR_COLORS,
                         opacity=0.8, center=center, zoom=zoom, map_style=ui.MAP_STYLE,
                         title="Spatial distribution of detections")
    st.plotly_chart(ui.style_fig(fig), width="stretch")

runs = ui.load(lambda: api.get("/analysis/runs"), "processing history")
if runs:
    rows = api.records(runs[0])
    st.subheader("Recent processing")
    if rows:
        latest = rows[0]
        c = st.columns(4)
        ui.stat(c[0], "Analysis ID", latest.get("analysis_id"))
        ui.stat(c[1], "Status", latest.get("status"))
        ui.stat(c[2], "Records processed", latest.get("records_processed"))
        ui.stat(c[3], "Processed at", latest.get("processed_at"))
        st.caption(f"Algorithm: {latest.get('algorithm', 'Not available')}")
    else:
        ui.empty("No processing runs recorded yet.")
