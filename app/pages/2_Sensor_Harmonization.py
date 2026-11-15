"""Page 04: Sensor Harmonization (MODIS-VIIRS spatial association)."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import ui
from services import api

ui.page_setup("Sensor Harmonization", "Backend-calculated spatial associations between MODIS and VIIRS detections")

st.info("A **spatial candidate** means the two detection coordinates satisfy the configured spatial criterion. "
        "It is **not** a confirmed same-fire event. Temporal status is shown separately and is unavailable "
        "while acquisition dates are missing.", icon="🔬")

with st.sidebar:
    st.subheader("Filters")
    statuses = st.multiselect("Association status", ui.ASSOCIATION_LABELS, default=ui.ASSOCIATION_LABELS)
    min_d = st.number_input("Min distance (m)", min_value=0.0, value=0.0, step=100.0)
    max_d = st.number_input("Max distance (m, 0 = no limit)", min_value=0.0, value=0.0, step=100.0)

result = ui.load(lambda: api.get_all("/associations", min_distance_m=min_d or None, max_distance_m=max_d or None),
                 "associations")
if result is None:
    st.stop()
rows, warnings, _ = result
ui.show_warnings(warnings)
all_df = ui.to_df(rows)
if all_df.empty:
    ui.unavailable("The backend returned no associations. Run the association analysis first.")
    st.stop()

# Summary cards always reflect everything the backend returned
c = st.columns(len(ui.ASSOCIATION_LABELS) + 1)
ui.stat(c[0], "Total associations", len(all_df))
for col, label in zip(c[1:], ui.ASSOCIATION_LABELS):
    ui.stat(col, label.replace("_", " ").title(), int((all_df["association_status"] == label).sum()))

df = all_df[all_df["association_status"].isin(statuses)]
if df.empty:
    ui.empty()
    st.stop()

left, right = st.columns(2)
with left:
    fig = px.histogram(df, x="distance_m", color="association_status", nbins=30,
                       color_discrete_map=ui.STATUS_COLORS, title="Distance distribution (m)")
    if "spatial_threshold_m" in df and df["spatial_threshold_m"].notna().any():
        thr = float(df["spatial_threshold_m"].median())
        fig.add_vline(x=thr, line_dash="dash", line_color=ui.ACCENT,
                      annotation_text=f"median threshold {thr:,.0f} m")
    st.plotly_chart(ui.style_fig(fig), width="stretch")
with right:
    if "similarity_score" in df and df["similarity_score"].notna().any():
        fig = px.scatter(df, x="distance_m", y="similarity_score", color="association_status",
                         color_discrete_map=ui.STATUS_COLORS, hover_name="association_id",
                         title="Distance vs similarity score")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
        st.caption("Similarity score is a mathematical similarity measure, not a probability of a true match.")
    else:
        ui.unavailable("Similarity scores were not provided by the backend.")

# Relationship lines on the map, using backend detection coordinates
det = ui.load(lambda: api.get_all("/detections"), "detections")
dets = ui.to_df(det[0]) if det else None
st.subheader("Candidate relationships on the map")
if dets is None or dets.empty or "detection_id" not in dets:
    ui.unavailable("Detection coordinates are needed to draw relationship lines.")
else:
    pos = dets.set_index("detection_id")[["latitude", "longitude"]]
    linked = df[df["modis_id"].isin(pos.index) & df["viirs_id"].isin(pos.index)]
    fig = go.Figure()
    for status, g in linked.groupby("association_status"):
        lats, lons = [], []
        for _, r in g.iterrows():
            m, v = pos.loc[r["modis_id"]], pos.loc[r["viirs_id"]]
            lats += [m["latitude"], v["latitude"], None]
            lons += [m["longitude"], v["longitude"], None]
        fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines", name=status,
                                    line=dict(width=2, color=ui.STATUS_COLORS.get(status, "#FFFFFF"))))
    ids = set(linked["modis_id"]) | set(linked["viirs_id"])
    pts = dets[dets["detection_id"].isin(ids)]
    for sensor, g in pts.groupby("sensor"):
        fig.add_trace(go.Scattermap(lat=g["latitude"], lon=g["longitude"], mode="markers", name=sensor,
                                    marker=dict(size=9, color=ui.SENSOR_COLORS.get(sensor)),
                                    text=g["detection_id"], hoverinfo="text"))
    center, zoom = ui.map_view(pts if not pts.empty else dets)
    fig.update_layout(map=dict(style=ui.MAP_STYLE, center=center, zoom=zoom))
    st.plotly_chart(ui.style_fig(fig, height=520), width="stretch")

st.subheader("Association table")
st.dataframe(df, width="stretch", hide_index=True)
st.download_button("Download associations (CSV)", df.to_csv(index=False).encode(), "associations.csv", "text/csv")

st.subheader("Association details")
chosen = st.selectbox("Select an association", df["association_id"].tolist())
a = df[df["association_id"] == chosen].iloc[0]
c = st.columns(4)
ui.stat(c[0], "Distance", float(a["distance_m"]), " m")
ui.stat(c[1], "Spatial threshold", float(a["spatial_threshold_m"]) if a.get("spatial_threshold_m") is not None else None, " m")
ui.stat(c[2], "Association status", a["association_status"])
ui.stat(c[3], "Temporal status", a.get("temporal_status"))
if dets is not None and not dets.empty and "detection_id" in dets:
    show = ["satellite", "latitude", "longitude", "frp", "confidence", "daynight", "scan_km", "track_km",
            "footprint_area_km2", "acq_time_utc", "acq_date"]
    m_col, v_col = st.columns(2)
    for col, det_id, sensor in ((m_col, a["modis_id"], "MODIS"), (v_col, a["viirs_id"], "VIIRS")):
        with col:
            st.markdown(f"**{sensor} · {det_id}**")
            match = dets[dets["detection_id"] == det_id]
            if match.empty:
                ui.unavailable("Detection not found in backend records.")
            else:
                r = match.iloc[0]
                st.table({"Field": show, "Value": [ui.fmt(r.get(k)) if k in r else "Not available" for k in show]})
    st.caption("FRP and confidence are shown in their original sensor-specific form and are never added together "
               "for a pair, which could double-count the same activity.")
