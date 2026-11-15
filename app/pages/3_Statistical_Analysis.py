"""Page 06: Statistical Analysis (FRP and sensitivity)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from components import ui
from services import api

ui.page_setup("Statistical Analysis", "Sensor-specific detection characteristics and threshold sensitivity")

st.subheader("A. Fire Radiative Power (FRP)")
frp = ui.load(lambda: api.get("/analysis/frp"), "FRP statistics")
if frp:
    data, warnings = frp
    ui.show_warnings(warnings)
    unit = data.get("unit", "MW") if isinstance(data, dict) else "MW"
    stats = {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}
    if stats:
        cols = st.columns(len(stats))
        for col, (sensor, s) in zip(cols, stats.items()):
            with col:
                st.markdown(f"**{sensor}**")
                c = st.columns(3)
                ui.stat(c[0], "Mean", s.get("mean"), f" {unit}")
                ui.stat(c[1], "Median", s.get("median"), f" {unit}")
                ui.stat(c[2], "Maximum", s.get("maximum"), f" {unit}")
        long = pd.DataFrame([{"sensor": k, "statistic": stat, "value": v.get(stat)}
                             for k, v in stats.items() for stat in ("mean", "median", "maximum")])
        fig = px.bar(long, x="statistic", y="value", color="sensor", barmode="group",
                     color_discrete_map=ui.SENSOR_COLORS, title=f"FRP statistics by sensor ({unit})")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    else:
        ui.unavailable("FRP statistics are not available from the backend.")
st.caption("MODIS and VIIRS FRP come from different sensors and pixel sizes. Differences here must not be read "
           "as proof that one sensor detects stronger or more accurate fires.")

det = ui.load(lambda: api.get_all("/detections"), "detections")
df = ui.to_df(det[0]) if det else pd.DataFrame()
if not df.empty and "frp" in df:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="frp", color="sensor", nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map=ui.SENSOR_COLORS, title="FRP distribution (MW)")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
    with right:
        fig = px.box(df, x="sensor", y="frp", color="sensor", color_discrete_map=ui.SENSOR_COLORS,
                     title="FRP by sensor (log scale)")
        fig.update_yaxes(type="log")
        st.plotly_chart(ui.style_fig(fig), width="stretch")

st.divider()
st.subheader("B. Sensitivity analysis")
sens = ui.load(lambda: api.get("/analysis/sensitivity"), "sensitivity results")
if sens:
    sdf = ui.to_df(api.records(sens[0]))
    if sdf.empty:
        ui.unavailable("Sensitivity results are not available from the backend.")
    else:
        fig = px.line(sdf, x="threshold_m", y="candidate_pairs", markers=True,
                      title="Spatial candidate pairs vs distance threshold")
        fig.update_traces(line_color=ui.ACCENT)
        fig.update_layout(xaxis_title="Threshold (m)", yaxis_title="Candidate pairs")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
        st.caption("Shows how strongly the number of candidate associations depends on the chosen threshold.")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Distance distribution")
    assoc = ui.load(lambda: api.get_all("/associations"), "associations")
    adf = ui.to_df(assoc[0]) if assoc else pd.DataFrame()
    if adf.empty:
        ui.unavailable("No associations available.")
    else:
        fig = px.histogram(adf, x="distance_m", nbins=30, title="Association distances (m)")
        fig.update_traces(marker_color=ui.ACCENT)
        st.plotly_chart(ui.style_fig(fig), width="stretch")
with right:
    st.subheader("Day / night distribution")
    if df.empty or "daynight" not in df or df["daynight"].isna().all():
        ui.unavailable("Day/night information is not available.")
    else:
        dn = df.groupby(["sensor", "daynight"]).size().reset_index(name="detections")
        dn["daynight"] = dn["daynight"].map({"D": "Day", "N": "Night"}).fillna(dn["daynight"])
        fig = px.bar(dn, x="daynight", y="detections", color="sensor", barmode="group",
                     color_discrete_map=ui.SENSOR_COLORS, title="Detections by day / night")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
