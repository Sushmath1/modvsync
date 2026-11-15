"""Page 03: Temporal Explorer.

The supplied datasets have no acquisition dates, so this page shows the backend's
temporal status and keeps its controls disabled. It never invents dates.
"""
import plotly.express as px
import streamlit as st

from components import ui
from services import api

ui.page_setup("Temporal Explorer", "Temporal distribution of active-fire observations")

result = ui.load(lambda: api.get("/analysis/temporal"), "temporal analysis")
if result is None:
    st.stop()
data, warnings = result
ui.show_warnings(warnings)
status = str(data.get("status", "UNAVAILABLE")).upper() if isinstance(data, dict) else "UNAVAILABLE"
available = status in {"AVAILABLE", "COMPLETED", "OK"}

c = st.columns(3)
ui.stat(c[0], "Temporal analysis", status)
ui.stat(c[1], "Records with valid dates", data.get("available_records") if isinstance(data, dict) else None)
ui.stat(c[2], "Reason", data.get("reason") if isinstance(data, dict) else None)

if not available:
    st.warning(f"**Temporal analysis unavailable.** {data.get('message', 'Valid acquisition dates are required.')}  \n"
               "No historical calendar is shown, because displaying synthetic dates as NASA observation dates "
               "would be misleading.", icon="📅")

st.subheader("Controls")
c = st.columns(3)
c[0].date_input("Date range", value=[], disabled=not available)
c[1].select_slider("Aggregation", ["Daily", "Weekly", "Monthly"], disabled=not available)
c[2].slider("Time of day (UTC)", 0, 23, (0, 23), disabled=not available)
if not available:
    st.caption("Calendar and timeline components are ready and will activate once the backend reports valid dates.")

if available:
    rows = api.records(data) or (data.get("series") if isinstance(data, dict) else [])
    tdf = ui.to_df(rows)
    if tdf.empty:
        ui.empty("The backend reported temporal data as available but returned no series.")
    else:
        x = "date" if "date" in tdf else tdf.columns[0]
        y = "count" if "count" in tdf else tdf.columns[-1]
        fig = px.line(tdf, x=x, y=y, color="sensor" if "sensor" in tdf else None,
                      color_discrete_map=ui.SENSOR_COLORS, markers=True, title="Sensor-wise timeline")
        st.plotly_chart(ui.style_fig(fig), width="stretch")
