"""Page 10: Data Quality & Reports."""
import pandas as pd
import streamlit as st

from components import ui
from services import api

ui.page_setup("Data Quality & Reports", "Dataset validation results and known scientific limitations")

result = ui.load(lambda: api.get("/data-quality"), "data quality")
if result is None:
    st.stop()
dq, warnings = result
ui.show_warnings(warnings)

c = st.columns(4)
ui.stat(c[0], "MODIS records", dq.get("modis_records"))
ui.stat(c[1], "VIIRS records", dq.get("viirs_records"))
ui.stat(c[2], "Missing acquisition dates", dq.get("missing_dates"))
ui.stat(c[3], "Synthetic dates used", dq.get("synthetic_dates_used"))
c = st.columns(4)
ui.stat(c[0], "Invalid coordinates", dq.get("invalid_coordinates"))
ui.stat(c[1], "Invalid FRP values", dq.get("invalid_frp_values"))
ui.stat(c[2], "Invalid footprints", dq.get("invalid_footprints"))
ui.stat(c[3], "Unknown confidence", dq.get("unknown_confidence"))
ui.status_badge("Temporal analysis", dq.get("temporal_analysis"))
ui.status_badge("Duplicate records", ui.fmt(dq.get("duplicate_records")))

st.subheader("Warnings and limitations")
dq_warnings = dq.get("warnings") or []
if dq_warnings:
    for w in dq_warnings:
        st.warning(w, icon="⚠️")
else:
    st.success("The backend reported no data quality warnings.")

st.markdown("""
**Known scientific limitations**
- A spatial candidate pair does not prove that two detections are the same fire.
- An unmatched detection is not automatically a sensor-only fire. That needs coverage and timing information.
- MODIS confidence (numeric) and VIIRS confidence (categorical) are kept in their original form.
- FRP values are sensor-specific and are not summed across a pair.
- No accuracy, precision, recall or F1 scores are reported, because no independent reference data is available.
""")

st.subheader("All indicators")
st.dataframe(pd.DataFrame([{"Indicator": k, "Value": ui.fmt(v)} for k, v in dq.items() if k != "warnings"]),
             width="stretch", hide_index=True)

st.subheader("Export")
c = st.columns(2)
for col, fmt_, mime in ((c[0], "csv", "text/csv"), (c[1], "json", "application/json")):
    with col:
        try:
            payload = api.get_bytes("/export", format=fmt_)
            st.download_button(f"Download processed data ({fmt_.upper()})", payload,
                               f"fireharmonize_export.{fmt_}", mime, width="stretch")
        except api.ApiError as exc:
            st.error(f"{fmt_.upper()} export unavailable: {exc}")
if api.is_mock():
    st.caption("Exports in mock mode contain mock data only.")
