"""Page 07: Analysis Workspace. Sends parameters to the backend, which recalculates."""
import streamlit as st

from components import ui
from services import api

ui.page_setup("Analysis Workspace", "Adjust parameters and request a backend recalculation")

st.caption("The frontend only sends parameters. All calculations run on the backend, which returns the "
           "parameters it actually applied.")

with st.form("run_associations"):
    st.subheader("Spatial association")
    c = st.columns(3)
    uncertainty = c[0].slider("Spatial uncertainty allowance (km)", 0.0, 2.0, 0.25, 0.05)
    mode = c[1].selectbox("Threshold mode", ["FOOTPRINT_BASED"],
                          help="Additional modes become available once the backend supports them.")
    c[2].number_input("Temporal window (minutes)", value=0, disabled=True,
                      help="Disabled: requires valid acquisition dates.")
    st.subheader("DBSCAN clustering")
    c = st.columns(2)
    c[0].number_input("Epsilon (km)", value=4.0, disabled=True)
    c[1].number_input("Minimum samples", value=4, disabled=True)
    st.caption("Clustering parameters are display-only until the backend exposes a clustering run endpoint.")
    submitted = st.form_submit_button("Run analysis on backend", type="primary")

if submitted:
    body = {"uncertainty_km": uncertainty, "distance_threshold_mode": mode, "time_window_minutes": None}
    result = ui.load(lambda: api.post("/analysis/associations/run", body), "analysis run")
    if result:
        data, warnings = result
        ui.show_warnings(warnings)
        st.success(f"Run **{data.get('analysis_id', 'unknown')}** finished with status "
                   f"**{data.get('status', 'unknown')}**.")
        c = st.columns(3)
        ui.stat(c[0], "Spatial candidate pairs", data.get("spatial_candidate_pairs"))
        ui.stat(c[1], "Temporal analysis", data.get("temporal_analysis_status"))
        ui.stat(c[2], "Analysis ID", data.get("analysis_id"))
        applied = data.get("applied_parameters") or data.get("parameters")
        st.markdown("**Parameters applied by the backend**")
        if applied:
            st.json(applied)
        else:
            ui.unavailable("The backend did not report the applied parameters.")
        st.cache_data.clear()
        st.caption("Cached results were cleared, so the other pages will load the updated backend results.")
