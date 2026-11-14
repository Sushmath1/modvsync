"""'Why did the sensors disagree?' panel: one cell-day explanation plus region stats."""
import plotly.graph_objects as go
import streamlit as st

from app.components import STATE_COLORS

SEEN_BY_LABELS = {"both": "Both", "modis": "MODIS only", "viirs": "VIIRS only"}


def explain_text(row):
    """Return a plain-English sentence for one calendar row, based on seen_by and the pass flags."""
    seen_by = row["seen_by"]
    if seen_by == "both":
        return "Both satellites detected this fire."
    if seen_by == "viirs":
        if row["modis_passed"]:
            return "Only VIIRS saw this. Likely a small fire below MODIS's 1 km resolution."
        return "Only VIIRS saw this. MODIS didn't pass over this area today."
    if seen_by == "modis":
        return "Only MODIS saw this. VIIRS may have passed at a different time, or not at all."
    if row["modis_passed"] or row["viirs_passed"]:
        return "A satellite passed over and saw no fire."
    return "No satellite passed over this square today, so we can't say."


def render_explain(cell_row):
    """Explain the selected cell-day (a calendar row), or show a hint if cell_row is None."""
    st.subheader("Why did the sensors disagree?")
    if cell_row is None:
        st.caption("Click a square on the map to explain that cell on the selected day.")
        return
    st.markdown(f"**{cell_row['cell_id']}** on {cell_row['date']:%Y-%m-%d}")
    st.write(explain_text(cell_row))
    if cell_row["reason"]:
        st.caption(f"Reason: {cell_row['reason']}")


def render_region_stats(df_region):
    """Fire cell-days by seen_by as a small bar chart, plus the share of cell-days not observed."""
    start, end = df_region["date"].min(), df_region["date"].max()
    st.markdown(f"**Region stats** · {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    counts = df_region["seen_by"].value_counts().reindex(list(SEEN_BY_LABELS), fill_value=0)
    fig = go.Figure(go.Bar(x=[SEEN_BY_LABELS[k] for k in counts.index], y=counts.values,
                           marker_color=STATE_COLORS["fire"],
                           hovertemplate="%{x}: %{y} fire cell-days<extra></extra>"))
    fig.update_layout(height=200, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      yaxis_title="Fire cell-days")
    st.plotly_chart(fig, width="stretch")

    not_observed = (df_region["state"] == "not_observed").mean()
    st.metric("Cell-days not observed", f"{not_observed:.1%}")
