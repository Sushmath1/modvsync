"""Metric cards for the selected day."""
import streamlit as st


def render_stats(df_day):
    """Show fire / clear / not observed cell counts for one day as three metric cards."""
    counts = df_day["state"].value_counts()
    col_fire, col_clear, col_none = st.columns(3)
    col_fire.metric("Fire cells", int(counts.get("fire", 0)))
    col_clear.metric("Clear cells", int(counts.get("clear", 0)))
    col_none.metric("Not observed cells", int(counts.get("not_observed", 0)))
