"""History of one grid cell across the full date range."""
import plotly.express as px
import streamlit as st

from app.components import STATE_COLORS, STATE_LABELS, STATE_ORDER

HINT = "Click a square on the map to see its history."


def render_strip(df_cell, selected_date):
    """One coloured marker per day for the cell, with the selected day marked."""
    df_plot = df_cell.assign(row="", state_label=df_cell["state"].map(STATE_LABELS))
    fig = px.scatter(
        df_plot, x="date", y="row", color="state_label",
        color_discrete_map={STATE_LABELS[s]: c for s, c in STATE_COLORS.items()},
        category_orders={"state_label": [STATE_LABELS[s] for s in STATE_ORDER]},
        hover_data={"row": False, "seen_by": True, "reason": True}, height=170,
    )
    fig.update_traces(marker={"symbol": "square", "size": 10})
    fig.add_vline(x=selected_date, line_width=1, line_dash="dot", line_color="#1F3A5F")
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text=None,
                      yaxis={"visible": False}, xaxis_title=None,
                      legend={"orientation": "h", "y": -0.35})
    st.plotly_chart(fig, width="stretch")


def render_cell_metrics(df_cell):
    """Total fire days and how they split between the sensors."""
    fire = df_cell[df_cell["state"] == "fire"]
    seen = df_cell["seen_by"].value_counts()
    cols = st.columns(4)
    cols[0].metric("Fire days", len(fire))
    cols[1].metric("Seen by both", int(seen.get("both", 0)))
    cols[2].metric("MODIS only", int(seen.get("modis", 0)))
    cols[3].metric("VIIRS only", int(seen.get("viirs", 0)))


def render_cell_history(df_region, cell_id, selected_date):
    """Show the 'Cell history' section for cell_id, or a hint if no cell is selected."""
    st.subheader("Cell history")
    df_cell = df_region[df_region["cell_id"] == cell_id].sort_values("date") if cell_id else None
    if df_cell is None or df_cell.empty:
        st.caption(HINT)
        return

    first = df_cell.iloc[0]
    st.markdown(f"**Cell {cell_id}** · centre {first['lat']:.2f}, {first['lon']:.2f}")
    render_strip(df_cell, selected_date)
    render_cell_metrics(df_cell)
