"""
dashboard.py
============

The dashboard page brings together insights from the data exploration
and modelling stages.  It provides a high‑level overview of the
dataset, displays recent model performance and offers quick access to
key visualisations.  In a production environment this page could be
extended to allow exporting to PDF/HTML or to embed charts in other
systems.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils import data_utils as du
from utils import plotting as pl
from utils import plotting as pl


def show_dashboard_page() -> None:
    """Render the dashboard page."""
    st.header("📊 Dashboard Overview")
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please load and prepare a dataset on the Data page first.")
        return
    
    # Use filtered data if available, otherwise use clean data
    df = st.session_state.get('filtered_df', st.session_state.clean_df)
    
    # Show filter status
    if 'filtered_df' in st.session_state and len(st.session_state.filtered_df) != len(st.session_state.clean_df):
        st.info(f"🔍 Dashboard showing filtered data: {len(df):,} rows (filtered from {len(st.session_state.clean_df):,} total rows)")

    # KPI selection
    st.subheader("📌 Key Performance Indicators")
    # Define available KPI options
    kpi_options = [
        'Row count', 'Column count', 'Missing values',
        'Mean of numeric column', 'Median of numeric column', 'Sum of numeric column',
        'Unique count of categorical column'
    ]
    # Create three selectors for KPI types
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    selections = []
    with col_sel1:
        selections.append(st.selectbox("KPI 1", options=kpi_options, key="kpi1"))
    with col_sel2:
        selections.append(st.selectbox("KPI 2", options=kpi_options, key="kpi2"))
    with col_sel3:
        selections.append(st.selectbox("KPI 3", options=kpi_options, key="kpi3"))

    # Compute and display the KPI cards
    card_cols = st.columns(3)
    for i, sel in enumerate(selections):
        with card_cols[i]:
            if sel == 'Row count':
                value = f"{len(df):,}"
                st.metric("Rows", value)
            elif sel == 'Column count':
                value = f"{df.shape[1]:,}"
                st.metric("Columns", value)
            elif sel == 'Missing values':
                missing_total = int(du.missing_value_summary(df)['missing_count'].sum())
                value = f"{missing_total:,}"
                st.metric("Missing Values", value)
            elif sel in {'Mean of numeric column', 'Median of numeric column', 'Sum of numeric column'}:
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    col_name = st.selectbox(f"Select column for {sel.lower()}", options=num_cols, key=f"kpi_col_{i}")
                    series = df[col_name]
                    if sel.startswith('Mean'):
                        val = series.mean()
                        st.metric(f"Mean of {col_name}", f"{val:.4f}")
                    elif sel.startswith('Median'):
                        val = series.median()
                        st.metric(f"Median of {col_name}", f"{val:.4f}")
                    else:
                        val = series.sum()
                        st.metric(f"Sum of {col_name}", f"{val:.4f}")
                else:
                    st.info("No numeric columns available.")
            elif sel == 'Unique count of categorical column':
                cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                if cat_cols:
                    col_name = st.selectbox("Select categorical column", options=cat_cols, key=f"kpi_cat_{i}")
                    val = df[col_name].nunique()
                    st.metric(f"Unique count of {col_name}", f"{val:,}")
                else:
                    st.info("No categorical columns available.")
            else:
                st.info("Select a KPI from the dropdown above.")

    # Quick correlation matrix with selectable columns
    with st.expander("🔗 Correlation & Latest Model"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect("Select numeric columns for correlation", options=numeric_cols, default=numeric_cols)
            if selected_cols:
                fig = pl.correlation_matrix_subset(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns for correlation analysis.")

        # Show last model metrics if available
        if 'last_metrics' in st.session_state:
            st.subheader("Latest Model Performance")
            metrics = st.session_state.last_metrics
            st.json(metrics)
        else:
            st.info("Train a model to see performance metrics here.")

    # Display any user‑saved charts in a three‑column layout
    if 'dashboard_charts' in st.session_state and st.session_state.dashboard_charts:
        st.subheader("📈 Saved Charts")
        charts = st.session_state.dashboard_charts
        # Create rows of three columns.  Use list slicing to handle arbitrary numbers of charts.
        for i in range(0, len(charts), 3):
            row = st.columns(3)
            for j in range(3):
                idx = i + j
                if idx < len(charts):
                    chart = charts[idx]
                    with row[j]:
                        st.markdown(f"**{chart['name']}**")
                        # Colour picker for each saved chart
                        selected_colour = st.color_picker(
                            "🎨 Choose colour", value="#1f77b4", key=f"dash_colour_{idx}"
                        )
                        fig = chart['figure']
                        # Apply a uniform colour to the saved chart.  Use the
                        # helper from ``utils.plotting`` to recolour all
                        # traces consistently instead of relying on the
                        # layout colourway.
                        pl.apply_single_colour(fig, selected_colour)
                        st.plotly_chart(fig, use_container_width=True)
                        # Remove button
                        if st.button("Remove", key=f"remove_chart_{idx}"):
                            # Remove the selected chart from the list and rerun the app
                            st.session_state.dashboard_charts.pop(idx)
                            st.rerun()
                else:
                    # If no chart for this cell, insert empty placeholder
                    with row[j]:
                        st.empty()