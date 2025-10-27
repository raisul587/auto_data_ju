"""
exploration.py
==============

This module defines the exploratory data analysis page.  Users can
inspect summary statistics for numeric and categorical variables,
visualise missing data patterns and explore relationships between
variables via correlation matrices and outlier detection.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Absolute imports to avoid relative import errors when run as a script
from utils import data_utils as du
from utils import plotting as pl


def show_exploration_page() -> None:
    """Render the exploration & summary page."""
    st.header("🔍 Exploration & Summaries")
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please upload and clean a dataset on the Data page first.")
        return

    # Use filtered data if available, otherwise use clean data
    df = st.session_state.get('filtered_df', st.session_state.clean_df)
    
    # Show filter status
    if 'filtered_df' in st.session_state and len(st.session_state.filtered_df) != len(st.session_state.clean_df):
        st.info(f"🔍 Analyzing filtered data: {len(df):,} rows (filtered from {len(st.session_state.clean_df):,} total rows)")

    # Numeric summary
    with st.expander("📈 Numeric Summary"):
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            st.warning("No numeric columns available.")
        else:
            summary = numeric_df.describe().T
            # Add skewness and kurtosis
            summary['skew'] = numeric_df.skew()
            summary['kurtosis'] = numeric_df.kurtosis()
            st.dataframe(summary.round(4))

    # Categorical summary
    with st.expander("🔤 Categorical Summary"):
        cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
        if cat_df.empty:
            st.warning("No categorical columns available.")
        else:
            summary_rows = []
            for col in cat_df.columns:
                counts = cat_df[col].value_counts(dropna=False)
                top = counts.index[0] if not counts.empty else None
                top_count = counts.iloc[0] if not counts.empty else None
                summary_rows.append({
                    'column': col,
                    'unique_values': cat_df[col].nunique(),
                    'most_frequent': top,
                    'freq_count': top_count
                })
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df)

    # Missing value summary and heatmap
    with st.expander("🚨 Missing Data Analysis"):
        msum = du.missing_value_summary(df)
        st.dataframe(msum, hide_index=True)
        heatmap_fig = pl.missing_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Correlation matrix and outliers
    with st.expander("🔗 Correlation & Outliers"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            # Allow user to select one or more columns for correlation analysis
            st.write("Select numeric columns to compute correlations:")
            corr_cols = st.multiselect("Columns", options=numeric_cols, default=numeric_cols, key="corr_cols")
            if corr_cols:
                # If two or more columns selected, show correlation matrix
                if len(corr_cols) >= 2:
                    corr_fig = pl.correlation_matrix_subset(df, columns=corr_cols)
                    st.plotly_chart(corr_fig, use_container_width=True)
                # If exactly two columns, display the correlation coefficient
                if len(corr_cols) == 2:
                    c1, c2 = corr_cols
                    corr_val = df[c1].corr(df[c2])
                    st.info(f"Correlation between **{c1}** and **{c2}**: {corr_val:.4f}")
        else:
            st.warning("Correlation analysis requires numeric columns.")

        # Outlier summary (non‑destructive)
        if st.checkbox("Show outlier summary (IQR method)"):
            summary = du.detect_outliers_iqr(df, columns=numeric_cols)
            st.dataframe(summary.round(2))