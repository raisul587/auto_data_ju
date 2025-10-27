"""
data.py
=======

This module implements the data management page.  Users can upload
local files, connect to SQL databases, clean and preprocess their
datasets and export the results.  The aim is to offer a one‑stop
environment for ingesting and preparing data before analysis.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

# Use absolute import to avoid relative import issues when executing app.py as a script
# Use relative import based on project root.  When app.py is executed as a script,
# the current directory is automatically added to sys.path, allowing us to
# import modules from sibling packages such as `utils`.
from utils import data_utils as du


def show_data_page() -> None:
    """Render the data management page."""
    st.header("📁 Dataset Management")
    st.write(
        """Use this page to clean and transform your loaded dataset.  The file
        upload and SQL import have been moved to the Home page.  Here you
        can rename columns, change data types, handle missing values,
        inspect duplicates and treat outliers."""
    )

    # Ensure a dataset is loaded
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please upload or import data on the Home page first.")
        return

    # Display and transform data
    # Use the filtered dataframe if filters are applied in the sidebar
    df = st.session_state.get('filtered_df', st.session_state.clean_df)
    st.subheader("Preview Data")
    if 'filtered_df' in st.session_state and len(df) != len(st.session_state.clean_df):
        st.info(f"Previewing filtered data: {len(df):,} rows (of {len(st.session_state.clean_df):,})")
    st.dataframe(df.head(100), use_container_width=True)

    # Column operations
    with st.expander("🛠 Rename Columns / Change Types"):
        col1, col2 = st.columns(2)
        with col1:
            col_to_rename = st.selectbox("Select column to rename", options=df.columns.tolist())
            new_name = st.text_input("New column name", key="rename_input")
            if st.button("Rename Column"):
                if new_name:
                    st.session_state.clean_df = du.rename_columns(st.session_state.clean_df, {col_to_rename: new_name})
                    # Update base and filtered datasets to reflect column rename
                    st.session_state.df = st.session_state.clean_df.copy()
                    # Assign a fresh copy for filtered_df to ensure filters operate on the new data
                    st.session_state.filtered_df = st.session_state.clean_df.copy()
                    # Persist or delete cached data based on user preference
                    if st.session_state.get('persist_data', False):
                        du.save_cached_dataset(st.session_state.clean_df)
                    else:
                        du.delete_cached_dataset()
                    st.success(f"Renamed column {col_to_rename} to {new_name}.")
                    st.rerun()
                else:
                    st.warning("Please enter a new name.")
        with col2:
            col_to_cast = st.selectbox("Select column to change type", options=df.columns.tolist(), key="dtype_select")
            # Allow a few common numeric targets including explicit 64‑bit types.  When converting
            # datetime columns to integers the system will downcast appropriately.
            dtype = st.selectbox(
                "Target dtype",
                options=['float', 'float64', 'int', 'int64', 'object', 'category', 'bool'],
                key="dtype_target"
            )
            if st.button("Change Type"):
                try:
                    st.session_state.clean_df = du.change_dtypes(st.session_state.clean_df, {col_to_cast: dtype})
                    # Update base and filtered datasets after dtype change
                    st.session_state.df = st.session_state.clean_df.copy()
                    st.session_state.filtered_df = st.session_state.clean_df.copy()
                    # Persist or delete cached data based on user preference
                    if st.session_state.get('persist_data', False):
                        du.save_cached_dataset(st.session_state.clean_df)
                    else:
                        du.delete_cached_dataset()
                    st.success(f"Changed type of {col_to_cast} to {dtype}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to convert column: {e}")

    # Missing values
    with st.expander("🚨 Missing Values"):
        msum = du.missing_value_summary(st.session_state.clean_df)
        st.dataframe(msum, hide_index=True)
        strategy = st.selectbox("Handling strategy", options=['mean', 'median', 'mode', 'constant', 'drop'])
        const_value = None
        if strategy == 'constant':
            const_value = st.text_input("Fill value", value="0")
        if st.button("Apply Missing Value Strategy"):
            try:
                st.session_state.clean_df = du.handle_missing_values(st.session_state.clean_df, strategy=strategy, fill_value=const_value)
                # Update base and filtered datasets after missing value handling
                st.session_state.df = st.session_state.clean_df.copy()
                st.session_state.filtered_df = st.session_state.clean_df.copy()
                if st.session_state.get('persist_data', False):
                    du.save_cached_dataset(st.session_state.clean_df)
                else:
                    du.delete_cached_dataset()
                st.success("Missing values handled successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error handling missing values: {e}")

    # Duplicates
    with st.expander("📑 Duplicates"):
        dup_count, dup_rows = du.duplicate_summary(st.session_state.clean_df)
        st.write(f"Found {dup_count} duplicate rows.")
        if dup_count > 0:
            if st.checkbox("Show duplicate rows"):
                st.dataframe(dup_rows, use_container_width=True)
            if st.button("Drop duplicates"):
                st.session_state.clean_df = du.drop_duplicates(st.session_state.clean_df)
                # Update base and filtered datasets after removing duplicates
                st.session_state.df = st.session_state.clean_df.copy()
                st.session_state.filtered_df = st.session_state.clean_df.copy()
                if st.session_state.get('persist_data', False):
                    du.save_cached_dataset(st.session_state.clean_df)
                else:
                    du.delete_cached_dataset()
                st.success("Duplicate rows removed.")
                st.rerun()

    # Outlier handling
    with st.expander("🎯 Outlier Detection & Treatment"):
        numeric_cols = st.session_state.clean_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns available for outlier detection.")
        else:
            selected_cols = st.multiselect("Select numeric columns to analyse", options=numeric_cols, default=numeric_cols)
            if st.button("Detect Outliers"):
                summary = du.detect_outliers_iqr(st.session_state.clean_df, columns=selected_cols)
                st.dataframe(summary.round(2))
            if st.button("Remove Outliers"):
                st.session_state.clean_df = du.remove_outliers_iqr(st.session_state.clean_df, columns=selected_cols)
                # Update base and filtered datasets after outlier removal
                st.session_state.df = st.session_state.clean_df.copy()
                st.session_state.filtered_df = st.session_state.clean_df.copy()
                if st.session_state.get('persist_data', False):
                    du.save_cached_dataset(st.session_state.clean_df)
                else:
                    du.delete_cached_dataset()
                st.success("Outlier rows removed using IQR method.")
                st.rerun()

    # Download cleaned dataset
    csv = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv", mime="text/csv")

    # Drop columns section
    with st.expander("🗑️ Drop Columns"):
        drop_cols = st.multiselect(
            "Select columns to remove", options=st.session_state.clean_df.columns.tolist(), key="drop_columns_select"
        )
        if drop_cols:
            if st.button("Remove Selected Columns", key="drop_columns_button"):
                st.session_state.clean_df = st.session_state.clean_df.drop(columns=drop_cols).reset_index(drop=True)
                # Update base and filtered datasets after dropping columns
                st.session_state.df = st.session_state.clean_df.copy()
                st.session_state.filtered_df = st.session_state.clean_df.copy()
                if st.session_state.get('persist_data', False):
                    du.save_cached_dataset(st.session_state.clean_df)
                else:
                    du.delete_cached_dataset()
                st.success(f"Removed columns: {', '.join(drop_cols)}")
                st.rerun()