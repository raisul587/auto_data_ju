"""
home.py
=======

The home page introduces users to the automated data analysis
application.  It provides a high‑level overview of the available
functionality and guides first‑time users towards uploading their
data.  This page lives separately from the rest of the app so that
Streamlit's multi‑page architecture can route here when the user
selects "Home" from the navigation menu.
"""

import streamlit as st
import pandas as pd

from utils import data_utils as du


def show_home() -> None:
    """Render the home page with dataset upload and overview."""
    st.title("📊 Automated Data Analysis Platform")
    st.markdown(
        """
        Welcome to the automated data analysis platform.  This tool is
        designed to take you from raw data to insightful models with
        minimal effort.  Upload your dataset below to get started or
        continue working on your last session.  Once your data is loaded
        you can explore, visualise and model it using the other tabs.

        **Key highlights**

        - Supports CSV, Excel and SQL data sources.
        - Provides interactive summaries and visualisations.
        - Includes feature engineering and advanced analytics tools.
        - Trains multiple models and compares their performance.
        - Exports cleaned data, visualisations and reports.
        """
    )

    # Allow the user to control whether the uploaded dataset should be persisted across sessions.
    # Caching large datasets can be time consuming.  If unchecked, the data will only live
    # in memory for the current session.
    # Default ``persist_data`` to True so that uploaded data is retained across
    # browser refreshes.  Users can opt out by unchecking the box.  Persisting
    # data avoids the situation where the dataset disappears after a page reload.
    st.session_state.persist_data = st.checkbox(
        "Keep dataset across sessions (disable for faster initial loading)",
        value=st.session_state.get('persist_data', True),
        help="If enabled, the dataset is saved to disk so it is available next time you open the app."
    )

    # Attempt to load a previously cached dataset on first render.  If the user
    # has disabled persistence we remove any existing cache to avoid stale
    # data being re‑loaded.
    if 'cache_checked' not in st.session_state:
        if st.session_state.get('persist_data', True):
            cached = du.load_cached_dataset()
            if cached is not None:
                st.session_state.df = cached.copy()
                st.session_state.clean_df = cached.copy()
                st.session_state.filtered_df = cached.copy()
        else:
            du.delete_cached_dataset()
        st.session_state.cache_checked = True

    # Initialise session state placeholders
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'clean_df' not in st.session_state:
        st.session_state.clean_df = pd.DataFrame()

    st.subheader("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"], key="home_file_upload")
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            df = du.load_data(uploaded_file, file_type)
            # Assign copies of the uploaded data to avoid accidental inplace modifications.
            st.session_state.df = df.copy()
            st.session_state.clean_df = df.copy()
            st.session_state.filtered_df = df.copy()
            # Persist dataset immediately if user opted in; otherwise remove any cached data
            if st.session_state.persist_data:
                du.save_cached_dataset(df)
            else:
                du.delete_cached_dataset()
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            st.info("Dataset loaded. Click the button below to enable filters for this session.")
            if st.button("Apply dataset", key="apply_dataset_file"):
                # Use the public rerun API introduced in Streamlit 1.25+.  The
                # previous experimental version was removed in newer versions.
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load data: {e}")

    with st.expander("🔌 SQL Import"):
        st.markdown(
            """Enter a SQLAlchemy connection string and a query to import data from a
            database.  For example: `postgresql+psycopg2://user:pass@host:port/dbname`."""
        )
        connection_string = st.text_input("Connection String", value="", key="home_sql_conn")
        query = st.text_area("SQL Query", value="SELECT * FROM table_name LIMIT 100", key="home_sql_query")
        if st.button("Run Query", key="home_sql_run"):
            if connection_string and query:
                try:
                    df_sql = du.load_sql(connection_string, query)
                    # Assign copies of the query result to avoid accidental inplace modifications
                    st.session_state.df = df_sql.copy()
                    st.session_state.clean_df = df_sql.copy()
                    st.session_state.filtered_df = df_sql.copy()
                    if st.session_state.persist_data:
                        du.save_cached_dataset(df_sql)
                    else:
                        du.delete_cached_dataset()
                    st.success(f"Query returned {df_sql.shape[0]} rows and {df_sql.shape[1]} columns.")
                    st.info("Query executed. Click the button below to enable filters for this session.")
                    if st.button("Apply query", key="apply_dataset_sql"):
                        st.rerun()
                except Exception as e:
                    st.error(f"Error executing SQL: {e}")
            else:
                st.warning("Please provide both a connection string and a query.")

    # Display dataset preview if available
    if not st.session_state.clean_df.empty:
        st.subheader("🧾 Dataset Preview")
        # Use the filtered DataFrame if filters are active, otherwise fall back to the clean data
        preview_df = st.session_state.get('filtered_df', st.session_state.clean_df)
        # Show a message when previewing a filtered subset
        if ('filtered_df' in st.session_state and
                len(st.session_state.filtered_df) != len(st.session_state.clean_df)):
            st.info(
                f"Previewing filtered data: {len(preview_df):,} rows (of {len(st.session_state.clean_df):,})"
            )
        st.dataframe(preview_df.head(100), use_container_width=True)