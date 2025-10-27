"""
app.py
======

Entry point for the Streamlit data analysis application.  This file
defines the overall layout, sets up session state, handles theme
toggling and routes between the various functional pages (home,
data, exploration, visualisation, modelling and dashboard).

To run the app locally, execute ``streamlit run app.py`` from the
root of the project.  Ensure that all required dependencies are
installed (see ``requirements.txt``).
"""

from __future__ import annotations

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import datetime, timedelta

from utils import data_utils as du

# Import page modules.  When app.py is executed directly, Python adds the
# current working directory to sys.path so that packages like `pages` can
# be imported relative to this file.  Using absolute imports with the
# package name would require installing the package or renaming the
# parent folder to match the package name.  Importing modules from
# `pages` directly is therefore more robust in user environments.
import pages.home as home
import pages.data as data
import pages.exploration as exploration
import pages.visualization as visualization
import pages.modeling as modeling
import pages.dashboard as dashboard

# Import filter utilities
from utils import filter_utils as fu

import threading


def set_theme(theme: str) -> None:
    """Apply a simple theme.  The app now defaults to a light theme without user selection."""
    css = """
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main() -> None:
    """Main entry point for the Streamlit app."""
    # Configure page
    st.set_page_config(page_title="Automated Data Analysis", layout="wide", initial_sidebar_state="auto")

    # Attempt to load cached dataset on startup if not already loaded.  The
    # ``load_cached_dataset`` helper returns ``None`` if no cached data is
    # available or if an error occurs.  If a cached dataset exists, we
    # populate both the clean and base DataFrames.  This ensures that
    # previously prepared data is immediately available when the app is
    # reopened without requiring the user to re‑upload their dataset.
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        cached_df = du.load_cached_dataset()
        if cached_df is not None:
            st.session_state.clean_df = cached_df.copy()
            st.session_state.df = cached_df.copy()
            st.session_state.filtered_df = cached_df.copy()

    # Initialise a list to store charts saved from the visualisation page.  This ensures
    # the dashboard page can iterate over the collection without KeyError.  Only
    # initialise if not already present to preserve saved charts on reruns.
    if 'dashboard_charts' not in st.session_state:
        st.session_state.dashboard_charts = []

    # Sidebar for global filtering
    with st.sidebar:
        st.title("🔍 Global Filters")
        # Check if data is loaded
        if 'clean_df' in st.session_state and not st.session_state.clean_df.empty:
            df = st.session_state.clean_df.copy()
            original_count = len(df)

            # Initialize filtered_df in session state
            if 'filtered_df' not in st.session_state:
                st.session_state.filtered_df = df.copy()

            # Detect column types
            col_types = fu.detect_column_types(df)

            # SQL Query Filter
            st.subheader("🧮 SQL Filter")
            sql_query = st.text_area(
                "Enter a SELECT query to filter the data (e.g. SELECT * FROM df WHERE age > 30)",
                value=st.session_state.get('global_sql', ''),
                key='global_sql_input',
                help="Only SELECT statements are allowed; reference the table as 'df'"
            )
            if sql_query != st.session_state.get('global_sql', ''):
                st.session_state.global_sql = sql_query
            # SQL filter enable toggle.  Uncheck to temporarily remove the SQL filter without deleting the query.
            sql_enabled = st.checkbox("Apply SQL filter", value=bool(sql_query), key="sql_enabled")
            if sql_query and sql_enabled:
                df = fu.apply_sql_query(df, sql_query)

            st.markdown("---")

            # Date Filters
            if col_types['datetime']:
                st.subheader("📅 Date Filters")
                for date_col in col_types['datetime']:
                    with st.expander(f"📆 {date_col}"):
                        min_date = df[date_col].min()
                        max_date = df[date_col].max()
                        if pd.notna(min_date) and pd.notna(max_date):
                            min_date = min_date.date() if hasattr(min_date, 'date') else min_date
                            max_date = max_date.date() if hasattr(max_date, 'date') else max_date
                            date_range = st.date_input(
                                "Select date range",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key=f"date_filter_{date_col}"
                            )
                            enable_date = st.checkbox(
                                "Apply filter", value=True, key=f"date_enable_{date_col}"
                            )
                            if enable_date and isinstance(date_range, tuple) and len(date_range) == 2:
                                df = fu.apply_datetime_filter(df, date_col, date_range[0], date_range[1])
                st.markdown("---")

            # Numeric Filters
            if col_types['numeric']:
                st.subheader("🔢 Numeric Filters")
                numeric_filter_cols = st.multiselect(
                    "Select columns to filter",
                    options=col_types['numeric'],
                    key='numeric_filter_selection'
                )
                for num_col in numeric_filter_cols:
                    with st.expander(f"📊 {num_col}"):
                        min_val = float(df[num_col].min())
                        max_val = float(df[num_col].max())
                        if min_val != max_val:
                            range_vals = st.slider(
                                "Range",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"numeric_filter_{num_col}"
                            )
                            enable_numeric = st.checkbox(
                                "Apply filter", value=True, key=f"numeric_enable_{num_col}"
                            )
                            if enable_numeric:
                                df = fu.apply_numeric_filter(df, num_col, range_vals[0], range_vals[1])
                st.markdown("---")

            # Categorical Filters
            if col_types['categorical']:
                st.subheader("🏷️ Categorical Filters")
                cat_filter_cols = st.multiselect(
                    "Select columns to filter",
                    options=col_types['categorical'],
                    key='cat_filter_selection'
                )
                for cat_col in cat_filter_cols:
                    unique_vals = df[cat_col].dropna().unique().tolist()
                    if len(unique_vals) <= 50:
                        with st.expander(f"🔖 {cat_col}"):
                            selected_vals = st.multiselect(
                                "Select values",
                                options=sorted(unique_vals, key=str),
                                default=unique_vals,
                                key=f"cat_filter_{cat_col}"
                            )
                            enable_cat = st.checkbox(
                                "Apply filter", value=True, key=f"cat_enable_{cat_col}"
                            )
                            if enable_cat:
                                df = fu.apply_categorical_filter(df, cat_col, selected_vals)
                st.markdown("---")

            # Boolean Filters
            if col_types['boolean']:
                st.subheader("✓ Boolean Filters")
                for bool_col in col_types['boolean']:
                    bool_val = st.radio(
                        f"{bool_col}",
                        options=["All", "True", "False"],
                        index=0,
                        key=f"bool_filter_{bool_col}",
                        horizontal=True
                    )
                    enable_bool = st.checkbox(
                        "Apply filter", value=True, key=f"bool_enable_{bool_col}"
                    )
                    if enable_bool and bool_val != "All":
                        df = fu.apply_boolean_filter(df, bool_col, bool_val == "True")
                st.markdown("---")

            # Update filtered dataframe
            st.session_state.filtered_df = df
            filtered_count = len(df)
            summary = fu.get_filter_summary(original_count, filtered_count)
            st.info(summary)

            # Reset filters
            if st.button("🔄 Reset All Filters", use_container_width=True):
                # Clear all filter‑related keys from session state
                keys_to_clear = [
                    key for key in st.session_state.keys()
                    if key.startswith('date_filter_') or key.startswith('date_enable_')
                    or key.startswith('numeric_filter_') or key.startswith('numeric_enable_')
                    or key.startswith('numeric_filter_selection')
                    or key.startswith('cat_filter_') or key.startswith('cat_enable_')
                    or key.startswith('cat_filter_selection')
                    or key.startswith('bool_filter_') or key.startswith('bool_enable_')
                    or key == 'sql_enabled'
                ]
                for k in keys_to_clear:
                    try:
                        del st.session_state[k]
                    except KeyError:
                        pass
                # Reset filtered data and global SQL query
                st.session_state.filtered_df = st.session_state.clean_df.copy()
                st.session_state.global_sql = ''
                st.rerun()
        else:
            st.info("📂 Load data from the Home page to enable filtering")
            st.markdown("---")
            st.markdown(
                """
                **Available Filters:**
                - 🧮 SQL query (SELECT) to filter data
                - 📅 Date range filters
                - 🔢 Numeric range sliders
                - 🏷️ Categorical multiselect
                - ✓ Boolean toggles
                """
            )

    # Apply light theme CSS by default
    set_theme('light')

    # Navigation menu
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data", "Exploration", "Visualization", "ML", "Dashboard"],
        icons=["house", "database", "bar-chart", "graph-up", "cpu", "speedometer"],
        default_index=0,
        orientation="horizontal",
        styles={
            'container': {'padding': '0!important', 'background-color': 'rgba(0,0,0,0)'},
            'icon': {'color': 'orange', 'font-size': '18px'},
            'nav-link': {'font-size': '16px', 'text-align': 'center', 'margin':'0px', '--hover-color': '#eee'},
            'nav-link-selected': {'background-color': '#fdc500', 'color': 'black'},
        }
    )

    # Route to appropriate page
    if selected == "Home":
        home.show_home()
    elif selected == "Data":
        data.show_data_page()
    elif selected == "Exploration":
        exploration.show_exploration_page()
    elif selected == "Visualization":
        visualization.show_visualization_page()
    elif selected == "ML":
        modeling.show_modeling_page()
    elif selected == "Dashboard":
        dashboard.show_dashboard_page()


if __name__ == '__main__':
    main()