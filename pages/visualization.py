"""
visualization.py
================

The visualisation dashboard exposes a suite of interactive charts
powered by Plotly.  Users can choose different chart types, select
columns to plot and customise basic appearance settings.  Charts are
rendered in the main area with full interactivity, including zoom
and hover tooltips.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils import plotting as pl


def show_visualization_page() -> None:
    """Render the visualisation dashboard."""
    st.header("📊 Visualisation Dashboard")
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please load and prepare a dataset on the Data page first.")
        return
    
    # Use filtered data if available, otherwise use clean data
    df = st.session_state.get('filtered_df', st.session_state.clean_df)
    
    # Show filter status
    if 'filtered_df' in st.session_state and len(st.session_state.filtered_df) != len(st.session_state.clean_df):
        st.info(f"🔍 Visualizing filtered data: {len(df):,} rows (filtered from {len(st.session_state.clean_df):,} total rows)")

    # Chart selection
    chart_type = st.selectbox(
        "Select chart type",
        options=[
            'Histogram', 'Boxplot', 'Scatter', 'Pairplot', 'Pie Chart',
            'Bar Chart', 'Line Chart', 'Area Chart', 'Violin Plot',
            'Density Heatmap', 'Spider Chart'
        ]
    )

    fig = None

    if chart_type == 'Histogram':
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        x_col = st.selectbox("Select numeric column", options=numeric_cols)
        color_col = st.selectbox("Optional colour by", options=[None] + cat_cols)
        if x_col:
            fig = pl.histogram(df, x_col, color=color_col)

    elif chart_type == 'Boxplot':
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        y_col = st.selectbox("Select numeric column", options=numeric_cols, key="box_y")
        group_col = st.selectbox("Group by (optional)", options=[None] + cat_cols, key="box_group")
        if y_col:
            fig = pl.boxplot(df, y_col, by=group_col)

    elif chart_type == 'Scatter':
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        x_col = st.selectbox("X axis", options=numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y axis", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
        color_col = st.selectbox("Colour by (optional)", options=[None] + df.columns.tolist(), key="scatter_color")
        if x_col and y_col:
            fig = pl.scatter(df, x_col, y_col, color=color_col if color_col is not None else None)

    elif chart_type == 'Pairplot':
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        selected_cols = st.multiselect("Select numeric columns", options=numeric_cols, default=numeric_cols[:5])
        if selected_cols:
            fig = pl.pairplot(df, columns=selected_cols)

    elif chart_type == 'Pie Chart':
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns available for a pie chart.")
        else:
            pie_col = st.selectbox("Select categorical column", options=cat_cols)
            fig = pl.pie(df, pie_col)

    # Additional chart types
    elif chart_type == 'Bar Chart':
        # For bar charts we need a categorical x and numeric y
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not cat_cols or not num_cols:
            st.warning("Bar chart requires at least one categorical and one numeric column.")
        else:
            x_col = st.selectbox("Categorical (x-axis)", options=cat_cols, key="bar_x")
            y_col = st.selectbox("Numeric (y-axis)", options=num_cols, key="bar_y")
            agg_method = st.selectbox("Aggregation", options=['sum', 'mean', 'count'], key="bar_agg")
            color_col = st.selectbox("Colour by (optional)", options=[None] + cat_cols, key="bar_color")
            if x_col and y_col:
                fig = pl.bar_chart(df, x_col, y_col, color=color_col, aggregation=agg_method)

    elif chart_type == 'Line Chart':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        x_col = st.selectbox("X axis", options=all_cols, key="line_x")
        y_col = st.selectbox("Y axis", options=num_cols, key="line_y")
        color_col = st.selectbox("Colour by (optional)", options=[None] + all_cols, key="line_color")
        if x_col and y_col:
            fig = pl.line_chart(df, x_col, y_col, color=color_col)

    elif chart_type == 'Area Chart':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        x_col = st.selectbox("X axis", options=all_cols, key="area_x")
        y_col = st.selectbox("Y axis", options=num_cols, key="area_y")
        color_col = st.selectbox("Colour by (optional)", options=[None] + all_cols, key="area_color")
        if x_col and y_col:
            fig = pl.area_chart(df, x_col, y_col, color=color_col)

    elif chart_type == 'Violin Plot':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if not num_cols or not cat_cols:
            st.warning("Violin plot requires one numeric and one categorical column.")
        else:
            y_col = st.selectbox("Numeric (y-axis)", options=num_cols, key="violin_y")
            x_col = st.selectbox("Categorical (x-axis)", options=cat_cols, key="violin_x")
            color_col = st.selectbox("Colour by (optional)", options=[None] + cat_cols, key="violin_color")
            fig = pl.violin_plot(df, x_col, y_col, color=color_col)

    elif chart_type == 'Density Heatmap':
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Density heatmap requires at least two numeric columns.")
        else:
            x_col = st.selectbox("X axis", options=num_cols, key="heatmap_x")
            y_col = st.selectbox("Y axis", options=[c for c in num_cols if c != x_col], key="heatmap_y")
            fig = pl.density_heatmap(df, x_col, y_col)

    # Spider chart (radar chart)
    elif chart_type == 'Spider Chart':
        # Require at least three numeric columns for a meaningful spider chart
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Spider chart requires two or more numeric columns.")
        else:
            selected_vals = st.multiselect(
                "Select numeric columns for axes",
                options=num_cols,
                default=num_cols[:min(5, len(num_cols))],
                key="spider_values"
            )
            cat_col = st.selectbox(
                "Group by (optional)",
                options=[None] + cat_cols,
                key="spider_cat"
            )
            agg_method = st.selectbox(
                "Aggregation method",
                options=['mean', 'sum'],
                key="spider_agg"
            )
            if selected_vals:
                fig = pl.spider_chart(df, value_cols=selected_vals, category_col=cat_col, aggregation=agg_method)

    # Display colour picker, chart and save option
    if fig is not None:
        # Colour picker allows customising the palette.  We use a unique key per chart type so that
        # switching chart types preserves independent colour selections.  The default colour is
        # drawn from Plotly's first category colour.
        default_colour = "#1f77b4"
        selected_colour = st.color_picker(
            "🎨 Pick a colour for the chart",
            value=default_colour,
            key=f"colour_picker_{chart_type}"
        )
        # Update the figure's colour palette
        # Apply the selected colour to the chart.  Colourway does not update
        # existing traces, so we delegate to a helper in the plotting module
        # that updates marker, line and fill colours as appropriate.
        pl.apply_single_colour(fig, selected_colour)

        st.plotly_chart(fig, use_container_width=True)

        # Allow the user to save the chart to the dashboard
        # Provide a name for the chart; default to the chart type
        chart_name = st.text_input(
            "Chart name",
            value=f"{chart_type}",
            key=f"chart_name_{chart_type}"
        )
        if st.button("💾 Save to Dashboard", key=f"save_dash_{chart_type}"):
            # Initialise the dashboard charts list in session state
            if 'dashboard_charts' not in st.session_state:
                st.session_state.dashboard_charts = []
            import copy  # local import to avoid overhead at module import time
            # Append a deep copy of the current figure and the chosen name.  Copying
            # avoids modifications to the original figure (e.g. colour changes) from
            # propagating across views.
            fig_copy = copy.deepcopy(fig)
            st.session_state.dashboard_charts.append({'name': chart_name, 'figure': fig_copy})
            st.success(f"Chart '{chart_name}' saved to dashboard.")