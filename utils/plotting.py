"""
plotting.py
===========

Utilities for creating interactive visualisations with Plotly.

The functions in this module return ``plotly.graph_objects.Figure``
instances so that the calling code has full control over how they are
displayed within Streamlit.  Providing a consistent API for building
plots also makes it easier to extend or customise chart behaviour in
one place.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def histogram(df: pd.DataFrame, column: str, color: Optional[str] = None) -> go.Figure:
    """Create a histogram for a single numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    column : str
        Column name to plot.
    color : str, optional
        Optional categorical column used to colour the bars.  If None,
        all bars share the same colour.

    Returns
    -------
    plotly.graph_objects.Figure
        Histogram figure.
    """
    fig = px.histogram(df, x=column, color=color, marginal="rug", nbins=30)
    fig.update_layout(title=f"Histogram of {column}", bargap=0.1)
    return fig


def boxplot(df: pd.DataFrame, column: str, by: Optional[str] = None) -> go.Figure:
    """Create a boxplot for a numeric column, optionally grouped by another column.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    column : str
        Numeric column to summarise.
    by : str, optional
        Categorical column used to group the data into separate boxes.

    Returns
    -------
    plotly.graph_objects.Figure
        Boxplot figure.
    """
    fig = px.box(df, x=by, y=column, points="all" if by is None else "outliers")
    fig.update_layout(title=f"Boxplot of {column}" + (f" by {by}" if by else ""))
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Create a scatter plot between two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Column for x‑axis.
    y : str
        Column for y‑axis.
    color : str, optional
        Optional column used to colour the points.

    Returns
    -------
    plotly.graph_objects.Figure
        Scatter plot figure.
    """
    fig = px.scatter(df, x=x, y=y, color=color, trendline="ols")
    fig.update_layout(title=f"Scatter plot of {y} vs {x}")
    return fig


def pairplot(df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
    """Create a scatter matrix (pairplot) for multiple numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    columns : list of str, optional
        Subset of numeric columns to include.  If None, all numeric
        columns are used.

    Returns
    -------
    plotly.graph_objects.Figure
        Scatter matrix figure.
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        numeric_cols = columns
    fig = px.scatter_matrix(df[numeric_cols], dimensions=numeric_cols)
    # Set a fixed height so that the matrix scales down gracefully in narrow columns.  Width
    # will be controlled by the container in Streamlit.
    fig.update_layout(title="Scatter Matrix", height=500)
    return fig


def pie(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a pie chart to show category proportions.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    column : str
        Categorical column to summarise.

    Returns
    -------
    plotly.graph_objects.Figure
        Pie chart figure.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.pie(counts, values='count', names=column, title=f"Pie chart of {column}")
    return fig


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Visualise missing values as a heatmap.

    Each cell is coloured based on whether it is missing (1) or not
    (0).  The heatmap can reveal patterns of missingness across the
    dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap figure.
    """
    mask = df.isnull().astype(int)
    fig = px.imshow(mask, aspect='auto', color_continuous_scale=['#ffffff','#FF4136'])
    fig.update_layout(title="Missing Value Heatmap", xaxis_title="Columns", yaxis_title="Rows")
    return fig


def correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Plot the correlation matrix of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    plotly.graph_objects.Figure
        Correlation matrix heatmap.
    """
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale="RdBu_r")
    fig.update_layout(title="Correlation Matrix")
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, aggregation: str = 'sum') -> go.Figure:
    """Create a bar chart for a categorical vs numeric relationship.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Categorical or discrete column for the x‑axis.
    y : str
        Numeric column for the y‑axis.  The data will be aggregated
        using the specified function.
    color : str, optional
        Optional column used to colour the bars.
    aggregation : str, optional
        Aggregation function to apply to ``y``.  One of ``'sum'``,
        ``'mean'`` or ``'count'``.  Defaults to 'sum'.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart figure.
    """
    if aggregation == 'mean':
        agg_df = df.groupby(x)[y].mean().reset_index()
    elif aggregation == 'count':
        agg_df = df.groupby(x)[y].count().reset_index()
    else:
        # default to sum
        agg_df = df.groupby(x)[y].sum().reset_index()
    fig = px.bar(agg_df, x=x, y=y, color=color)
    fig.update_layout(title=f"Bar chart of {y} by {x}")
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Create a line chart for sequential data.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Column for the x‑axis (typically time or ordered category).
    y : str
        Numeric column for the y‑axis.
    color : str, optional
        Optional column used to colour the lines.

    Returns
    -------
    plotly.graph_objects.Figure
        Line chart figure.
    """
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(title=f"Line chart of {y} over {x}")
    return fig


def area_chart(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Create an area chart.

    The area chart is essentially a stacked line chart with the area
    filled.  It is useful for showing cumulative totals over time or
    comparing several groups.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Column for the x‑axis.
    y : str
        Numeric column for the y‑axis.
    color : str, optional
        Optional column used to separate areas by category.

    Returns
    -------
    plotly.graph_objects.Figure
        Area chart figure.
    """
    fig = px.area(df, x=x, y=y, color=color, groupnorm=None)
    fig.update_layout(title=f"Area chart of {y} over {x}")
    return fig


def violin_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None) -> go.Figure:
    """Create a violin plot to show distribution across categories.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Categorical column to group by.
    y : str
        Numeric column whose distribution will be shown.
    color : str, optional
        Optional column used to colour the violins.

    Returns
    -------
    plotly.graph_objects.Figure
        Violin plot figure.
    """
    fig = px.violin(df, x=x, y=y, color=color, box=True, points='all')
    fig.update_layout(title=f"Violin plot of {y} by {x}")
    return fig


def density_heatmap(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Create a two‑dimensional density heatmap for two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    x : str
        Numeric column for the x‑axis.
    y : str
        Numeric column for the y‑axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Density heatmap figure.
    """
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=30, nbinsy=30, color_continuous_scale='Viridis')
    fig.update_layout(title=f"Density heatmap of {y} vs {x}")
    return fig


def spider_chart(df: pd.DataFrame, value_cols: List[str], category_col: Optional[str] = None, aggregation: str = 'mean') -> go.Figure:
    """Create a radar (spider) chart.

    A spider chart visualises multiple quantitative variables on axes
    starting from the same point.  When a categorical column is
    provided, one polygon is drawn per category using the specified
    aggregation function across the value columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    value_cols : list of str
        Numeric columns to plot on the radial axes.
    category_col : str, optional
        Optional column used to group the data.  Each unique value
        in this column will result in a separate trace.
    aggregation : str, optional
        Aggregation function applied when grouping by ``category_col``.
        One of ``'mean'`` or ``'sum'``.  Defaults to 'mean'.

    Returns
    -------
    plotly.graph_objects.Figure
        Radar chart figure.
    """
    import plotly.graph_objects as go  # imported here to avoid circular import
    categories = value_cols
    fig = go.Figure()
    if category_col:
        groups = df[category_col].dropna().unique()
        for g in groups:
            subset = df[df[category_col] == g][value_cols]
            if aggregation == 'sum':
                values = subset.sum()
            else:
                values = subset.mean()
            fig.add_trace(go.Scatterpolar(r=values.values, theta=categories, fill='toself', name=str(g)))
    else:
        # Use the first row of the numeric columns if no category provided
        if not df[value_cols].empty:
            values = df[value_cols].iloc[0]
            fig.add_trace(go.Scatterpolar(r=values.values, theta=categories, fill='toself', name='Values'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Spider Chart"
    )
    return fig


def apply_single_colour(fig: go.Figure, colour: str) -> go.Figure:
    """Apply a uniform colour to all traces in a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure whose traces should be recoloured.
    colour : str
        Any valid CSS colour string (hex, rgb, etc.).

    Returns
    -------
    plotly.graph_objects.Figure
        The modified figure (mutated in place).  The return is for
        convenience allowing chaining.

    Notes
    -----
    Plotly's ``update_layout`` ``colorway`` only affects the colour of
    *newly created* traces and does not recolour existing traces.  To
    consistently recolour traces after they have been constructed, we
    update their marker, marker line, line and fill attributes.  Not
    all trace types expose all of these attributes; unsupported
    properties are silently ignored.
    """
    try:
        # These update calls work on most trace types.  If any of the
        # keyword arguments are unsupported they will raise an exception
        # which we catch below.
        fig.update_traces(
            marker_color=colour,
            marker_line_color=colour,
            line_color=colour,
            fillcolor=colour,
            selector=dict()
        )
    except Exception:
        # Fall back to manually iterating through traces when the bulk
        # update fails (e.g. older Plotly versions).  We attempt
        # individual attribute updates and ignore unsupported ones.
        for trace in fig.data:
            try:
                trace.update(marker=dict(color=colour))
            except Exception:
                pass
            # Update marker line colour if present
            try:
                trace.update(marker_line_color=colour)
            except Exception:
                pass
            try:
                trace.update(line=dict(color=colour))
            except Exception:
                pass
            if hasattr(trace, 'fillcolor'):
                try:
                    trace.update(fillcolor=colour)
                except Exception:
                    pass
    return fig


def correlation_matrix_subset(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Plot the correlation matrix for a subset of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Numeric columns to include in the correlation matrix.

    Returns
    -------
    plotly.graph_objects.Figure
        Correlation matrix heatmap for the selected columns.
    """
    subset = df[columns].select_dtypes(include=['number'])
    corr = subset.corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale="RdBu_r")
    fig.update_layout(title=f"Correlation Matrix for {', '.join(columns)}")
    return fig