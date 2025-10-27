"""
filter_utils.py
===============

Utility functions for intelligent global filtering across the application.
Automatically detects column types and applies appropriate filters.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, date

import pandas as pd
import numpy as np
import streamlit as st
import sqlite3


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize columns by their data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    dict
        Dictionary with keys: 'numeric', 'categorical', 'datetime', 'boolean'
        Each containing a list of column names.
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'boolean': []
    }
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            column_types['boolean'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numeric'].append(col)
        else:
            # Categorical or object type
            column_types['categorical'].append(col)
    
    return column_types


def apply_numeric_filter(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> pd.DataFrame:
    """Filter DataFrame by numeric range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to filter.
    min_val : float
        Minimum value (inclusive).
    max_val : float
        Maximum value (inclusive).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[(df[column] >= min_val) & (df[column] <= max_val)]


def apply_categorical_filter(df: pd.DataFrame, column: str, selected_values: List[Any]) -> pd.DataFrame:
    """Filter DataFrame by categorical values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to filter.
    selected_values : list
        List of values to keep.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if not selected_values:
        return df
    return df[df[column].isin(selected_values)]


def apply_datetime_filter(df: pd.DataFrame, column: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter DataFrame by date range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to filter.
    start_date : date
        Start date (inclusive).
    end_date : date
        End date (inclusive).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    # Convert column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    
    # Convert dates to datetime for comparison
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    return df[(df[column] >= start_datetime) & (df[column] <= end_datetime)]


def apply_boolean_filter(df: pd.DataFrame, column: str, value: Optional[bool]) -> pd.DataFrame:
    """Filter DataFrame by boolean value.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to filter.
    value : bool or None
        Boolean value to filter by. If None, no filter is applied.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if value is None:
        return df
    return df[df[column] == value]


def apply_text_search(df: pd.DataFrame, search_term: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply text search across specified columns or all columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    search_term : str
        Text to search for (case-insensitive).
    columns : list of str, optional
        Columns to search in. If None, searches all columns.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if not search_term:
        return df
    
    if columns is None:
        columns = df.columns.tolist()
    
    mask = df[columns].apply(
        lambda row: row.astype(str).str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    
    return df[mask]


def apply_sql_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Execute a SQL query against the in‑memory representation of a DataFrame.

    This helper creates a temporary SQLite database, loads the DataFrame
    into a table called ``df`` and then executes the supplied SQL
    statement.  The query must be a valid SQLite SELECT statement.
    If the query is invalid or empty, the original DataFrame is
    returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  The column names become SQL field names.
    query : str
        SQL query string.  Should reference the table ``df``.  Example:
        ``SELECT * FROM df WHERE age > 30``.

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame from the SQL query.  If an error occurs,
        the original DataFrame is returned.
    """
    if not query:
        return df.copy()
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("df", conn, index=False, if_exists="replace")
        # Only allow SELECT statements for safety
        if not query.strip().lower().startswith("select"):
            raise ValueError("Only SELECT statements are permitted.")
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        # If SQL fails, fallback to original DataFrame and optionally warn the user
        st.warning(f"SQL query error: {e}")
        return df.copy()


def get_filter_summary(original_count: int, filtered_count: int) -> str:
    """Generate a summary message for applied filters.
    
    Parameters
    ----------
    original_count : int
        Original number of rows.
    filtered_count : int
        Number of rows after filtering.
    
    Returns
    -------
    str
        Summary message.
    """
    if original_count == filtered_count:
        return f"📊 Showing all **{original_count:,}** rows"
    else:
        pct = (filtered_count / original_count * 100) if original_count > 0 else 0
        return f"🔍 Filtered: **{filtered_count:,}** / {original_count:,} rows ({pct:.1f}%)"
