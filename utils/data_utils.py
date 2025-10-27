"""
data_utils.py
=================

This module provides utility functions for loading and preparing data within the
Streamlit data analysis application.  The goal of these helpers is to
encapsulate common data‑handling patterns so that the user interface logic
remains clean and easy to follow.

Key features include:

* Loading tabular data from CSV, Excel or SQL databases.
* Simple search functionality across all columns.
* Changing data types and renaming columns.
* Detecting and handling missing values with several strategies.
* Inspecting duplicate rows and removing them.

Each function returns a new DataFrame rather than mutating the input
in‑place.  The caller is responsible for updating any session state or
maintaining multiple versions of the dataset as required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import pickle

import pandas as pd
  # NOTE: The duplicate module header and import statements were removed
  # during refactoring.  The file should contain only one docstring and one
  # set of import statements.  See the top of this file for the canonical
  # definitions.  Do not reintroduce duplicate imports or docstrings here.

# -----------------------------------------------------------------------------
# Optional dependency detection
#
# SQLAlchemy is an optional dependency used only for executing SQL queries
# against external databases.  To avoid slowing down the initial load of
# the application, we probe for SQLAlchemy availability here.  The
# ``load_sql`` function uses this flag to decide whether to attempt
# importing ``create_engine`` at runtime.  If SQLAlchemy is not installed
# ``load_sql`` will raise an informative ``ModuleNotFoundError``.
try:
    import sqlalchemy  # type: ignore  # noqa: F401
    _SQLALCHEMY_AVAILABLE = True
except Exception:
    _SQLALCHEMY_AVAILABLE = False


def load_data(file: Any, file_type: str) -> pd.DataFrame:
    """Load a dataset from an uploaded file.

    Parameters
    ----------
    file : Any
        File‑like object returned from ``st.file_uploader``.
    file_type : str
        The declared type of the file (e.g. ``'csv'`` or ``'excel'``).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the contents of the uploaded file.

    Raises
    ------
    ValueError
        If an unsupported file type is provided.
    """
    if file is None:
        return pd.DataFrame()

    if file_type.lower() == "csv":
        df = pd.read_csv(file)
    elif file_type.lower() in {"xls", "xlsx", "excel"}:
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return df


def load_sql(connection_string: str, query: str) -> pd.DataFrame:
    """Execute a SQL query against a database and return the results.

    This helper uses SQLAlchemy to manage the database connection.  To avoid
    slowing down the application on startup, SQLAlchemy is imported on
    demand within this function.  If SQLAlchemy is not available, a
    ``ModuleNotFoundError`` will be raised to the caller.

    Parameters
    ----------
    connection_string : str
        A valid SQLAlchemy connection string.
    query : str
        The SQL query to execute.  Typically a ``SELECT`` statement.

    Returns
    -------
    pd.DataFrame
        Result set as a DataFrame.  If the query returns no rows,
        the DataFrame will be empty.
    """
    if not _SQLALCHEMY_AVAILABLE:
        raise ModuleNotFoundError(
            "SQLAlchemy is not installed. Please install sqlalchemy to use SQL import functionality."
        )
    # Import ``create_engine`` only when needed.  This avoids paying the cost of
    # importing the entire SQLAlchemy library at module import time.
    from sqlalchemy import create_engine  # type: ignore
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df


def search_df(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Filter a DataFrame by searching for a term across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    search_term : str
        Substring to search for.  Matching is case sensitive.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only rows where at least one value
        contains the search term.  If the search term is empty, the
        original DataFrame is returned unchanged.
    """
    if not search_term:
        return df.copy()
    mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=True, na=False)).any(axis=1)
    return df[mask].copy()


def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Rename columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    rename_map : dict
        Mapping from current column names to new names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    return df.rename(columns=rename_map)


def change_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """Change the data types of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dtype_map : dict
        Mapping from column names to target pandas dtypes (e.g. 'float', 'int', 'category').

    Returns
    -------
    pd.DataFrame
        DataFrame with updated data types.  Any conversion errors will
        propagate as exceptions.
    """
    new_df = df.copy()
    for col, dtype in dtype_map.items():
        # Convert datetime columns to int64 first when casting to integer types.
        # If the source column is a datetime, handle numeric conversions via int64 first
        if pd.api.types.is_datetime64_any_dtype(new_df[col]):
            # For any integer or floating target dtype, first convert the datetime to int64
            # to avoid pandas' ``ConversionError: cannot convert ...`` exceptions.  We check
            # the string prefix instead of an explicit list so that targets like
            # 'int32', 'int64', 'float32' and 'float64' are caught.  Non‑numeric targets
            # fall through to the simple astype call.
            if dtype.startswith('int') or dtype.startswith('float'):
                try:
                    new_df[col] = new_df[col].astype('int64').astype(dtype)
                except Exception:
                    # Some platforms may not allow direct astype to a platform int alias; fallback to object
                    new_df[col] = new_df[col].astype('int64').astype('float64').astype(dtype)
            else:
                new_df[col] = new_df[col].astype(dtype)
        else:
            try:
                new_df[col] = new_df[col].astype(dtype)
            except Exception:
                # Fallback: first cast to object then attempt the target dtype
                new_df[col] = new_df[col].astype('object').astype(dtype)
    return new_df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Table with columns: 'column', 'missing_count', 'missing_pct'.
    """
    total = len(df)
    summary = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
    })
    summary['missing_pct'] = (summary['missing_count'] / total) * 100
    return summary


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', fill_value: Optional[Any] = None) -> pd.DataFrame:
    """Fill or remove missing values using a chosen strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str, optional
        How to handle missing values.  Choices are:

        * 'mean'    – replace numeric columns with their mean.
        * 'median'  – replace numeric columns with their median.
        * 'mode'    – replace each column with its mode.
        * 'constant'– replace all missing values with ``fill_value``.
        * 'drop'    – drop any rows containing missing values.

        Default is 'mean'.

    fill_value : Any, optional
        Constant value used when ``strategy`` == 'constant'.  Ignored
        otherwise.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled according to the strategy.
    """
    df_filled = df.copy()

    if strategy == 'drop':
        df_filled = df_filled.dropna()
    elif strategy == 'mean':
        # Replace missing values only in numeric columns with the column mean.  Non‑numeric
        # columns are left unchanged.  The copy ensures that the original DataFrame is
        # unaffected.
        for col in df_filled.select_dtypes(include=['float', 'int']).columns:
            mean_val = df_filled[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
    elif strategy == 'median':
        for col in df_filled.select_dtypes(include=['float', 'int']).columns:
            med = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(med)
    elif strategy == 'mode':
        for col in df_filled.columns:
            mode_series = df_filled[col].mode()
            if not mode_series.empty:
                df_filled[col] = df_filled[col].fillna(mode_series[0])
    elif strategy == 'constant':
        df_filled = df_filled.fillna(fill_value)
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    return df_filled


def duplicate_summary(df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Identify and count duplicate rows in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[int, pd.DataFrame]
        A tuple of the number of duplicate rows and the duplicate rows
        themselves (if any).  The duplicates are returned in the order
        they appear in the input.
    """
    dup_mask = df.duplicated(keep=False)
    dup_rows = df[dup_mask].copy()
    return dup_mask.sum(), dup_rows


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed.  The first occurrence of
        each row is kept.
    """
    return df.drop_duplicates(keep='first').reset_index(drop=True)

# -----------------------------------------------------------------------------
# Persistent caching and outlier handling utilities
#
# In addition to loading and cleaning data, it is useful to persist the
# currently active dataset across page refreshes.  Streamlit restarts the
# Python interpreter on each run, so storing the cleaned DataFrame on disk
# allows us to restore state when the app is reopened.  The following
# functions implement a simple file‑based cache using pickle.  They live
# alongside the data helpers for convenience.

# Define the cache file relative to this module.  We place it in the
# ``models`` directory one level up from utils so that it sits outside of
# source control but still within the project.  If the directory does not
# exist it will be created on first save.
CACHE_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cached_dataset.pkl')


def save_cached_dataset(df: pd.DataFrame) -> None:
    """Persist the provided DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to cache.
    """
    # Ensure target directory exists
    cache_dir = os.path.dirname(CACHE_FILENAME)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(CACHE_FILENAME, 'wb') as f:
            pickle.dump(df, f)
    except Exception:
        # Intentionally swallow exceptions; caching is a best effort
        pass


def load_cached_dataset() -> Optional[pd.DataFrame]:
    """Load a previously cached DataFrame from disk.

    Returns
    -------
    pd.DataFrame or None
        The cached DataFrame if present, otherwise None.
    """
    if not os.path.exists(CACHE_FILENAME):
        return None
    try:
        with open(CACHE_FILENAME, 'rb') as f:
            df = pickle.load(f)
        # Validate it is a DataFrame
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        return None
    return None


def delete_cached_dataset() -> None:
    """Remove the cached dataset from disk if it exists.

    This helper complements :func:`save_cached_dataset`.  If the caller
    disables persistent data storage (for example, by unchecking the
    "Keep dataset across sessions" option on the Home page), the cached
    file can be removed to free up disk space and prevent old data
    from being loaded on subsequent runs.

    If the file does not exist or cannot be removed, the function
    silently returns without raising an exception.
    """
    try:
        if os.path.exists(CACHE_FILENAME):
            os.remove(CACHE_FILENAME)
    except Exception:
        # Swallow any exceptions; failure to remove the cache should not
        # interrupt the user experience.
        pass


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Identify potential outliers using the interquartile range (IQR) method.

    For each specified numeric column the function computes the first
    (Q1) and third (Q3) quartiles.  Any values below ``Q1 - 1.5 * IQR`` or
    above ``Q3 + 1.5 * IQR`` are flagged as outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Numeric columns to analyse.  If None, all numeric columns
        in ``df`` are considered.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame with columns ``['column','num_outliers','pct_outliers']``.
        The ``pct_outliers`` column expresses the outlier count as a
        percentage of the DataFrame length.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if columns is not None:
        numeric_cols = [c for c in columns if c in numeric_cols]
    summary_rows: List[Dict[str, Any]] = []
    n_total = len(df)
    for col in numeric_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        pct = (count / n_total * 100) if n_total > 0 else 0
        summary_rows.append({'column': col, 'num_outliers': count, 'pct_outliers': pct})
    return pd.DataFrame(summary_rows)


def remove_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove rows containing IQR outliers in any specified column.

    If ``columns`` is None, all numeric columns are examined.  Any row
    where the value of a selected column falls outside the IQR bounds
    is dropped.  This operation may reduce the DataFrame size
    substantially for heavy‑tailed distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Subset of numeric columns to evaluate.  Defaults to all numeric.

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier rows removed.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if columns is not None:
        numeric_cols = [c for c in columns if c in numeric_cols]
    clean_df = df.copy()
    for col in numeric_cols:
        series = clean_df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series >= lower) & (series <= upper)
        clean_df = clean_df[mask]
    return clean_df.reset_index(drop=True)