"""
feature_engineering.py
======================

Helper functions for feature engineering operations such as scaling,
encoding, correlation analysis and time series forecasting.

The functions here are designed to be flexible and operate on
arbitrary pandas DataFrames.  They return both the transformed
DataFrame and any fitted transformers or encoders so that the caller
can reuse them for prediction pipelines if desired.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    # Prophet has been renamed to prophet; fallback to fbprophet if needed.
    from prophet import Prophet
except ImportError:  # pragma: no cover
    try:
        from fbprophet import Prophet  # type: ignore
    except ImportError:
        Prophet = None  # type: ignore


def scale_data(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """Scale numeric columns using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Numeric columns to scale.
    method : str, optional
        Either ``'standard'`` for StandardScaler or ``'minmax'`` for
        MinMaxScaler.  Default is ``'standard'``.

    Returns
    -------
    (pd.DataFrame, object)
        A tuple of the transformed DataFrame and the fitted scaler.
    """
    new_df = df.copy()
    if not columns:
        return new_df, None
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    new_df[columns] = scaler.fit_transform(new_df[columns])
    return new_df, scaler


def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply a log1p transformation to selected numeric columns.

    The log1p transformation computes ``log(1 + x)`` which is defined
    for zero and positive values.  It can help reduce skewness in
    heavily right‑skewed distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Numeric columns to transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with log1p applied to specified columns.
    """
    new_df = df.copy()
    for col in columns:
        # Ensure non‑negativity by shifting if necessary
        min_val = new_df[col].min()
        if min_val <= -1:
            shift = abs(min_val) + 1
            new_df[col] = new_df[col] + shift
        new_df[col] = np.log1p(new_df[col])
    return new_df


def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Encode categorical variables for machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Categorical columns to encode.
    method : str, optional
        ``'onehot'`` to perform one‑hot encoding or ``'label'`` to
        perform ordinal encoding via sklearn's LabelEncoder.  One‑hot
        encoding will expand the DataFrame with new dummy columns.

    Returns
    -------
    (pd.DataFrame, dict)
        The transformed DataFrame and a mapping of encoders used for
        each column.
    """
    new_df = df.copy()
    encoders: Dict[str, object] = {}
    if method == 'onehot':
        # Use pandas get_dummies which automatically handles NaNs
        new_df = pd.get_dummies(new_df, columns=columns, drop_first=False)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            new_df[col] = new_df[col].astype(str)  # ensure string type
            new_df[col] = le.fit_transform(new_df[col])
            encoders[col] = le
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    return new_df, encoders


def correlation_feature_selection(df: pd.DataFrame, target: str, threshold: float = 0.8) -> List[str]:
    """Identify highly correlated features for potential removal.

    This function computes the absolute correlation between each
    feature and the target variable.  Features with correlation
    greater than the provided threshold are returned.  This is a
    simplistic heuristic often used to reduce multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    threshold : float, optional
        Correlation threshold above which features are returned.  Must
        be between 0 and 1.  Default is 0.8.

    Returns
    -------
    list of str
        List of feature names whose absolute correlation with the
        target is greater than the threshold.
    """
    corr = df.corr(numeric_only=True)
    # Exclude the target from the list of predictors
    corr_target = corr[target].drop(labels=[target])
    high_corr = corr_target[abs(corr_target) > threshold].index.tolist()
    return high_corr


def detect_time_series_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns of datetime type which could represent time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list of str
        Columns with a datetime64 dtype.
    """
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]


def forecast_time_series(df: pd.DataFrame, date_col: str, target_col: str, periods: int = 30) -> Optional[pd.DataFrame]:
    """Perform time series forecasting using Prophet.

    The function requires the Prophet library to be installed.  If
    Prophet is unavailable, the function will return ``None``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column and a target column.
    date_col : str
        Name of the datetime column.
    target_col : str
        Name of the target variable to forecast.
    periods : int, optional
        Number of future periods to forecast.  Default is 30.

    Returns
    -------
    pd.DataFrame or None
        Forecast DataFrame with columns ['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']
        if Prophet is available, otherwise ``None``.
    """
    if Prophet is None:
        return None
    # Prepare data for Prophet
    ts_df = df[[date_col, target_col]].dropna().copy()
    ts_df = ts_df.rename(columns={date_col: 'ds', target_col: 'y'})
    model = Prophet()
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]