"""
ml_utils.py
===========

Machine learning helper functions for training and evaluating models.

This module centralises model selection, splitting of data, and evaluation of
metrics for the built-in algorithms.  By keeping
model training code here, we avoid duplication across pages and make
the rest of the app easier to maintain.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

# Import of XGBoost models has been deferred.  The variables below
# remain ``None`` until explicitly loaded via ``_get_xgboost_models``.
XGBClassifier = None  # type: ignore
XGBRegressor = None  # type: ignore


def _get_xgboost_models() -> Tuple[Any, Any]:
    """Dynamically import XGBoost models.

    Returns
    -------
    (XGBClassifier, XGBRegressor)
        The XGBoost classifier and regressor classes.  If the
        ``xgboost`` package is not installed, both entries will be ``None``.
    """
    global XGBClassifier, XGBRegressor
    if XGBClassifier is None or XGBRegressor is None:
        try:
            from xgboost import XGBClassifier as _XGBClassifier, XGBRegressor as _XGBRegressor  # type: ignore
        except ImportError:
            # Leave the placeholders as None if xgboost is unavailable
            XGBClassifier, XGBRegressor = None, None  # type: ignore
        else:
            XGBClassifier, XGBRegressor = _XGBClassifier, _XGBRegressor  # type: ignore
    return XGBClassifier, XGBRegressor


def _detect_problem_type(y: pd.Series) -> str:
    """Infer whether the problem is regression or classification.

    If the target data type is numeric and has more than a handful of
    unique values, treat it as regression.  Otherwise treat it as
    classification.
    """
    if pd.api.types.is_numeric_dtype(y):
        # Heuristic: classification if there are few unique values
        unique_vals = y.nunique()
        if unique_vals <= 10:
            return 'classification'
        return 'regression'
    else:
        return 'classification'


def train_model(df: pd.DataFrame, target: str, algorithm: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Dict[str, float]]:
    """Train a machine learning model and return metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and target.
    target : str
        Name of the target column.
    algorithm : str
        One of ``'Linear Regression'``, ``'Logistic Regression'``,
        ``'Random Forest'``, ``'XGBoost'``.  The function will
        automatically select the classification or regression variant
        where appropriate.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducible splits.

    Returns
    -------
    (model, metrics)
        A tuple containing the trained model and a dictionary of
        evaluation metrics.  The keys of the metrics dictionary depend
        on whether the problem is classification or regression.
    """
    # Prepare features and target
    X = df.drop(columns=[target]).copy()
    y = df[target]

    # Convert datetime columns to numeric to avoid dtype promotion errors
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            # Convert to integer timestamp (nanosecond resolution)
            # using view('int64') preserves ordering and avoids floats
            X[col] = X[col].values.astype('datetime64[ns]').astype('int64')

    problem_type = _detect_problem_type(y)
    # Handle categorical features by one‑hot encoding if not already numeric
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Handle missing values in features using mean imputation for numeric columns
    imputer = SimpleImputer(strategy='mean')
    X_encoded_imputed = pd.DataFrame(
        imputer.fit_transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index
    )

    # Handle missing values in target variable by dropping rows with NaN in target
    valid_idx = y.notna()
    X_encoded_imputed = X_encoded_imputed[valid_idx]
    y = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X_encoded_imputed, y, test_size=test_size, random_state=random_state)

    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif algorithm == 'Random Forest':
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    elif algorithm == 'XGBoost':
        # Lazily import the XGBoost models only when needed.  This avoids
        # incurring the cost of importing xgboost at application startup.
        xgb_cls, xgb_reg = _get_xgboost_models()
        if xgb_cls is None or xgb_reg is None:
            raise ImportError("XGBoost is not installed. Please install xgboost to use this algorithm.")
        if problem_type == 'classification':
            model = xgb_cls(n_estimators=200, learning_rate=0.1, random_state=random_state, eval_metric='logloss')
        else:
            model = xgb_reg(n_estimators=200, learning_rate=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {}
    if problem_type == 'classification':
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
    else:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = float(np.sqrt(mse))
        metrics['r2'] = float(r2_score(y_test, y_pred))

    # Attach the input feature names to the model for later use (e.g. prediction)
    try:
        model.input_features_ = X_encoded.columns.tolist()
    except Exception:
        pass
    return model, metrics


def get_feature_importance(model: Any, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract and format feature importances from tree‑based models.

    For linear models, the absolute value of the coefficients is used
    instead.  If the model does not expose a straightforward notion of
    feature importance, ``None`` is returned.

    Parameters
    ----------
    model : estimator
        A trained scikit‑learn estimator.
    feature_names : list of str
        Names of the features corresponding to the columns used to fit
        the model.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``['feature', 'importance']``, sorted
        descending by importance.  Returns None if importance cannot
        be computed.
    """
    importances: Optional[np.ndarray] = None
    # Tree‑based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Linear models
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            # For multiclass classification, sum absolute coefficients across classes
            importances = np.sum(np.abs(coef), axis=0)
    else:
        return None
    # Align the length of feature_names and importances.  Some models may expose
    # importances for a subset of features (e.g. after one‑hot encoding).  To
    # prevent ValueError due to differing lengths, zip the names and importances.
    pairs = list(zip(feature_names, importances))
    importance_df = pd.DataFrame(pairs, columns=['feature', 'importance'])
    importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    return importance_df