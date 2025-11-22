"""
ml_utils.py
===========

Machine learning helper functions for training and evaluating models.

This module centralises model selection, splitting of data, evaluation of
metrics and optional AutoML functionality via PyCaret.  By keeping
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

"""
Lazy imports for optional heavy dependencies
-------------------------------------------

The original implementation imported heavy libraries such as PyCaret and
XGBoost at the module level.  Importing these packages unconditionally
causes a noticeable delay when the app first starts up, even if the
corresponding functionality is never used.  To make the application
more responsive on initial load, the imports have been refactored
so that they occur only when needed.  The helper functions below
attempt to import the required modules at call time and will raise
ImportError if the packages are unavailable.
"""

# Placeholders for optional imports.  They remain ``None`` until the
# first time the corresponding functionality is invoked.  See
# ``_get_pycaret_functions`` and ``_get_xgboost_models`` for details.
cls_setup = None  # type: ignore
compare_cls = None  # type: ignore
reg_setup = None  # type: ignore
compare_reg = None  # type: ignore

def _get_pycaret_functions() -> Tuple[Any, Any, Any, Any]:
    """Dynamically import PyCaret setup and compare_models functions.

    Returns
    -------
    tuple
        A 4‑tuple containing the classification setup, classification
        compare_models, regression setup and regression compare_models
        callables from PyCaret.  If PyCaret cannot be imported, an
        ImportError is raised.
    """
    global cls_setup, compare_cls, reg_setup, compare_reg
    if cls_setup is None or compare_cls is None or reg_setup is None or compare_reg is None:
        try:
            from pycaret.classification import setup as _cls_setup, compare_models as _compare_cls  # type: ignore
            from pycaret.regression import setup as _reg_setup, compare_models as _compare_reg  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyCaret is not installed. Please install pycaret>=3.0 to use AutoML."
            ) from e
        cls_setup = _cls_setup  # type: ignore
        compare_cls = _compare_cls  # type: ignore
        reg_setup = _reg_setup  # type: ignore
        compare_reg = _compare_reg  # type: ignore
    return cls_setup, compare_cls, reg_setup, compare_reg

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


def auto_ml(df: pd.DataFrame, target: str, problem_type: Optional[str] = None) -> Tuple[Any, Any]:
    """Run an automated machine learning experiment via PyCaret.

    PyCaret is an optional dependency that provides AutoML functionality.
    To minimise the startup time of the application, the PyCaret
    components are imported only when this function is called.  If
    PyCaret is unavailable, an informative ImportError is raised.  The
    function detects whether the task is classification or regression
    based on the target column if ``problem_type`` is not specified.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including the target column.
    target : str
        Name of the target column.
    problem_type : str, optional
        Force either 'classification' or 'regression'.  If None, the
        problem type will be inferred.

    Returns
    -------
    (best_model, leaderboard)
        A tuple containing the best model returned by PyCaret and
        whatever object PyCaret's ``compare_models`` returns.  Depending
        on the PyCaret version this may be a DataFrame leaderboard or
        another experiment object.
    """
    # Load PyCaret functions on demand
    try:
        cls_setup_fn, compare_cls_fn, reg_setup_fn, compare_reg_fn = _get_pycaret_functions()
    except ImportError:
        # Re‑raise with a clearer message for the caller
        raise ImportError("PyCaret is not installed. Please install pycaret>=3.0 to use AutoML.")

    # Infer problem type if not explicitly provided
    if problem_type is None:
        problem_type = _detect_problem_type(df[target])

    data = df.copy()
    # Run the appropriate PyCaret workflow and capture the experiment object
    if problem_type == 'classification':
        # Remove deprecated 'silent' parameter; Preprocess=True remains for consistency
        exp = cls_setup_fn(data=data, target=target, session_id=42, preprocess=True)
        best_candidates = compare_cls_fn(n_select=1)
    else:
        exp = reg_setup_fn(data=data, target=target, session_id=42, preprocess=True)
        best_candidates = compare_reg_fn(n_select=1)

    # `compare_models` may return a single estimator or a list/tuple of estimators
    if isinstance(best_candidates, (list, tuple)):
        best_model = best_candidates[0] if len(best_candidates) > 0 else None
    else:
        best_model = best_candidates

    # Attempt to extract the leaderboard DataFrame produced by PyCaret's experiment
    leaderboard: Optional[pd.DataFrame]
    try:
        leaderboard = exp.pull()
    except Exception:
        leaderboard = None

    return best_model, leaderboard