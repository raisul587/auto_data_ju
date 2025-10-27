"""
modeling.py
===========

This module contains the machine learning page.  It allows users to
configure feature engineering steps, select a target variable, choose
an algorithm and view the resulting metrics and feature importances.
Time series forecasting and AutoML are also available for advanced
analysis.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Any, Dict

from utils import feature_engineering as fe
from utils import ml_utils as mu


def show_modeling_page() -> None:
    """Render the machine learning page."""
    st.header("🤖 Machine Learning & Advanced Analytics")
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please load and prepare a dataset on the Data page first.")
        return
    
    # Use filtered data if available, otherwise use clean data
    df = st.session_state.get('filtered_df', st.session_state.clean_df)
    
    # Show filter status
    if 'filtered_df' in st.session_state and len(st.session_state.filtered_df) != len(st.session_state.clean_df):
        st.info(f"🔍 Training on filtered data: {len(df):,} rows (filtered from {len(st.session_state.clean_df):,} total rows)")

    # Initialize model_df in session state
    if 'model_df' not in st.session_state or st.session_state.model_df.empty:
        st.session_state.model_df = df.copy()

    # Feature engineering settings
    with st.expander("⚙️ Feature Engineering"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        # Scaling
        scale_cols = st.multiselect("Select numeric columns to scale", options=numeric_cols)
        scale_method = st.selectbox("Scaling method", options=['standard', 'minmax'])
        if st.button("Apply Scaling"):
            st.session_state.model_df, scaler = fe.scale_data(st.session_state.model_df, scale_cols, method=scale_method)
            st.success("Scaling applied.")

        # Log transform
        log_cols = st.multiselect("Select numeric columns to log transform", options=numeric_cols, key="log_cols")
        if st.button("Apply Log Transform"):
            st.session_state.model_df = fe.log_transform(st.session_state.model_df, log_cols)
            st.success("Log transformation applied.")

        # Encoding
        encode_cols = st.multiselect("Select categorical columns to encode", options=cat_cols)
        encode_method = st.selectbox("Encoding method", options=['onehot', 'label'])
        if st.button("Apply Encoding"):
            st.session_state.model_df, encoders = fe.encode_categorical(st.session_state.model_df, encode_cols, method=encode_method)
            st.success("Encoding applied.")

    # Time series detection and forecasting
    with st.expander("📆 Time Series Forecasting"):
        ts_cols = fe.detect_time_series_columns(df)
        if not ts_cols:
            st.info("No datetime columns detected for time series forecasting.")
        else:
            st.write(f"Detected datetime columns: {', '.join(ts_cols)}")
            date_col = st.selectbox("Select date column", options=ts_cols)
            target_col_ts = st.selectbox("Select target column", options=df.select_dtypes(include=['number']).columns.tolist())
            periods = st.number_input("Forecast periods", min_value=1, max_value=365, value=30, step=1)
            if st.button("Run Forecast"):
                forecast = fe.forecast_time_series(df, date_col, target_col_ts, periods=int(periods))
                if forecast is None:
                    st.error("Prophet is not installed. Please install prophet or fbprophet to use forecasting.")
                else:
                    st.subheader("Forecast Results")
                    st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])

    # ML training
    with st.expander("🧠 Train/Test Models"):
        # Choose target
        target_col = st.selectbox("Select target variable", options=st.session_state.model_df.columns.tolist())
        features = [c for c in st.session_state.model_df.columns if c != target_col]
        alg = st.selectbox("Choose algorithm", options=['Linear Regression', 'Logistic Regression', 'Random Forest', 'XGBoost'])
        test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        if st.button("Train Model"):
            try:
                model, metrics = mu.train_model(st.session_state.model_df[[*features, target_col]], target_col, alg, test_size=test_size)
                st.session_state.last_model = model
                st.session_state.last_metrics = metrics
                st.session_state.last_target = target_col
                st.success("Model trained successfully.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                # Provide a simple tip when dtype errors occur
                st.info("Tip: Ensure that all feature columns are numeric or encoded. Convert datetime columns to numeric values before training.")

        # Display metrics
        if 'last_metrics' in st.session_state:
            st.subheader("Metrics")
            metrics = st.session_state.last_metrics
            st.json(metrics)
            # Feature importance
            with st.expander("Feature Importance"):
                # Use the input_features_ attribute from the model if available; otherwise fall back
                feature_names = getattr(st.session_state.last_model, 'input_features_', None)
                if feature_names is None:
                    feature_names = st.session_state.model_df.drop(columns=[target_col]).columns.tolist()
                imp_df = mu.get_feature_importance(st.session_state.last_model, feature_names)
                if imp_df is not None:
                    st.dataframe(imp_df.head(20))
                else:
                    st.info("The selected model does not expose feature importance.")

    # AutoML
    with st.expander("⚡ AutoML (PyCaret)"):
        if st.button("Run AutoML"):
            try:
                best_model, leaderboard = mu.auto_ml(st.session_state.model_df, target_col)
                st.success("AutoML completed.")
                st.subheader("Best Model")
                st.write(best_model)
                st.subheader("Leaderboard")
                st.dataframe(leaderboard)
            except Exception as e:
                st.error(f"AutoML failed: {e}")

    # Prediction section
    if 'last_model' in st.session_state:
        st.subheader("🔮 Predict on New Data")
        model = st.session_state.last_model
        target = st.session_state.get('last_target')
        # Determine feature columns used for training
        train_df = st.session_state.model_df
        feature_cols = [col for col in train_df.columns if col != target]
        # Build input interface
        user_inputs: Dict[str, Any] = {}
        for col in feature_cols:
            series = train_df[col]
            if pd.api.types.is_bool_dtype(series):
                default_val = bool(series.mode().iloc[0]) if not series.mode().empty else False
                user_inputs[col] = st.checkbox(f"{col}", value=default_val, key=f"pred_{col}")
            elif pd.api.types.is_numeric_dtype(series):
                default_val = float(series.mean()) if len(series) > 0 else 0.0
                user_inputs[col] = st.number_input(f"{col}", value=default_val, key=f"pred_{col}")
            else:
                options = series.dropna().unique().tolist()
                if len(options) == 0:
                    options = ['']
                user_inputs[col] = st.selectbox(f"{col}", options=options, key=f"pred_{col}")
        if st.button("Predict", key="predict_button"):
            # Create a one‑row DataFrame from user inputs
            input_df = pd.DataFrame([user_inputs])
            # Convert datetime columns to numeric
            for c in input_df.columns:
                if pd.api.types.is_datetime64_any_dtype(train_df[c]):
                    input_df[c] = pd.to_datetime(input_df[c]).astype('int64')
            # One‑hot encode categorical columns
            encoded_input = pd.get_dummies(input_df, drop_first=True)
            # Align columns with model input features
            model_features = getattr(model, 'input_features_', None)
            if model_features is not None:
                encoded_input = encoded_input.reindex(columns=model_features, fill_value=0)
            # Predict
            try:
                pred = model.predict(encoded_input)[0]
                st.success(f"Predicted value: {pred}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")