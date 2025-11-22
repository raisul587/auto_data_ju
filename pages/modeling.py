
"""
modeling.py
===========

Machine learning page logic: handles feature engineering, training, prediction
and optional time-series forecasting.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Any, Dict

from utils import feature_engineering as fe
from utils import ml_utils as mu


def show_modeling_page() -> None:
    st.header("🤖 Machine Learning & Advanced Analytics")
    if 'clean_df' not in st.session_state or st.session_state.clean_df.empty:
        st.info("Please load and prepare a dataset on the Data page first.")
        return
    
    # Use filtered data if available, otherwise use clean data
    df = st.session_state.get('filtered_df', st.session_state.clean_df)

    # Show filter status
    if 'filtered_df' in st.session_state and len(st.session_state.filtered_df) != len(st.session_state.clean_df):
        st.info(f"🔍 Training on filtered data: {len(df):,} rows (filtered from {len(st.session_state.clean_df):,} total rows)")

    # Reset model_df when underlying dataset structure changes (e.g. columns dropped)
    dataset_signature = (
        tuple(df.columns.tolist()),
        len(df)
    )
    stored_signature = st.session_state.get('model_df_signature')

    if (
        'model_df' not in st.session_state
        or st.session_state.model_df.empty
        or stored_signature != dataset_signature
    ):
        st.session_state.model_df = df.copy()
        st.session_state.encoding_groups = {}
        st.session_state.label_encoders = {}
        st.session_state.model_df_signature = dataset_signature

    st.session_state.setdefault('encoding_groups', {})
    st.session_state.setdefault('label_encoders', {})

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

        # Encoding part
        encode_cols = st.multiselect("Select categorical columns to encode", options=cat_cols)
        encode_method = st.selectbox("Encoding method", options=['onehot', 'label'])
        if st.button("Apply Encoding"):
            if not encode_cols:
                st.warning("Please select at least one column to encode.")
            else:
                before_df = st.session_state.model_df.copy()
                st.session_state.model_df, encoders = fe.encode_categorical(st.session_state.model_df, encode_cols, method=encode_method)
                if encode_method == 'onehot':
                    new_columns = [c for c in st.session_state.model_df.columns if c not in before_df.columns]
                    for col in encode_cols:
                        if col not in before_df.columns:
                            continue
                        prefix = f"{col}_"
                        dummy_cols = [c for c in new_columns if c.startswith(prefix)]
                        if not dummy_cols:
                            continue
                        categories = before_df[col].dropna().unique().tolist()
                        st.session_state.encoding_groups[col] = {
                            'dummy_cols': dummy_cols,
                            'categories': categories,
                        }
                        st.session_state.label_encoders.pop(col, None)
                else:
                    for col, encoder in encoders.items():
                        st.session_state.label_encoders[col] = encoder
                        st.session_state.encoding_groups.pop(col, None)
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
                st.info("Tip: Ensure all features are numeric/encoded and that datetime columns are converted before training.")

        if 'last_metrics' in st.session_state:
            st.subheader("Metrics")
            st.json(st.session_state.last_metrics)
            with st.expander("Feature Importance"):
                last_model = st.session_state.get('last_model')
                last_target = st.session_state.get('last_target')
                feature_names = getattr(last_model, 'input_features_', None) if last_model is not None else None
                if feature_names is None and last_target is not None:
                    feature_names = st.session_state.model_df.drop(columns=[last_target]).columns.tolist()
                if last_model is not None and feature_names is not None:
                    imp_df = mu.get_feature_importance(last_model, feature_names)
                    if imp_df is not None:
                        st.dataframe(imp_df.head(20))
                    else:
                        st.info("The selected model does not expose feature importance.")
                else:
                    st.info("Train a model to view feature importance.")

    # Prediction section
    if 'last_model' in st.session_state:
        st.subheader("🔮 Predict on New Data")
        model = st.session_state.last_model
        target = st.session_state.get('last_target')
        train_df = st.session_state.model_df
        feature_cols = [col for col in train_df.columns if col != target]

        encoding_groups = st.session_state.get('encoding_groups', {})
        label_encoders = st.session_state.get('label_encoders', {})
        dummy_columns = {dummy for info in encoding_groups.values() for dummy in info.get('dummy_cols', [])}

        default_values: Dict[str, Any] = {}
        for col in feature_cols:
            series = train_df[col]
            if pd.api.types.is_bool_dtype(series):
                default_values[col] = bool(series.mode().iloc[0]) if not series.mode().empty else False
            elif pd.api.types.is_numeric_dtype(series):
                default_values[col] = float(series.mean()) if len(series) > 0 else 0.0
            else:
                default_values[col] = series.mode().iloc[0] if not series.mode().empty else ""

        feature_entries: List[Dict[str, Any]] = []
        for original, info in encoding_groups.items():
            feature_entries.append({'name': original, 'type': 'onehot', 'info': info})

        for col in feature_cols:
            if col in dummy_columns:
                continue
            entry: Dict[str, Any] = {'name': col, 'column': col}
            if col in label_encoders:
                entry['type'] = 'label'
                entry['encoder'] = label_encoders[col]
            elif pd.api.types.is_bool_dtype(train_df[col]):
                entry['type'] = 'bool'
            elif pd.api.types.is_numeric_dtype(train_df[col]):
                entry['type'] = 'numeric'
            else:
                entry['type'] = 'categorical'
                entry['options'] = train_df[col].dropna().unique().tolist()
            feature_entries.append(entry)

        st.caption("Provide real-world inputs for the selected features. Categorical fields accept the original labels even if the dataset was encoded.")
        user_inputs: Dict[str, Any] = {}
        for entry in feature_entries:
            label = entry['name']
            key = f"pred_{label}"
            if entry['type'] == 'onehot':
                categories = entry['info'].get('categories') or []
                if not categories:
                    categories = [col.split(f"{label}_", 1)[1] for col in entry['info'].get('dummy_cols', []) if col.startswith(f"{label}_") and '_' in col]
                categories = categories or ['']
                user_inputs[label] = st.selectbox(label, options=categories, key=key, format_func=str)
            elif entry['type'] == 'label':
                options = entry['encoder'].classes_.tolist()
                default_numeric = default_values.get(entry['column'])
                default_label = options[0] if options else ''
                if default_numeric is not None and options:
                    try:
                        default_label = entry['encoder'].inverse_transform([int(default_numeric)])[0]
                    except Exception:
                        pass
                idx = options.index(default_label) if options and default_label in options else 0
                user_inputs[label] = st.selectbox(label, options=options, index=idx, key=key)
            elif entry['type'] == 'bool':
                user_inputs[label] = st.checkbox(label, value=bool(default_values.get(entry['column'], False)), key=key)
            elif entry['type'] == 'numeric':
                user_inputs[label] = st.number_input(label, value=float(default_values.get(entry['column'], 0.0)), key=key)
            else:
                options = entry.get('options') or []
                options = options or ['']
                default_choice = default_values.get(entry['column'], options[0])
                if default_choice not in options:
                    default_choice = options[0]
                idx = options.index(default_choice) if default_choice in options else 0
                user_inputs[label] = st.selectbox(label, options=options, index=idx, key=key, format_func=str)

        if st.button("Predict", key="predict_button"):
            row_values: Dict[str, Any] = default_values.copy()
            for entry in feature_entries:
                value = user_inputs.get(entry['name'])
                if entry['type'] == 'onehot':
                    dummy_cols = entry['info'].get('dummy_cols', [])
                    for dummy in dummy_cols:
                        row_values[dummy] = 1.0 if dummy == f"{entry['name']}_{value}" else 0.0
                elif entry['type'] == 'label':
                    row_values[entry['column']] = float(entry['encoder'].transform([value])[0])
                else:
                    row_values[entry['column']] = value

            input_df = pd.DataFrame([row_values])
            for c in input_df.columns:
                if c in train_df.columns and pd.api.types.is_datetime64_any_dtype(train_df[c]):
                    input_df[c] = pd.to_datetime(input_df[c]).astype('int64')
            encoded_input = pd.get_dummies(input_df, drop_first=True)
            model_features = getattr(model, 'input_features_', None)
            if model_features is not None:
                encoded_input = encoded_input.reindex(columns=model_features, fill_value=0)
            try:
                pred = model.predict(encoded_input)[0]
                st.success(f"Predicted value: {pred}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
