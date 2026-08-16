from __future__ import annotations

import importlib.metadata
import io
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# Foundation Models
try:
    from tabicl import TabICLClassifier, TabICLRegressor
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False

try:
    from tabfm import TabFMClassifier, TabFMRegressor
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    TABFM_AVAILABLE = True
except ImportError:
    TABFM_AVAILABLE = False

# Traditional ML & AutoML
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import optuna
    import shap
    from flaml import AutoML
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False

APP_TITLE = "Tabtester - Benchmark Edition"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

ALL_MODELS = [
    "TabFM", "TabICLv2", 
    "XGBoost (Default)", "LightGBM", "CatBoost", 
    "XGBoost (Tuned)", "FLAML", "AutoGluon"
]


@st.cache_resource(show_spinner="Loading TabFM base model weights...")
def load_tabfm_base_model(task: str, device: str) -> Any:
    model_type = "regression" if task == "Regression" else "classification"
    target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = tabfm_v1_0_0.load(model_type=model_type)
    model = model.to(target_device)
    return model


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def read_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not decode CSV. Tried utf-8-sig, utf-8, and cp932.")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_stratify(y: pd.Series, test_size: float) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    n_test = max(1, math.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    if n_test < len(counts) or n_train < len(counts):
        return None
    return y


def run_optuna_xgboost(X_train, y_train, task, n_trials=10, random_state=42):
    """Tunes XGBoost using Optuna and returns the best parameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    is_regression = (task == "Regression")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
            'n_jobs': -1
        }
        scores = []
        if is_regression:
            model = xgb.XGBRegressor(**params)
            cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
            for tr_idx, va_idx in cv.split(X_train):
                model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                scores.append(r2_score(y_train.iloc[va_idx], model.predict(X_train.iloc[va_idx])))
        else:
            model = xgb.XGBClassifier(**params, eval_metric='logloss')
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            for tr_idx, va_idx in cv.split(X_train, y_train):
                model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                scores.append(accuracy_score(y_train.iloc[va_idx], model.predict(X_train.iloc[va_idx])))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['n_jobs'] = -1
    if not is_regression:
        best_params['eval_metric'] = 'logloss'
    return best_params


def generate_shap_plot(model, X_data, title="SHAP Summary"):
    """Generates and returns a SHAP summary plot figure."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data, show=False)
        plt.title(title)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not generate SHAP plot: {e}")
        return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Evaluate and benchmark Foundation Models (TabFM/TabICL) against Traditional ML & AutoML.")

    with st.sidebar:
        st.subheader("Environment Status")
        st.text(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
        st.text(f"ML Libs: {'Loaded' if ML_AVAILABLE else 'Missing'}")
        st.text(f"AutoML/SHAP: {'Loaded' if AUTOML_AVAILABLE else 'Missing'}")
        st.text(f"AutoGluon: {'Loaded' if AG_AVAILABLE else 'Missing'}")
        
        st.divider()
        st.subheader("Benchmark Settings")
        task = st.radio("Task Type", ["Regression", "Classification"], index=0)
        
        # Determine available models based on imports
        available_models = ["TabFM"] if TABFM_AVAILABLE else []
        if TABICL_AVAILABLE: available_models.append("TabICLv2")
        if ML_AVAILABLE: available_models.extend(["XGBoost (Default)", "LightGBM", "CatBoost"])
        if AUTOML_AVAILABLE: available_models.extend(["XGBoost (Tuned)", "FLAML"])
        if AG_AVAILABLE: available_models.append("AutoGluon")
            
        selected_models = st.multiselect(
            "Select Models to Benchmark", 
            available_models, 
            default=["TabFM", "TabICLv2", "XGBoost (Default)"] if "TabICLv2" in available_models else available_models[:1]
        )
        
        device = st.selectbox("Foundation Model Device", ["auto", "cpu", "cuda"], index=0)
        random_state = st.number_input("Random Seed", min_value=0, value=DEFAULT_RANDOM_STATE, step=1)
        
        with st.expander("AutoML & Tuning Settings", expanded=False):
            n_trials = st.slider("Optuna Trials (XGB Tuned)", min_value=5, max_value=50, value=10, step=5)
            time_budget = st.slider("AutoML Time Budget (sec)", min_value=10, max_value=300, value=30, step=10)

    # 1. Data Upload & Preview
    train_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if train_file is None:
        st.info("Upload a CSV dataset to begin the benchmark.")
        return

    try:
        df = read_csv(train_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # 2. Feature & Target Selection
    target = st.selectbox("Target Column", list(df.columns), index=len(df.columns) - 1)
    candidate_excluded = [c for c in df.columns if c != target]
    excluded = st.multiselect("Exclude Columns (e.g., IDs)", candidate_excluded, default=[])

    # Clean target NaNs
    before_len = len(df)
    df_clean = df.dropna(subset=[target]).copy()
    dropped_rows = before_len - len(df_clean)
    
    X = df_clean.drop(columns=[target] + excluded)
    y = df_clean[target].copy()

    if task == "Regression":
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().sum() > 0:
            st.error("Regression target contains non-numeric values.")
            return
    else:
        st.caption(f"Detected Classes: {y.nunique(dropna=False)}")

    tab_eval, tab_predict, tab_impute = st.tabs(["Benchmark & Evaluation", "Predict New Rows", "Impute Missing Values"])

    # ---------------------------------------------------------
    # TAB 1: Benchmark & Evaluation
    # ---------------------------------------------------------
    with tab_eval:
        st.markdown("### Model Performance Benchmark")
        test_size = st.slider("Test Fraction", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)
        
        if st.button("Run Benchmark", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("Please select at least one model from the sidebar.")
                return
                
            stratify = safe_stratify(y, test_size) if task == "Classification" else None
            
            # Split for Foundation Models (raw data)
            X_train_fm, X_test_fm, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
            
            # Split for ML Models (dummy encoded for categorical)
            if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
                X_ml = pd.get_dummies(X, drop_first=True)
            else:
                X_ml = X.copy()
            X_train_ml, X_test_ml, _, _ = train_test_split(
                X_ml, y, test_size=test_size, random_state=random_state, stratify=stratify
            )

            benchmark_results = []
            preds_dict = {}
            trained_ml_models = {}

            # Progress tracking
            progress_text = "Running Benchmark..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, model_name in enumerate(selected_models):
                my_bar.progress((i) / len(selected_models), text=f"Processing {model_name}...")
                
                prep_time = 0.0
                start_train = time.time()
                
                try:
                    if model_name == "TabFM":
                        base_model = load_tabfm_base_model(task, device)
                        model = TabFMRegressor(model=base_model) if task == "Regression" else TabFMClassifier(model=base_model)
                        model.fit(X_train_fm, y_train.values)
                        preds = model.predict(X_test_fm)
                        
                    elif model_name == "TabICLv2":
                        model = TabICLRegressor(random_state=random_state) if task == "Regression" else TabICLClassifier(random_state=random_state)
                        model.fit(X_train_fm, y_train.values)
                        preds = model.predict(X_test_fm)

                    elif model_name == "XGBoost (Default)":
                        model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1) if task == "Regression" else xgb.XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1)
                        model.fit(X_train_ml, y_train)
                        preds = model.predict(X_test_ml)
                        trained_ml_models[model_name] = model

                    elif model_name == "LightGBM":
                        model = lgb.LGBMRegressor(random_state=random_state, verbose=-1, n_jobs=-1) if task == "Regression" else lgb.LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1)
                        model.fit(X_train_ml, y_train)
                        preds = model.predict(X_test_ml)
                        trained_ml_models[model_name] = model

                    elif model_name == "CatBoost":
                        model = cb.CatBoostRegressor(random_state=random_state, verbose=0, thread_count=-1) if task == "Regression" else cb.CatBoostClassifier(random_state=random_state, verbose=0, thread_count=-1)
                        model.fit(X_train_ml, y_train)
                        preds = model.predict(X_test_ml)
                        trained_ml_models[model_name] = model

                    elif model_name == "XGBoost (Tuned)":
                        start_tune = time.time()
                        best_p = run_optuna_xgboost(X_train_ml, y_train, task, n_trials, random_state)
                        prep_time = time.time() - start_tune
                        start_train = time.time() # Reset train time after tuning
                        
                        model = xgb.XGBRegressor(**best_p) if task == "Regression" else xgb.XGBClassifier(**best_p)
                        model.fit(X_train_ml, y_train)
                        preds = model.predict(X_test_ml)
                        trained_ml_models[model_name] = model

                    elif model_name == "FLAML":
                        start_tune = time.time()
                        automl = AutoML()
                        flaml_task = 'regression' if task == "Regression" else 'classification'
                        flaml_metric = 'r2' if task == "Regression" else 'accuracy'
                        automl.fit(X_train=X_train_ml, y_train=y_train, task=flaml_task, metric=flaml_metric, time_budget=time_budget, verbose=0)
                        prep_time = time.time() - start_tune
                        start_train = time.time()
                        
                        preds = automl.predict(X_test_ml)
                        trained_ml_models[model_name] = automl.best_estimator

                    elif model_name == "AutoGluon":
                        start_tune = time.time()
                        train_data = X_train_ml.copy()
                        train_data[target] = y_train
                        predictor = TabularPredictor(label=target, verbosity=0, path="ag_temp_model").fit(
                            train_data, time_limit=time_budget, presets='medium_quality'
                        )
                        prep_time = time.time() - start_tune
                        start_train = time.time()
                        
                        preds = predictor.predict(X_test_ml).values

                    train_infer_time = time.time() - start_train
                    preds_dict[model_name] = preds
                    
                    # Calculate Metric
                    metric_val = r2_score(y_test, preds) if task == "Regression" else accuracy_score(y_test, preds)
                    metric_name = "R2 Score" if task == "Regression" else "Accuracy"
                    
                    benchmark_results.append({
                        "Model": model_name,
                        metric_name: metric_val,
                        "Prep Time (s)": prep_time,
                        "Train+Infer Time (s)": train_infer_time,
                        "Total Time (s)": prep_time + train_infer_time
                    })
                    
                except Exception as e:
                    st.error(f"Error in {model_name}: {e}")
            
            my_bar.empty()
            
            # --- Results Rendering ---
            if benchmark_results:
                res_df = pd.DataFrame(benchmark_results)
                
                st.subheader("Integrated Report")
                st.dataframe(res_df.style.format(precision=4), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{metric_name} Comparison**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=res_df, x="Model", y=metric_name, palette="viridis", ax=ax)
                    ax.set_ylim(0, 1.0)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    
                with col2:
                    st.markdown("**Execution Time Comparison**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    res_df.set_index("Model")[["Prep Time (s)", "Train+Infer Time (s)"]].plot(
                        kind="bar", stacked=True, colormap="magma", ax=ax
                    )
                    ax.set_ylabel("Seconds")
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

                st.subheader("Detailed Visual Analysis")
                if task == "Regression":
                    fig_reg = plot_regression(np.asarray(y_test), preds_dict, target)
                    st.pyplot(fig_reg)
                else:
                    fig_clf = plot_classification(np.asarray(y_test), preds_dict, target)
                    st.pyplot(fig_clf)

                # Feature Importance (SHAP)
                shap_candidate = next((m for m in ["XGBoost (Tuned)", "LightGBM", "XGBoost (Default)", "CatBoost"] if m in trained_ml_models), None)
                if shap_candidate:
                    st.subheader(f"Feature Importance (SHAP via {shap_candidate})")
                    fig_shap = generate_shap_plot(trained_ml_models[shap_candidate], X_test_ml)
                    if fig_shap:
                        st.pyplot(fig_shap)
                        
                # Download
                result_df = X_test_fm.copy()
                result_df[f"Actual_{target}"] = np.asarray(y_test)
                for m_name, p in preds_dict.items():
                    result_df[f"Pred_{m_name}"] = p
                
                st.download_button(
                    "Download Full Benchmark Predictions",
                    data=csv_bytes(result_df),
                    file_name="tabtester_benchmark_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ---------------------------------------------------------
    # TAB 3: Impute Missing Values
    # ---------------------------------------------------------
    with tab_impute:
        st.markdown("### Missing Value Imputation")
        st.caption("Use Foundation Models (TabFM / TabICL) to intelligently fill missing values in your dataset.")
        
        missing_mask = df[target].isna()
        missing_count = int(missing_mask.sum())
        
        if missing_count == 0:
            st.info(f"The selected target column '{target}' has no missing values to impute.")
        else:
            st.write(f"**{missing_count}** missing values detected in '{target}'.")
            
            impute_model = st.selectbox("Select Imputation Engine", [m for m in ["TabFM", "TabICLv2"] if m in available_models])
            
            if st.button("Run Imputation", type="primary"):
                out_df = df.copy()
                X_missing = df.loc[missing_mask].drop(columns=[target] + excluded)
                
                try:
                    with st.spinner(f"Imputing with {impute_model}..."):
                        if impute_model == "TabFM":
                            base_model = load_tabfm_base_model(task, device)
                            model = TabFMRegressor(model=base_model) if task == "Regression" else TabFMClassifier(model=base_model)
                        else:
                            model = TabICLRegressor(random_state=random_state) if task == "Regression" else TabICLClassifier(random_state=random_state)
                            
                        # Train on clean data, predict on missing
                        model.fit(X, y.values)
                        preds = model.predict(X_missing)
                        
                        imputed_col = f"{target}_imputed"
                        out_df[imputed_col] = out_df[target]
                        out_df.loc[missing_mask, imputed_col] = preds
                        
                    st.success("Imputation complete!")
                    st.dataframe(out_df.loc[missing_mask, [target, imputed_col] + list(X_missing.columns)].head(50))
                    
                    st.download_button(
                        "Download Imputed Dataset",
                        data=csv_bytes(out_df),
                        file_name=f"dataset_imputed_{target}.csv",
                        mime="text/csv"
                    )
                except Exception as exc:
                    st.error(f"Error during imputation: {exc}")


if __name__ == "__main__":
    main()
