import streamlit as st
import pandas as pd
from pycaret.regression import pull, compare_models

# Set page configuration
st.set_page_config(
    page_title="Train Models",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize session state variables
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None
if "compare_models_class" not in st.session_state:
    st.session_state.compare_models_class = None
if "compare_models_pull" not in st.session_state:
    st.session_state.compare_models_pull = None
if "metric" not in st.session_state:
    st.session_state.metric = None

# Dictionary of available models
dic = {
    "Linear Regression": "lr",
    "Lasso Regression": "lasso",
    "Ridge Regression": "ridge",
    "Elastic Net": "en",
    "Least Angle Regression": "lar",
    "Lasso Least Angle Regression": "llar",
    "Orthogonal Matching Pursuit": "omp",
    "Bayesian Ridge": "br",
    "Automatic Relevance Determination": "ard",
    "Passive Aggressive Regressor": "par",
    "Random Sample Consensus": "ransac",
    "TheilSen Regressor": "tr",
    "Huber Regressor": "huber",
    "Kernel Ridge": "kr",
    "SVM - Linear Kernel": "svm",
    "K Neighbors Regressor": "knn",
    "Decision Tree Regressor": "dt",
    "Random Forest Regressor": "rf",
    "Extra Trees Regressor": "et",
    "Ada Boost Regressor": "ada",
    "Gradient Boosting Regressor": "gbr",
    "MLP Regressor": "mlp",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting Machine": "lightgbm",
    "CatBoost Regressor": "catboost"
}

# Number of available models
n_mod = len(dic)

# Header
st.header("🤖Training models", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)
# Default models to exclude/include
exclude = ["Light Gradient Boosting Machine", "CatBoost Regressor", "Gradient Boosting Regressor", "Extreme Gradient Boosting"]
include = ["Linear Regression", "Elastic Net", "Ridge Regression", "K Neighbors Regressor", "Random Forest Regressor"]

# Check if setup is available
setup_class = st.session_state.setup_class
if setup_class is not None:
    with st.form(""" # Train models"""):
        include_exclude = st.selectbox("Include/Exclude models", ["Include", "Exclude"])
        if include_exclude == "Exclude":
            models = st.multiselect("Select models to exclude ", options=dic.keys(), default=exclude)
        else:
            models = st.multiselect("Select models to include ", options=dic.keys(), default=include)
        mod = [dic[i] for i in models]
        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader("Metric")
            metric = st.selectbox("Select a metric", ["RMSE", "R2", "MAPE", "MSE", "MAE", "RMSLE"])
            st.session_state.metric = metric
        with col_2:
            st.subheader("Number of models to save")
            num_models = st.slider("Number of models", max_value=(n_mod - len(exclude)), min_value=1, value=int((1 * (n_mod - len(exclude))) / 4))
        train_boutton = st.form_submit_button("🏄 Train !")
    if train_boutton:
        with st.spinner("Training in process..."):
            if include_exclude == "Exclude":
                compare_models_class = compare_models(exclude=mod, sort=metric, n_select=num_models, turbo=False)
            else:
                compare_models_class = compare_models(include=mod, sort=metric, n_select=num_models, turbo=False)
        compare_models_pull = pull()
        st.session_state.compare_models_class = compare_models_class
        st.session_state.compare_models_pull = compare_models_pull
        st.success("Training successful ! ")
        st.dataframe(compare_models_pull.style.highlight_max(axis=0), use_container_width=True)
else:
    st.warning("No setup ! ")
