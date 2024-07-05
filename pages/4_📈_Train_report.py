import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Train Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize session state variables
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None
if "compare_models_class" not in st.session_state:
    st.session_state.compare_models_class = None
if "compare_models_pull" not in st.session_state:
    st.session_state.compare_models_pull = None

# Function to get top models by metric
@st.cache_data
def top_models_by_metric(pull_report, metric, m):
    # Sort the pull_report DataFrame by the specified metric (in descending order for R2, ascending order for others)
    if metric == "R2":
        pull_report_sorted = pull_report.sort_values(by=metric, ascending=False)
    else:
        pull_report_sorted = pull_report.sort_values(by=metric, ascending=True)

    # Select the top m rows (i.e., the top m performing models)
    top_m_models = pull_report_sorted.head(m)

    return top_m_models

# Header
st.header("📈  Train Report", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Get the pull report and best model
pull_report = st.session_state.compare_models_pull
best_model = st.session_state.compare_models_class

# Check if a model is trained
if best_model is not None:
    # Sidebar with number of models to compare and metrics to use
    num_compare = st.sidebar.slider("Number of models to compare", min_value=2, max_value=len(pull_report), value=int((len(pull_report) * 3) / 4))
    metric = st.sidebar.multiselect("Metric", pull_report.drop("Model", axis=1).columns, default=["RMSE", "R2", "MAPE", "MSE"])

    # Plot bar charts for each metric
    for metrics in metric:
        to_plot = top_models_by_metric(pull_report, metrics, num_compare)
        fig = px.bar(to_plot, x=to_plot.index, y=metrics, color="Model", title=f"Model comparison by {metrics}")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No model trained ! ")
