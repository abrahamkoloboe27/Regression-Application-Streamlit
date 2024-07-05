import streamlit as st
from pycaret.regression import plot_model

# Set page configuration
st.set_page_config(
    page_title="Regression Plots",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

# Initialize session state variables
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None
if "compare_models_class" not in st.session_state:
    st.session_state.compare_models_class = None
if "compare_models_pull" not in st.session_state:
    st.session_state.compare_models_pull = None

# Dictionary of available plots
dic = {
    "Schematic drawing of the preprocessing pipeline": "pipeline",
    "Interactive Residual plots": "residuals_interactive",
    "Residuals Plot": "residuals",
    "Prediction Error Plot": "error",
    "Cooks Distance Plot": "cooks",
    "Recursive Feat. Selection": "rfe",
    "Learning Curve": "learning",
    "Validation Curve": "vc",
    "Manifold Learning": "manifold",
    "Feature Importance": "feature",
    "Feature Importance (All)": "feature_all",
    "Model Hyperparameter": "parameter",
    "Decision Tree": "tree",
}

# Header
st.header("📈 Regression plots", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Get the trained model
model = st.session_state.compare_models_class

# Check if a model is trained
if model is not None:
    with st.expander("Regression Plots", True):
        mod = st.selectbox("Select a model", model)
        plot_ = st.multiselect("Select a plot", options=dic.keys(),
                               default=["Residuals Plot", "Feature Importance", "Learning Curve", "Prediction Error Plot", "Recursive Feat. Selection"])
        plot_reg_ = st.button("Plot")
    col_1, col_2 = st.columns(2)
    i = 1
    if plot_reg_:
        for plt in plot_:
            if plt != "Feature Importance":
                if i == 1:
                    with col_1:
                        st.subheader(plt)
                        try:
                            plot_model(estimator=mod, plot=dic[plt], display_format='streamlit')
                        except:
                            st.warning("Unable to display this plot")
                        i = 2
                else:
                    with col_2:
                        st.subheader(plt)
                        try:
                            plot_model(estimator=mod, plot=dic[plt], display_format='streamlit')
                        except:
                            st.warning("Unable to display this plot")
                        i = 1
            else:
                if i == 1:
                    with col_1:
                        st.subheader(plt)
                        try:
                            plot_model(estimator=mod, plot=dic[plt], display_format='streamlit', save=True)
                            st.image("Feature Importance.png")
                        except:
                            st.warning("Unable to display this plot")
                        i = 2
                else:
                    with col_2:
                        st.subheader(plt)
                        try:
                            plot_model(estimator=mod, plot=dic[plt], display_format='streamlit', save=True)
                            st.image("Feature Importance.png")
                        except:
                            st.warning("Unable to display this plot")
                        i = 1
else:
    st.warning("No model trained ! ")
