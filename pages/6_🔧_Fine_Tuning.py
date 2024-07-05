import streamlit as st
from pycaret.regression import tune_model, compare_models, pull

# Set page configuration
st.set_page_config(
    page_title="Fine Tuning",
    page_icon="🔨",
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

# Header
st.header("🔨 Fine Tuning", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Get the trained models
model = st.session_state.compare_models_class

# Check if models are trained
if model is not None:
    with st.expander("Fine-Tuning", True):
        col_1, col_2 = st.columns(2)
        lib = {
            "Scikit-learn": "scikit-learn",
            "Scikit-optimize": "scikit-optimize",
            "Tune-sklearn": "tune-sklearn",
            "Optuna": "optuna"
        }
        with col_1:
            search_lib = st.selectbox("Search library", lib.keys())
        if search_lib == "Scikit-learn":
            algo = {
                "Random grid search": "random",
                "Grid search": "grid"
            }
        elif search_lib == "Scikit-optimize":
            algo = {
                "Bayesian search": "bayesian"
            }
        elif search_lib == "Tune-sklearn":
            algo = {
                "Random grid search": "random",
                "Grid search": "grid",
                "Bayesian": "bayesian",
                "Hyperopt": "hyperopt",
                "Optuna": "optuna",
                "Bohb": "bohb"
            }
        else:
            algo = {
                "Tree-structured Parzen Estimator search": "tpe",
                "Randomized Search": "random"
            }
        with col_2:
            search_algo = st.selectbox("Search algorithm", algo.keys())
        if st.button("Tune model"):
            st.subheader("Fine tuning in process...")
            col1, col2 = st.columns(2)
            col = 1
            for i in range(len(model)):
                if col == 1:
                    with col1:
                        with st.spinner(f"Fine tuning of model {i+1}"):
                            model[i] = tune_model(model[i], search_library=lib[search_lib], search_algorithm=algo[search_algo])
                            st.write(f"Model {i+1}")
                            st.dataframe(pull())
                            st.write(model[i])
                    col = 2
                else:
                    with col2:
                        with st.spinner(f"Fine tuning of model {i+1}"):
                            model[i] = tune_model(model[i], search_library=lib[search_lib], search_algorithm=algo[search_algo])
                            st.write(f"Model {i+1}")
                            st.dataframe(pull())
                            st.write(model[i])
                    col = 1
            st.success("Fine tuning done!")
            with st.spinner("Comparaison of tuned model in process"):
                st.session_state.compare_models_class = compare_models(model, sort=st.session_state.metric, n_select=len(model))
            st.write(pull().style.highlight_max(axis=0), use_container_width=True)
            st.success("Comparaison of tuned models done!")
else:
    st.warning("No model trained!")
