import streamlit as st
from pycaret.regression import save_model, finalize_model

# Initialize session state variables
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None
if "compare_models_class" not in st.session_state:
    st.session_state.compare_models_class = None
if "compare_models_pull" not in st.session_state:
    st.session_state.compare_models_pull = None

# Set page configuration
st.set_page_config(
    page_title="Finalization and Saving",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded")

# Get the trained models
model = st.session_state.compare_models_class

# Header
st.header("💾 Finalization and Saving", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Check if models are trained
if model is not None:
    with st.form("Finalization and Saving"):
        best_model = st.selectbox("Choice model", model)
        nom_model = st.text_input("Name your model")
        submit = st.form_submit_button("Save")
    if submit:
        with st.spinner("Saving in progress..."):
            save_model(finalize_model(best_model), model_name=nom_model)
        st.success("Saving successful!")
        # Download the model
        with open(f"{nom_model}.pkl", 'rb') as f:
            st.download_button('Download Model', f, file_name=f"{nom_model}.pkl")
else:
    st.warning("No model trained!")
