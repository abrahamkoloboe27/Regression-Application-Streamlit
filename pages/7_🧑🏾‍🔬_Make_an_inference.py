import streamlit as st
from pycaret.regression import predict_model, pull
import plotly.express as px
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Make an Inference",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded")

# Function to get input value based on column data type
def get_input_value(column):
    if st.session_state.df[column].dtype == "int64" or st.session_state.df[column].dtype == "int32":
        return int(st.number_input(f"{column}", min_value=st.session_state.df[column].min(), max_value=st.session_state.df[column].max(), value=st.session_state.df[column].min()))
    elif st.session_state.df[column].dtype == "float64":
        return st.number_input(f"{column}", min_value=st.session_state.df[column].min(), max_value=None, value=st.session_state.df[column].mean())
    else:
        return st.selectbox(f"{column}", st.session_state.df[column].unique())

# Initialize session state variables
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None
if "compare_models_class" not in st.session_state:
    st.session_state.compare_models_class = None
if "compare_models_pull" not in st.session_state:
    st.session_state.compare_models_pull = None
if "columns_to_use" not in st.session_state:
    st.session_state.columns_to_use = None
if "target_variable" not in st.session_state:
    st.session_state.target_variable = None
if "df" not in st.session_state:
    st.session_state.df = None

# Header
st.header("Make an Inference", divider='rainbow')

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Check if a model is trained
if st.session_state.compare_models_class is not None:
    with st.expander("Inference", True):
        model_used = st.selectbox("Select a model", st.session_state.compare_models_class)
        st.subheader("Enter the characteristics")
        col_1, col_2, col_3 = st.columns(3)
        columns_to_use = st.session_state.columns_to_use
        i = 1
        params_name = []
        params_value = []
        dic = {}
        for columns in st.session_state.df.drop(st.session_state.target_variable, axis=1).columns:
            if i == 1:
                with col_1:
                    params_value = get_input_value(columns)
            elif i == 2:
                with col_2:
                    params_value = get_input_value(columns)
            else:
                with col_3:
                    params_value = get_input_value(columns)
            dic[columns] = [params_value]
            i = (i % 3) + 1
        to_predict = pd.DataFrame(dic)
        predict_button = st.button("Predict")
    if predict_button:
        predict_dataframe = predict_model(model_used, to_predict)["prediction_label"]
        st.metric(f"""**:red[Predicted value : {st.session_state.target_variable}]**""", predict_dataframe, delta_color="inverse")
else:
    st.warning("No model trained!")
