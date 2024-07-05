import streamlit as st
import pandas as pd
from pycaret.datasets import get_data
# Set page configuration
st.set_page_config(
    page_title="Import your data",
    page_icon="📥 ",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("📥 Import your data ", divider="rainbow")
# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "data" not in st.session_state:
    st.session_state.data = None
if "columns_to_use" not in st.session_state:
    st.session_state.columns_to_use = None
if "target_variable" not in st.session_state:
    st.session_state.target_variable = None

# Function to load data based on format
@st.cache_data
def load_data(file, formats, sp):
    if formats == "csv":
        data = pd.read_csv(file, sep=sp)
    else:
        data = pd.read_excel(file)
    return data
with st.expander("Data", True):
    data_selection = st.radio("Select data source", ["Use sample data", "Load data"], horizontal =True)
    if data_selection=="Use sample data":
        data_options = st.selectbox("Select a dataset", ["insurance", "diamond", "house", "charges"])
        if data_options == "insurance":
            st.markdown("## Insurance Dataset")
            st.markdown("This dataset contains information about insurance charges. It includes features such as age, sex, BMI, number of children, smoker status, and region. The target variable is the insurance charges.")
        elif data_options == "diamond":
            st.markdown("## Diamond Dataset")
            st.markdown("This dataset contains information about diamond prices. It includes features such as carat weight, cut, color, clarity, depth, table, and dimensions. The target variable is the price of the diamond.")
        elif data_options == "house":
            st.markdown("## House Dataset")
            st.markdown("This dataset contains information about house prices. It includes features such as number of bedrooms, number of bathrooms, size of the living area, size of the lot, and location. The target variable is the price of the house.")
        else:   
            st.markdown("## Charges Dataset")
            st.markdown("This dataset contains information about charges. It includes features such as category, description, quantity, and price. The target variable is the price of the charges.")
        if st.button("Load data"):    
            st.session_state.data = get_data(data_options)
            st.dataframe(st.session_state.data.head(), use_container_width=True)
        file = 1
    else :
        # Expander to load data
        
            col_1, col_2 = st.columns([2, 6])
            with col_1:
                formats = ["csv", "xlsx", "xls"]
                selected_format = st.radio('Format', formats, horizontal=True)
                if selected_format == "csv":
                    sep = st.radio("", [",", ";"], horizontal=True)
            with col_2:
                file = st.file_uploader("Upload your data here", type=[selected_format])

# If file is uploaded, display data and filter options
if file is not None:
    with st.expander("Data", True):
        if file == 1:
            pass
        else:
            st.session_state.data = load_data(file, selected_format, sep)
        if st.toggle("Show data"):
            st.dataframe(st.session_state.data, use_container_width=True)
        columns_to_use = []
        if st.session_state.data is not None : 
            if st.toggle("Filter dataset", True):
                c_1, c_2 = st.columns(2)
                with c_1:
                    st.subheader("Select columns")
                    
                    columns_to_use = st.multiselect("\nSelect columns to exclude for regression",
                                                    options=st.session_state.data.columns)
                with c_2:
                    if st.checkbox("Use all rows", False):
                        pass
                    else:
                        st.subheader("Number of rows")
                        n_rows = st.slider("Number of rows",
                                        min_value=int(len(st.session_state.data) / 10),
                                        value=int(4 * len(st.session_state.data) / 10),
                                        max_value=len(st.session_state.data), step=1)

                valider = st.button("Exclude")

                if valider:
                    st.session_state.df = st.session_state.data.drop(columns_to_use, axis=1)
                    st.session_state.df = st.session_state.df.iloc[:n_rows]
                    st.dataframe(st.session_state.df, use_container_width=True)
            else:
                st.session_state.df = st.session_state.data
                columns_to_use = st.session_state.data.columns
        st.session_state.columns_to_use = columns_to_use
