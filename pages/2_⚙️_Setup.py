import streamlit as st
import pandas as pd
from pycaret.regression import setup, pull

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "data" not in st.session_state:
    st.session_state.data = None
if "columns_to_use" not in st.session_state:
    st.session_state.columns_to_use = None
if "target_variable" not in st.session_state:
    st.session_state.target_variable = None
if "setup_class" not in st.session_state:
    st.session_state.setup_class = None

# Set page configuration
st.set_page_config(
    page_title="Regression Setup",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded")
st.header("⚙️Regression Setup", divider='rainbow')
# Check if data is available
if st.session_state.df is not None:
    with st.form("Setup"):
        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader("Target variable")
            target_variable = st.selectbox("Choose the target variable", st.session_state.df.columns)
        st.session_state.target_variable = target_variable

        with col_2:
            st.subheader("Train Percentage")
            train_size = st.slider("Train size", min_value=0.0, value=75.0, max_value=100.0)

        with col_1:
            st.write("\n")
            st.subheader("Numeric imputation")
            dic = {
                "mean": "Mean of column",
                "drop": "Drop rows containing missing values",
                "median": "Median of column",
                "mode": "Most frequent value",
                "knn": "Using a K-Nearest Neighbors approach",
                "int or float": "Impute with provided numerical value"
            }
            dic2 = {v: k for k, v in dic.items()}
            st.write()
            choice = st.selectbox("Choose a method", dic.values())
            num_imputation = dic2[choice]
            if choice == "Impute with provided numerical value":
                type_num = st.selectbox("Int or Float", ["Int", "Float"])
                if type_num == "Int":
                    num_imputation = int(st.number_input("Int"))
                else:
                    num_imputation = float(st.number_input("Float"))

        with col_2:
            st.subheader("Categorical imputation")
            dic = {
                "mode": "Most frequent value",
                "drop": "Drop rows containing missing values",
                "str": "Impute with provided string"
            }
            dic2 = {v: k for k, v in dic.items()}
            choice = st.selectbox("Choose a method", dic.values())
            cat_imputation = dic2[choice]
            if choice == "Impute with provided string":
                cat_imputation = st.text_input("Provide a string")

        st.subheader("Normalization")
        dic = {
            "ZScore": "zscore",
            "MinMax Scaler": "minmax",
            "MaxAbs Scaler": "maxabs",
            "Robust Scaler": "robust"
        }
        norm_method = dic[st.selectbox("Select a method", dic.keys())]

        st.subheader("Cross Validation")
        col_1, col_2 = st.columns(2)
        dic = {
            "K-Fold": "kfold",
            "Group K-Fold": "groupkfold",
            "Time Series": "timeseries"
        }
        with col_1:
            strategy = dic[st.selectbox("Fold Strategy", dic.keys())]
        with col_2:
            num_fold = int(st.number_input("Number of folds", min_value=2, value=5, max_value=int(len(st.session_state.df[target_variable]) / 5)))

        perform_button = st.form_submit_button("Lancer setup")

    if perform_button:
        setup_class = setup(data=st.session_state.df,
                            target=target_variable,
                            train_size=train_size / 100,
                            numeric_imputation=num_imputation,
                            categorical_imputation=cat_imputation,
                            normalize=True,
                            normalize_method=norm_method,
                            fold_strategy=strategy,
                            fold=num_fold)
        st.session_state.setup_class = setup_class
        setup_pull = pull()

        st.header("Setup of training", divider='rainbow')
        st.dataframe(setup_pull, use_container_width=True)
else:
    st.warning("No data !")
