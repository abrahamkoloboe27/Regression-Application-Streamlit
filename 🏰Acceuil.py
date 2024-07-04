import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Welcome",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded")

# Header
st.header("👋 Welcome to the Regression App 📊", divider='rainbow')

# About the author section
if st.sidebar.toggle("About the author", True):
    with st.expander("Author", True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image("images/About the author.png")
        with c2:
            st.header("""**S. Abraham Z. KOLOBOE**""")
            st.markdown("""
            *:blue[Data Scientist | Engineer in Mathematics and Modeling]*

            Hello,

            I am Abraham, a Data Scientist and Engineer in Mathematics and Modeling.
            My expertise lies in the fields of data science and artificial intelligence.
            With a technical and concise approach, I am committed to providing efficient and accurate solutions in my projects.

            * Email : <abklb27@gmail.com>
            * WhatsApp : +229 91 83 84 21
            * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
            """)

# Sidebar with author information
with st.sidebar:
    st.markdown("""
    ## Author
    :blue[Abraham KOLOBOE]
    * Email : <abklb27@gmail.com>
    * WhatsApp : +229 91 83 84 21
    * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
    """)

# Application description
st.write("This application allows you to perform end-to-end regression analysis on your data. Here's how it works:")

# Page descriptions
pages = {
    "📥 Import your data": "Upload your data and select the target variable.",
    "🔧 Setup": "Configure the preprocessing pipeline and training settings.",
    "🤖 Train models": "Train multiple regression models and compare their performance.",
    "🔨 Fine-tuning": "Fine-tune the selected models to improve their performance.",
    "📈 Regression plots": "Visualize the performance of the trained models.",
    "🔮 Make an inference": "Make predictions using the trained models.",
    "💾 Finalization and saving": "Finalize and save the best model for deployment."}

# Display page descriptions
for page, description in pages.items():
    st.subheader(page)
    st.write(description)
