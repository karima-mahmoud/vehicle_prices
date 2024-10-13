import streamlit as st
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Load the trained model
model_path = '/mnt/data/RandomForestRegressor.pkl'
model = joblib.load(model_path)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
main_container = st.container()
with main_container:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Diabetes Prediction App")
        st.write("This app predicts the likelihood of diabetes based on various health metrics.")
        
        # Input features
        st.subheader("Input Features")
        age = st.slider("Age", 0, 100, 25)
        bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=180, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        
        # Prepare the input array for prediction
        features = np.array([[age, bmi, glucose, insulin, blood_pressure, skin_thickness, dpf]])
        
        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(features)
            st.markdown(f"<div class='prediction-result'>The predicted likelihood of diabetes is: {prediction[0]:.2f}</div>", unsafe_allow_html=True)
    
    with col2:
        # Load and display Lottie animation
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=300, key="lottie")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This Diabetes Prediction App uses machine learning to estimate the likelihood of diabetes based on various health metrics. It's important to note that this tool is for informational purposes only and should not be considered as a substitute for professional medical advice.")

# Footer
st.markdown("---")
st.markdown("Developed by Hagar Sherif | © 2024")

# Hide Streamlit footer and hamburger menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
