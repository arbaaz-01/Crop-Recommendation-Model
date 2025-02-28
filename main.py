import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained stacking model and label encoder
@st.cache_resource  # Cache to avoid reloading on every run
def load_model_and_encoder():
    with open('lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()

# Define feature engineering function (same as training)
def create_features(input_data):
    input_data['rainfall_humidity'] = input_data['rainfall'] * input_data['humidity']
    input_data['pH_K'] = input_data['ph'] * input_data['K']
    input_data['N_P'] = input_data['N'] * input_data['P']
    input_data['temp_sq'] = input_data['temperature'] ** 2
    input_data['rainfall_log'] = np.log1p(input_data['rainfall'])
    input_data['climate_index'] = input_data['temperature'] * input_data['humidity'] / 100
    return input_data

# Streamlit UI
st.title("Crop Recommendation System")
st.markdown("""
    Enter the soil and environmental parameters below to get a crop recommendation 
    based on a machine learning model with 99.84% accuracy.
""")

# Input fields for the 7 base parameters
st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N) [ppm]", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
    P = st.number_input("Phosphorus (P) [ppm]", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
    K = st.number_input("Potassium (K) [ppm]", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
    temperature = st.number_input("Temperature [°C]", min_value=0.0, max_value=50.0, value=20.8, step=0.1)

with col2:
    humidity = st.number_input("Humidity [%]", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall [mm]", min_value=0.0, max_value=500.0, value=202.9, step=1.0)

# Button to predict
if st.button("Recommend Crop"):
    # Create input dictionary
    input_data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    
    # Convert to DataFrame and add engineered features
    input_df = pd.DataFrame([input_data])
    input_df = create_features(input_df)
    
    # Ensure all 13 features are present in the correct order
    expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
                        'rainfall_humidity', 'pH_K', 'N_P', 'temp_sq', 'rainfall_log', 'climate_index']
    input_df = input_df[expected_columns]
    
    # Predict
    prediction = model.predict(input_df)
    crop_name = le.inverse_transform(prediction)[0]
    
    
    # Optional: Show prediction probability
    probas = model.predict_proba(input_df)[0]
    top_proba = np.max(probas) * 100
    if top_proba<=85:
        st.success(f"Recommended Crop: **Unknown**")
    else:
        st.success(f"Recommended Crop: **{crop_name.capitalize()}**")

    # Display result
    st.write(f"Confidence: {top_proba:.2f}%")
# Footer
st.markdown("""
    ---
    *Model Accuracy: 99.84% | Data: Crop Recommendation Dataset*
""")