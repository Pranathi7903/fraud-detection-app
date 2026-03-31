import streamlit as st
import pickle
import numpy as np

# Load model and scaler safely
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Enter transaction details below")

# Warning note
st.warning("⚠️ This is a predictive system and may not be 100% accurate.")

# Sidebar (better UI)
st.sidebar.header("Input Features")

# User Inputs
amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)

v1 = st.slider("V1", -10.0, 10.0, 0.0)
v2 = st.slider("V2", -10.0, 10.0, 0.0)
v3 = st.slider("V3", -10.0, 10.0, 0.0)

# Prediction Button
if st.button("Predict"):

    # Create full feature array (30 features)
    input_data = np.zeros((1, 30))

    # Assign values correctly
    input_data[0][0] = 0        # Time (default)
    input_data[0][1] = v1
    input_data[0][2] = v2
    input_data[0][3] = v3
    input_data[0][-1] = amount  # Amount is last column

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Normal Transaction")

    # Show probability
    st.write(f"Fraud Probability: {probability[0][1]*100:.2f}%")