import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("💨 Medical Charges Prediction App")
st.write("This app predicts insurance charges based on Age, BMI, and Smoking status using a Linear Regression model.")

# User inputs
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=60.0, value=25.0)
smoker = st.selectbox("Are you a smoker?", ["No", "Yes"])

# Convert categorical input to numeric
smoker_yes = 1 if smoker == "Yes" else 0

# Predict button
if st.button("Predict Charges"):
    # Prepare features for prediction
    features = np.array([[age, bmi, smoker_yes]])

    # Make prediction
    prediction = model.predict(features)

    # Display result
    st.success(f"💰 Estimated Insurance Charges: ${prediction[0]:,.2f}")
