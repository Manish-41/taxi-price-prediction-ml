import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

st.title("🚖 Taxi Trip Price Prediction")

st.write("Enter trip details below:")

# =========================
# User Inputs
# =========================

trip_distance = st.number_input("Trip Distance (km)", min_value=0.0)
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10)

traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy"])

base_fare = st.number_input("Base Fare")
per_km_rate = st.number_input("Per Km Rate")
per_minute_rate = st.number_input("Per Minute Rate")
trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0)

# =========================
# Create DataFrame
# =========================

input_data = pd.DataFrame({
    "Trip_Distance_km": [trip_distance],
    "Time_of_Day": [time_of_day],
    "Day_of_Week": [day_of_week],
    "Passenger_Count": [passenger_count],
    "Traffic_Conditions": [traffic],
    "Weather": [weather],
    "Base_Fare": [base_fare],
    "Per_Km_Rate": [per_km_rate],
    "Per_Minute_Rate": [per_minute_rate],
    "Trip_Duration_Minutes": [trip_duration]
})

# =========================
# Prediction
# =========================

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Trip Price: ₹ {prediction[0]:.2f}")