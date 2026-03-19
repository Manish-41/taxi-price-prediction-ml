import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="New York Taxi Price Predictor", layout="wide")
# Load model & columns
model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

st.title("🚖 Taxi Price Prediction Dashboard")
st.markdown("### Predict fare based on trip details")

# Sidebar Inputs
st.sidebar.header("📥 Enter Trip Details")

trip_distance = st.sidebar.slider("Trip Distance (km)", 0.0, 50.0, 5.0)
passenger_count = st.sidebar.slider("Passenger Count", 1, 10, 2)
trip_duration = st.sidebar.slider("Trip Duration (minutes)", 1.0, 120.0, 20.0)

base_fare = st.sidebar.number_input("Base Fare", value=50.0)
per_km_rate = st.sidebar.number_input("Per Km Rate", value=10.0)
per_minute_rate = st.sidebar.number_input("Per Minute Rate", value=2.0)

time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
traffic = st.sidebar.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy"])

# Create Input Data
input_dict = {
    "Trip_Distance_km": trip_distance,
    "Passenger_Count": passenger_count,
    "Trip_Duration_Minutes": trip_duration,
    "Base_Fare": base_fare,
    "Per_Km_Rate": per_km_rate,
    "Per_Minute_Rate": per_minute_rate,
    "Time_of_Day": time_of_day,
    "Day_of_Week": day_of_week,
    "Traffic_Conditions": traffic,
    "Weather": weather
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Layout Columns

col1, col2 = st.columns(2)
# Prediction Section
with col1:
    st.subheader("📊 Prediction")

    if st.button("Predict Price"):
        prediction = model.predict(input_encoded)[0]

        st.metric(label="💰 Estimated Price", value=f"₹ {prediction:.2f}")
# Input Data Preview
with col2:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df)
# Simple Insights Section

st.subheader("📈 Quick Insights")

# Example calculations
estimated_cost = base_fare + (trip_distance * per_km_rate) + (trip_duration * per_minute_rate)

col3, col4, col5 = st.columns(3)

col3.metric("Base Fare", f"$ {base_fare}")
col4.metric("Distance Cost", f"$ {trip_distance * per_km_rate:.2f}")
col5.metric("Time Cost", f"$ {trip_duration * per_minute_rate:.2f}")

st.info(f"📌 Approximate calculated fare (without ML): $ {estimated_cost:.2f}")
# Footer
st.markdown("---")
st.markdown("Built by Manish 🚀 | ML Project")
