# New York Taxi Price Prediction System

##  Overview
This project is an end-to-end Machine Learning application that predicts taxi fares based on trip details such as distance, duration, traffic conditions, and weather.

The system includes data preprocessing, model training, and an interactive dashboard built using Streamlit.

---

## Features
- Data cleaning and preprocessing
- Feature engineering using one-hot encoding
- Machine learning model using Random Forest Regressor
- Model evaluation using R² Score and MSE
- Interactive dashboard using Streamlit
- Real-time price prediction

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Input Features
- Trip Distance (km)
- Passenger Count
- Trip Duration (minutes)
- Base Fare
- Per Km Rate
- Per Minute Rate
- Time of Day
- Day of Week
- Traffic Conditions
- Weather

---

## Output
- Predicted Taxi Fare (₹)

---

## How It Works
1. Data is preprocessed (handling missing values, encoding categorical features)
2. Features are transformed using `get_dummies`
3. Model is trained using Random Forest Regressor
4. Input data is aligned with training features
5. Prediction is generated in real-time

---

## Project Structure
taxi-price-prediction/
│
├── app/ # Streamlit app
├── src/ # Training scripts
├── models/ # Saved model files
├── data/ # Dataset
├── notebooks/ # EDA notebooks
├── requirements.txt
└── README.md
