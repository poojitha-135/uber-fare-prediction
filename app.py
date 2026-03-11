import streamlit as st
import numpy as np
import pickle
import math
import pandas as pd
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# Initialize geocoder
geolocator = Nominatim(user_agent="uber_fare_app")

st.set_page_config(
    page_title="Uber Fare Predictor",
    page_icon="🚕",
    layout="wide"
)

# ---------------- STUDENT DETAILS ---------------- #

st.title("🚕 Uber Fare Prediction System")

st.write("### Student Details")
st.write("**Name:** D Poojitha")
st.write("**Registration Number:** 2023BCSE07AED296")
st.write("**Class:** CSE-AIML-C")

st.image("poojitha.jpg", width=200)

st.write("**Google Colab Project:**")
st.markdown("https://colab.research.google.com/drive/1XXA4uWd-xyUBUXdkLIg3SdltrkY9UmmP?usp=sharing")

st.divider()

# ---------------- PROJECT DESCRIPTION ---------------- #

st.header("Project Overview")

st.markdown("""
This project develops a **Machine Learning based system to predict Uber ride fares** using trip-related information such as pickup and drop-off locations, passenger count, and travel distance.  
The system estimates the fare amount by learning patterns from historical Uber trip data and provides predictions through an interactive **Streamlit web application**.
""")

st.header("Problem Statement")

st.markdown("""
Ride-hailing platforms such as Uber calculate fares based on multiple factors including **distance, passenger count, and trip location**.  
The objective of this project is to build a **regression-based machine learning model** that can accurately predict the expected fare amount for a trip using historical ride data.
""")

st.header("Dataset Information")

st.markdown("""
The dataset used for this project is the **Uber Fares Dataset from Kaggle**.

It contains historical trip records including:
- Pickup latitude and longitude
- Drop-off latitude and longitude
- Passenger count
- Fare amount
""")

st.header("Features Used")

st.markdown("""
- Pickup Latitude  
- Pickup Longitude  
- Drop-off Latitude  
- Drop-off Longitude  
- Passenger Count  
- Distance (calculated using the **Haversine Formula**)
""")

st.header("Machine Learning Method")

st.markdown("""
This project uses **Random Forest Regression** for predicting the Uber fare.

Random Forest:
- Combines multiple decision trees
- Improves prediction accuracy
- Reduces overfitting
- Handles nonlinear relationships effectively
""")

st.header("System Workflow")

st.markdown("""
1. Data Collection  
2. Data Cleaning  
3. Feature Engineering (Haversine Distance)  
4. Feature Selection  
5. Model Training (Random Forest)  
6. Hyperparameter Optimization  
7. Model Deployment using Streamlit
""")

st.divider()

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("uber_fare_model.pkl", "rb"))

# ---------------- USER INPUT ---------------- #

st.header("🚕 Fare Prediction")

pickup_address = st.text_input(
    "Enter Pickup Location",
    "Alliance University Anekal Road Bangalore"
)

drop_address = st.text_input(
    "Enter Drop Location",
    "Electronic City Bangalore"
)

passengers = st.slider("Passenger Count", 1, 6, 1)

# ---------------- GEOCODING FUNCTION ---------------- #

def get_coordinates(address):
    location = geolocator.geocode(address)

    if location:
        return location.latitude, location.longitude
    else:
        return None, None


# ---------------- DISTANCE FUNCTION ---------------- #

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1, lon1, lat2, lon2 = map(math.radians,[lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


st.divider()

# ---------------- PREDICTION ---------------- #

if st.button("🚕 Predict Fare"):

    pickup_lat, pickup_lon = get_coordinates(pickup_address)
    drop_lat, drop_lon = get_coordinates(drop_address)

    if pickup_lat is None or drop_lat is None:
        st.error("Invalid address. Please enter a valid location.")
    else:

        distance = haversine(pickup_lat, pickup_lon, drop_lat, drop_lon)

        features = np.array([[pickup_lat, pickup_lon,
                              drop_lat, drop_lon,
                              passengers,
                              distance]])

        prediction = model.predict(features)[0]

        st.success(f"Estimated Fare: ${prediction:.2f}")
        st.info(f"Trip Distance: {distance:.2f} km")

        # Map display
        map_data = pd.DataFrame({
            'lat':[pickup_lat, drop_lat],
            'lon':[pickup_lon, drop_lon]
        })

        st.subheader("Trip Route Map")
        st.map(map_data)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text':"Estimated Fare ($)"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"green"}
            }
        ))

        st.plotly_chart(fig)

st.divider()

st.caption("🚀 Machine Learning Model: Random Forest | Built with Streamlit")
