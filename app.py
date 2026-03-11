import streamlit as st
import numpy as np
import pickle
import math
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Uber Fare Predictor",
    page_icon="🚕",
    layout="wide"
)
st.title("🚕 Uber Fare Prediction System")
st.write("**Student Details**")
st.write("**Name:** D Poojitha")
st.write("**Registration Number:** 2023BCSE07AED296")
st.write("**Class:** CSE-AIML-C")
st.image("poojitha.jpg", width=200)
st.write("**Google Colab Project:**")
st.markdown("https://colab.research.google.com/drive/1XXA4uWd-xyUBUXdkLIg3SdltrkY9UmmP?usp=sharing")
st.divider()

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

This dataset helps train the model to understand the relationship between trip features and fare prices.
""")

st.header("Features Used")

st.markdown("""
The following features are used as input for the machine learning model:

- **Pickup Latitude**
- **Pickup Longitude**
- **Drop-off Latitude**
- **Drop-off Longitude**
- **Passenger Count**
- **Trip Distance (calculated using the Haversine Formula)**

Distance between pickup and drop locations is computed to better capture the main factor affecting ride fares.
""")

st.header("Machine Learning Method")

st.markdown("""
This project uses the **Random Forest Regression algorithm** for fare prediction.

Random Forest is an ensemble learning technique that:
- Combines multiple decision trees
- Improves prediction accuracy
- Reduces overfitting
- Handles nonlinear relationships effectively

The model was trained and optimized using **GridSearchCV** for better performance.
""")

st.header("System Workflow")

st.markdown("""
The workflow of the system is as follows:

1. **Data Collection** – Load Uber fare dataset.
2. **Data Cleaning** – Remove missing values and invalid entries.
3. **Feature Engineering** – Calculate trip distance using the Haversine formula.
4. **Feature Selection** – Select relevant variables for prediction.
5. **Model Training** – Train regression models such as Linear Regression and Random Forest.
6. **Model Optimization** – Tune hyperparameters using GridSearchCV.
7. **Model Deployment** – Deploy the trained model using Streamlit for interactive predictions.
""")
st.divider()
# Load model
model = pickle.load(open("uber_fare_model.pkl","rb"))

st.title("🚕 Uber Fare Prediction System")

st.markdown(
"""
Estimate Uber ride fare using **Machine Learning**.  
Select pickup and drop locations on the map.
"""
)

st.divider()

# Default NYC coordinates
pickup_lat = 40.761432
pickup_lon = -73.979815

drop_lat = 40.651311
drop_lon = -73.880333


# Map data
map_data = pd.DataFrame({
    'lat':[pickup_lat, drop_lat],
    'lon':[pickup_lon, drop_lon]
})

st.subheader("📍 Trip Location")

st.map(map_data)

st.divider()

col1,col2,col3 = st.columns(3)

with col1:
    pickup_lat = st.number_input("Pickup Latitude", value=pickup_lat)

with col2:
    pickup_lon = st.number_input("Pickup Longitude", value=pickup_lon)

with col3:
    passengers = st.slider("Passengers",1,6,1)

col4,col5 = st.columns(2)

with col4:
    drop_lat = st.number_input("Dropoff Latitude", value=drop_lat)

with col5:
    drop_lon = st.number_input("Dropoff Longitude", value=drop_lon)


# Distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c


st.divider()

if st.button("🚕 Predict Fare"):

    distance = haversine(pickup_lat,pickup_lon,drop_lat,drop_lon)

    features = np.array([[pickup_lat,pickup_lon,drop_lat,drop_lon,passengers,distance]])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Fare: ${prediction:.2f}")

    st.info(f"Trip Distance: {distance:.2f} km")

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

st.caption("🚀 Machine Learning Model: Random Forest | Built using Streamlit")





