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

st.markdown("""
### Project Description
This project predicts the **estimated Uber fare** based on trip distance, pickup and drop-off locations, and passenger count using a **Machine Learning Random Forest Regression model**.  
The system calculates distance using the **Haversine formula** and provides fare estimation through an interactive Streamlit web application.
""")

st.write("**Name:** D Poojitha")
st.write("**Registration Number:** 2023BCSE07AED296")
st.write("**Class:** CSE-AIML-C")
st.image("poojitha.jpg", width=200)
st.write("**Google Colab Project:**")
st.markdown("https://colab.research.google.com/drive/1XXA4uWd-xyUBUXdkLIg3SdltrkY9UmmP?usp=sharing")
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




