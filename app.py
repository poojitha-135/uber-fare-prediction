import streamlit as st
import numpy as np
import pickle
import math

model = pickle.load(open("uber_fare_model.pkl","rb"))

st.title("Uber Fare Prediction")

pickup_lat = st.number_input("Pickup Latitude")
pickup_lon = st.number_input("Pickup Longitude")
drop_lat = st.number_input("Dropoff Latitude")
drop_lon = st.number_input("Dropoff Longitude")
passengers = st.number_input("Passenger Count", min_value=1)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

if st.button("Predict Fare"):
    distance = haversine(pickup_lat,pickup_lon,drop_lat,drop_lon)

    features = np.array([[pickup_lat,pickup_lon,drop_lat,drop_lon,passengers,distance]])

    prediction = model.predict(features)

    st.success(f"Estimated Fare: ${prediction[0]:.2f}")