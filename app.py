import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Airline Fare Prediction", layout="wide")

# Title
st.title("Airline Fare Prediction")
st.markdown("""
This application predicts airline ticket prices based on various features using a Random Forest model.
Enter flight details below to get a predicted fare.
""")

@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_pipeline_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Confirm model features
expected_columns = [
    'Carrier', 'MktMilesFlown', 'NonStopMiles', 'Market_HHI', 'Market_share',
    'MktCoupons', 'DaysToDeparture', 'DepartureMonth', 'DepartureDayOfWeek',
    'CarrierPax', 'Pax', 'LCC_Comp', 'FarePerMile', 'CarrierShare',
    'IsWeekendDeparture', 'IsLCC'
]
st.success("✅ Model loaded successfully with exact training features.")

# Sidebar inputs
st.sidebar.header("Flight Details")
carrier = st.sidebar.selectbox("Carrier", [
    'Other', 'American', 'Alaska', 'JetBlue', 'Delta', 'Frontier', 'United', 'Hawaiian',
    'Southwest', 'Spirit', 'Allegiant', 'Sun Country', 'Virgin America', 'SkyWest',
    'ExpressJet', 'Envoy Air', 'PSA Airlines', 'Mesa Airlines', 'Endeavor Air', 'Horizon Air',
    'US Airways', 'Republic Airways', 'Compass Airlines', 'GoJet Airlines', 'Trans States Airlines'
])
carrier_code = [
    'Other', 'American', 'Alaska', 'JetBlue', 'Delta', 'Frontier', 'United', 'Hawaiian',
    'Southwest', 'Spirit', 'Allegiant', 'Sun Country', 'Virgin America', 'SkyWest',
    'ExpressJet', 'Envoy Air', 'PSA Airlines', 'Mesa Airlines', 'Endeavor Air', 'Horizon Air',
    'US Airways', 'Republic Airways', 'Compass Airlines', 'GoJet Airlines', 'Trans States Airlines'
].index(carrier)

days_to_departure = st.sidebar.slider("Days to Departure", 1, 90, 30)
departure_month = st.sidebar.slider("Departure Month", 1, 12, 6)
departure_day = st.sidebar.slider("Departure Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
mkt_miles = st.sidebar.number_input("Market Miles Flown", value=1000)
non_stop_miles = st.sidebar.number_input("Non-Stop Miles", value=1000)
carrier_pax = st.sidebar.number_input("Carrier Passengers", value=10000)
pax = st.sidebar.number_input("Total Passengers", value=50000)
lcc_comp = st.sidebar.number_input("Low-Cost Carrier Competition", value=10)
market_hhi = st.sidebar.number_input("Market HHI", value=2500)
market_share = st.sidebar.number_input("Market Share", value=0.5)
mkt_coupons = st.sidebar.number_input("Market Coupons", value=1)

# Derived features
fare_per_mile = 0.25
carrier_share = carrier_pax / pax if pax > 0 else 0
is_weekend_departure = 1 if departure_day in [5, 6] else 0
is_lcc = 1 if lcc_comp > 0 else 0

# Input DataFrame
input_data = pd.DataFrame({
    'Carrier': [carrier_code],
    'MktMilesFlown': [mkt_miles],
    'NonStopMiles': [non_stop_miles],
    'Market_HHI': [market_hhi],
    'Market_share': [market_share],
    'MktCoupons': [mkt_coupons],
    'DaysToDeparture': [days_to_departure],
    'DepartureMonth': [departure_month],
    'DepartureDayOfWeek': [departure_day],
    'CarrierPax': [carrier_pax],
    'Pax': [pax],
    'LCC_Comp': [lcc_comp],
    'FarePerMile': [fare_per_mile],
    'CarrierShare': [carrier_share],
    'IsWeekendDeparture': [is_weekend_departure],
    'IsLCC': [is_lcc]
})

# Ensure columns in right order
input_data = input_data[expected_columns]

# Predict
try:
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Fare")
    st.write(f"The predicted average fare is **${prediction:.2f}**")
except Exception as e:
    st.error(f"Prediction error: {e}")

# Footer
st.markdown("""
**About**: This app uses a Random Forest model trained on the MarketFarePredictionData dataset from Mendeley, collected by the US Department of Transportation (May 15, 2025).  
Dataset link: [Mendeley](https://data.mendeley.com/datasets/m5mvxdx2wp/2)
""")
