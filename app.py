import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io

# Set page configuration
st.set_page_config(page_title="Airline Fare Prediction", layout="wide")

# Title and description
st.title("Airline Fare Prediction")
st.markdown("""
This application predicts airline ticket prices based on various features using a Random Forest model.
Enter the flight details below to get a predicted fare, or explore the visualizations to understand the data.
""")

# Define expected columns (fallback if model is not a pipeline)
expected_columns = [
    'num__MktCoupons', 'num__OriginCityMarketID', 'num__DestCityMarketID', 'num__OriginAirportID',
    'num__DestAirportID', 'num__Carrier', 'num__NonStopMiles', 'num__RoundTrip', 'num__ODPairID',
    'num__Pax', 'num__CarrierPax', 'num__Market_share', 'num__Market_HHI', 'num__LCC_Comp',
    'num__Multi_Airport', 'num__Circuity', 'num__Slot', 'num__Non_Des', 'num__MktMilesFlown',
    'num__OriginCityMarketID_freq', 'num__DestCityMarketID_freq', 'num__OriginAirportID_freq',
    'num__DestAirportID_freq', 'num__Carrier_freq', 'num__ODPairID_freq', 'num__DaysToDeparture',
    'num__FarePerMile', 'num__CarrierShare', 'num__IsWeekendDeparture', 'num__IsLCC'
]

# Load model from Google Drive
@st.cache_resource
def load_model_from_drive():
    file_id = "1_5qDOp1fF3IMrIsNxkDRXnYtfVej5hTU"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the trained model
model = load_model_from_drive()

if model:
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        try:
            expected_columns = model.named_steps['preprocessor'].get_feature_names_out().tolist()
            st.write("Using pipeline feature names.")
        except Exception as e:
            st.warning(f"Failed to get feature names from pipeline: {e}. Using fallback columns.")
    else:
        st.warning("Model is not a pipeline; using predefined feature list.")
    st.success("✅ Model loaded successfully!")
    st.write("Model type:", str(type(model)))
    st.write("Expected columns:", expected_columns)
else:
    st.stop()

# Carrier mapping
carrier_mapping = {
    0: 'Other', 1: 'American', 2: 'Alaska', 3: 'JetBlue', 4: 'Delta', 5: 'Frontier',
    6: 'United', 7: 'Hawaiian', 8: 'Southwest', 9: 'Spirit', 10: 'Allegiant',
    11: 'Sun Country', 12: 'Virgin America', 13: 'SkyWest', 14: 'ExpressJet',
    15: 'Envoy Air', 16: 'PSA Airlines', 17: 'Mesa Airlines', 18: 'Endeavor Air',
    19: 'Horizon Air', 20: 'US Airways', 21: 'Republic Airways', 22: 'Compass Airlines',
    23: 'GoJet Airlines', 24: 'Trans States Airlines'
}

# Placeholder values
placeholder_values = {
    'OriginCityMarketID': 31703, 'DestCityMarketID': 31703, 'OriginAirportID': 11292,
    'DestAirportID': 11292, 'RoundTrip': 1, 'ODPairID': 123456, 'Multi_Airport': 0,
    'Slot': 0, 'Non_Des': 0, 'OriginCityMarketID_freq': 1000, 'DestCityMarketID_freq': 1000,
    'OriginAirportID_freq': 1000, 'DestAirportID_freq': 1000, 'Carrier_freq': 1000,
    'ODPairID_freq': 1000
}
fare_per_mile_median = 0.25

# Sidebar input
st.sidebar.header("Flight Details")
carrier = st.sidebar.selectbox("Carrier", list(carrier_mapping.values()))
days_to_departure = st.sidebar.slider("Days to Departure", 1, 90, 30)
mkt_miles_flown = st.sidebar.number_input("Market Miles Flown", min_value=0, max_value=10000, value=1000)
non_stop_miles = st.sidebar.number_input("Non-Stop Miles", min_value=0, max_value=10000, value=1000)
carrier_pax = st.sidebar.number_input("Carrier Passengers", min_value=0, max_value=1000000, value=10000)
pax = st.sidebar.number_input("Total Passengers", min_value=0, max_value=1000000, value=50000)
lcc_comp = st.sidebar.number_input("Low-Cost Carrier Competition", min_value=0, max_value=100, value=10)
market_hhi = st.sidebar.number_input("Market HHI", min_value=0, max_value=10000, value=2500)
market_share = st.sidebar.number_input("Market Share", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
mkt_coupons = st.sidebar.number_input("Market Coupons", min_value=0, max_value=100, value=1)
departure_month = st.sidebar.slider("Departure Month", 1, 12, 6)
departure_day_of_week = st.sidebar.slider("Departure Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

# Feature engineering
non_stop = 1 if mkt_miles_flown == non_stop_miles else 0
is_weekend_departure = 1 if departure_day_of_week in [5, 6] else 0
is_lcc = 1 if lcc_comp > 0 else 0
carrier_share = carrier_pax / pax if pax > 0 else 0
circuity = mkt_miles_flown / non_stop_miles if non_stop_miles > 0 else 1
fare_per_mile = fare_per_mile_median

# Prepare input data
input_data = pd.DataFrame({
    'num__MktCoupons': [mkt_coupons],
    'num__OriginCityMarketID': [placeholder_values['OriginCityMarketID']],
    'num__DestCityMarketID': [placeholder_values['DestCityMarketID']],
    'num__OriginAirportID': [placeholder_values['OriginAirportID']],
    'num__DestAirportID': [placeholder_values['DestAirportID']],
    'num__Carrier': [list(carrier_mapping.values()).index(carrier)],
    'num__NonStopMiles': [non_stop_miles],
    'num__RoundTrip': [placeholder_values['RoundTrip']],
    'num__ODPairID': [placeholder_values['ODPairID']],
    'num__Pax': [pax],
    'num__CarrierPax': [carrier_pax],
    'num__Market_share': [market_share],
    'num__Market_HHI': [market_hhi],
    'num__LCC_Comp': [lcc_comp],
    'num__Multi_Airport': [placeholder_values['Multi_Airport']],
    'num__Circuity': [circuity],
    'num__Slot': [placeholder_values['Slot']],
    'num__Non_Des': [non_stop],
    'num__MktMilesFlown': [mkt_miles_flown],
    'num__OriginCityMarketID_freq': [placeholder_values['OriginCityMarketID_freq']],
    'num__DestCityMarketID_freq': [placeholder_values['DestCityMarketID_freq']],
    'num__OriginAirportID_freq': [placeholder_values['OriginAirportID_freq']],
    'num__DestAirportID_freq': [placeholder_values['DestAirportID_freq']],
    'num__Carrier_freq': [placeholder_values['Carrier_freq']],
    'num__ODPairID_freq': [placeholder_values['ODPairID_freq']],
    'num__DaysToDeparture': [days_to_departure],
    'num__FarePerMile': [fare_per_mile],
    'num__CarrierShare': [carrier_share],
    'num__IsWeekendDeparture': [is_weekend_departure],
    'num__IsLCC': [is_lcc]
})
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

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
