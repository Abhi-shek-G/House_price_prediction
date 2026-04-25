import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ── Load model & scaler ──────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model  = pickle.load(open(os.path.join(BASE_DIR, 'house_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'house_scaler.pkl'), 'rb'))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Fill in the details below to estimate the house price.")

# ── Input fields ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    square_footage   = st.number_input('Square Footage (sqft)', min_value=300,  max_value=10000, value=1500)
    num_bedrooms     = st.number_input('Number of Bedrooms',    min_value=1,    max_value=10,    value=3)
    num_bathrooms    = st.number_input('Number of Bathrooms',   min_value=1,    max_value=10,    value=2)
    year_built       = st.number_input('Year Built',            min_value=1900, max_value=2024,  value=2000)

with col2:
    lot_size             = st.number_input('Lot Size (acres)',      min_value=0.1, max_value=10.0, value=0.5)
    garage_size          = st.number_input('Garage Size (cars)',    min_value=0,   max_value=4,    value=1)
    neighborhood_quality = st.slider('Neighborhood Quality',        min_value=1,   max_value=10,   value=7)

# ── Build input dataframe ─────────────────────────────────────────────────────
input_features = pd.DataFrame({
    'Square_Footage':       [square_footage],
    'Num_Bedrooms':         [num_bedrooms],
    'Num_Bathrooms':        [num_bathrooms],
    'Year_Built':           [year_built],        # ← moved up
    'Lot_Size':             [lot_size],          # ← moved down
    'Garage_Size':          [garage_size],
    'Neighborhood_Quality': [neighborhood_quality],
    # Proximity_to_City removed — not in the trained model
})

# Scale using the LOADED scaler (do NOT refit)
input_scaled = scaler.transform(input_features)

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button('🔍 Predict Price'):
    prediction = model.predict(input_scaled)
    price = np.expm1(prediction[0])          # reverse log1p applied during training
    st.success(f"💰 Estimated House Price: **${price:,.2f}**")
    st.info("This estimate is based on the provided features using a trained ML model.")
