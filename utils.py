import pandas as pd
import numpy as np
import joblib
import json
import random
import streamlit as st  # For caching

# Load model stuff - cached for performance
@st.cache_resource
def load_model_stuff():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()

# Feature engineering - straight from training, fixed lighting to 'night'
def engineer_features(df):
    df_eng = df.copy()
    df_eng['high_speed'] = (df_eng['speed_limit'] >= 60).astype(int)
    df_eng['night_lighting'] = (df_eng['lighting'] == 'night').astype(int)  # Matches synthetic: 'night'
    df_eng['bad_weather'] = (df_eng['weather'] != 'clear').astype(int)
    df_eng['high_accidents'] = (df_eng['num_reported_accidents'] > 2).astype(int)
    df_eng['curvature_high_speed'] = df_eng['curvature'] * df_eng['high_speed']
    df_eng['accidents_binned'] = pd.cut(df_eng['num_reported_accidents'], bins=[-1, 0, 2, np.inf], labels=[0, 1, 2])
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(int)
    return df_eng

# Preprocess for prediction - aligns OHE and scales
categorical_features_for_ohe = ['road_type', 'lighting', 'weather', 'time_of_day', 'accidents_binned']

def preprocess_input(df, feature_names, scaler):
    df_eng = engineer_features(df)
    df_encoded = pd.get_dummies(df_eng, columns=categorical_features_for_ohe, drop_first=True)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

# Predict - clips to 0-1
def predict_risk(df, model, feature_names, scaler):
    scaled = preprocess_input(df, feature_names, scaler)
    pred = model.predict(scaled)
    return np.clip(pred, 0, 1)[0]

# Generate realistic road - based on your synthetic generator, but with lane logic
# Rural: 1-2 lanes (no 4-lane backroads), urban: 2-4, highway: 3-6
# Poisson for accidents (lam=1.5), matching choices
def generate_road(seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    road_type = np.random.choice(["highway", "urban", "rural"])
    
    # Realistic lanes by type
    if road_type == "rural":
        num_lanes = np.random.randint(1, 3)  # 1-2 lanes
    elif road_type == "urban":
        num_lanes = np.random.randint(2, 5)  # 2-4 lanes
    else:  # highway
        num_lanes = np.random.randint(3, 7)  # 3-6 lanes
    
    road = {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "curvature": np.round(np.random.uniform(0.0, 1.0), 2),
        "speed_limit": np.random.choice([25, 35, 45, 60, 70]),
        "lighting": np.random.choice(["daylight", "night", "dim"]),
        "weather": np.random.choice(["clear", "rainy", "foggy"]),
        "road_signs_present": np.random.choice([True, False]),
        "public_road": np.random.choice([True, False]),
        "time_of_day": np.random.choice(["morning", "evening", "afternoon"]),
        "holiday": np.random.choice([True, False]),
        "school_season": np.random.choice([True, False]),
        "num_reported_accidents": np.random.poisson(lam=1.5)  # Like your generator
    }
    return road