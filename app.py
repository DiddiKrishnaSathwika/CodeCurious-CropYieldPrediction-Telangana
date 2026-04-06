import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Crop Yield Prediction System - Telangana", layout="wide")

DB_FILE = "crop_yield.db"
MODEL_FILE = "model/crop_model.pkl"
LE_CROP_FILE = "model/le_crop.pkl"
LE_DISTRICT_FILE = "model/le_district.pkl"
LE_SOIL_FILE = "model/le_soil.pkl"
LE_WATER_FILE = "model/le_water.pkl"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_type TEXT,
            district TEXT,
            rainfall REAL,
            temperature REAL,
            humidity REAL,
            soil_type TEXT,
            soil_ph REAL,
            water_availability TEXT,
            area REAL,
            fertilizer REAL,
            pesticide REAL,
            production REAL,
            predicted_yield REAL,
            prediction_date TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(crop_type, district, rainfall, temperature, humidity, soil_type, soil_ph,
                    water_availability, area, fertilizer, pesticide, production, predicted_yield):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            crop_type, district, rainfall, temperature, humidity,
            soil_type, soil_ph, water_availability, area,
            fertilizer, pesticide, production, predicted_yield, prediction_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        crop_type, district, rainfall, temperature, humidity,
        soil_type, soil_ph, water_availability, area,
        fertilizer, pesticide, production, predicted_yield,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

def load_assets():
    needed = [MODEL_FILE, LE_CROP_FILE, LE_DISTRICT_FILE, LE_SOIL_FILE, LE_WATER_FILE]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        st.error("Model files are missing. Please run train_model.py first.")
        st.stop()

    model = joblib.load(MODEL_FILE)
    le_crop = joblib.load(LE_CROP_FILE)
    le_district = joblib.load(LE_DISTRICT_FILE)
    le_soil = joblib.load(LE_SOIL_FILE)
    le_water = joblib.load(LE_WATER_FILE)

    return model, le_crop, le_district, le_soil, le_water

def get_options(encoder):
    return list(encoder.classes_)

district_defaults = {
    "Adilabad": {"rainfall": 920, "temperature": 28, "humidity": 70, "soil_type": "Black", "soil_ph": 6.7},
    "Bhadradri Kothagudem": {"rainfall": 1100, "temperature": 28, "humidity": 72, "soil_type": "Clay", "soil_ph": 6.5},
    "Hanumakonda": {"rainfall": 860, "temperature": 30, "humidity": 60, "soil_type": "Black", "soil_ph": 6.7},
    "Hyderabad": {"rainfall": 700, "temperature": 32, "humidity": 55, "soil_type": "Sandy", "soil_ph": 6.3},
    "Jagtial": {"rainfall": 850, "temperature": 30, "humidity": 62, "soil_type": "Black", "soil_ph": 6.7},
    "Jangaon": {"rainfall": 800, "temperature": 30, "humidity": 60, "soil_type": "Loamy", "soil_ph": 6.4},
    "Jayashankar Bhupalpally": {"rainfall": 980, "temperature": 29, "humidity": 68, "soil_type": "Clay", "soil_ph": 6.5},
    "Jogulamba Gadwal": {"rainfall": 720, "temperature": 32, "humidity": 55, "soil_type": "Loamy", "soil_ph": 6.4},
    "Kamareddy": {"rainfall": 840, "temperature": 30, "humidity": 61, "soil_type": "Loamy", "soil_ph": 6.4},
    "Karimnagar": {"rainfall": 830, "temperature": 30, "humidity": 60, "soil_type": "Black", "soil_ph": 6.7},
    "Khammam": {"rainfall": 1000, "temperature": 30, "humidity": 68, "soil_type": "Clay", "soil_ph": 6.5},
    "Kumuram Bheem Asifabad": {"rainfall": 950, "temperature": 28, "humidity": 70, "soil_type": "Black", "soil_ph": 6.7},
    "Mahabubabad": {"rainfall": 900, "temperature": 29, "humidity": 66, "soil_type": "Clay", "soil_ph": 6.5},
    "Mahabubnagar": {"rainfall": 730, "temperature": 32, "humidity": 55, "soil_type": "Loamy", "soil_ph": 6.4},
    "Mancherial": {"rainfall": 890, "temperature": 30, "humidity": 63, "soil_type": "Black", "soil_ph": 6.7},
    "Medak": {"rainfall": 780, "temperature": 30, "humidity": 58, "soil_type": "Black", "soil_ph": 6.7},
    "Medchal-Malkajgiri": {"rainfall": 710, "temperature": 31, "humidity": 56, "soil_type": "Sandy", "soil_ph": 6.3},
    "Mulugu": {"rainfall": 1150, "temperature": 28, "humidity": 74, "soil_type": "Clay", "soil_ph": 6.5},
    "Nagarkurnool": {"rainfall": 760, "temperature": 31, "humidity": 57, "soil_type": "Loamy", "soil_ph": 6.4},
    "Nalgonda": {"rainfall": 770, "temperature": 32, "humidity": 56, "soil_type": "Clay", "soil_ph": 6.5},
    "Narayanpet": {"rainfall": 700, "temperature": 32, "humidity": 54, "soil_type": "Sandy", "soil_ph": 6.3},
    "Nirmal": {"rainfall": 910, "temperature": 28, "humidity": 69, "soil_type": "Black", "soil_ph": 6.7},
    "Nizamabad": {"rainfall": 880, "temperature": 30, "humidity": 62, "soil_type": "Loamy", "soil_ph": 6.4},
    "Peddapalli": {"rainfall": 820, "temperature": 30, "humidity": 60, "soil_type": "Black", "soil_ph": 6.7},
    "Rajanna Sircilla": {"rainfall": 810, "temperature": 30, "humidity": 60, "soil_type": "Black", "soil_ph": 6.7},
    "Rangareddy": {"rainfall": 720, "temperature": 30, "humidity": 58, "soil_type": "Sandy", "soil_ph": 6.3},
    "Sangareddy": {"rainfall": 760, "temperature": 30, "humidity": 58, "soil_type": "Black", "soil_ph": 6.7},
    "Siddipet": {"rainfall": 790, "temperature": 30, "humidity": 59, "soil_type": "Loamy", "soil_ph": 6.4},
    "Suryapet": {"rainfall": 750, "temperature": 31, "humidity": 57, "soil_type": "Clay", "soil_ph": 6.5},
    "Vikarabad": {"rainfall": 800, "temperature": 29, "humidity": 61, "soil_type": "Loamy", "soil_ph": 6.4},
    "Wanaparthy": {"rainfall": 730, "temperature": 31, "humidity": 56, "soil_type": "Loamy", "soil_ph": 6.4},
    "Warangal": {"rainfall": 870, "temperature": 30, "humidity": 61, "soil_type": "Black", "soil_ph": 6.7},
    "Yadadri Bhuvanagiri": {"rainfall": 760, "temperature": 31, "humidity": 58, "soil_type": "Clay", "soil_ph": 6.5}
}

init_db()
model, le_crop, le_district, le_soil, le_water = load_assets()

st.title("Crop Yield Prediction System - Telangana")
st.markdown("District-wise prototype crop yield prediction for Telangana with all available crops in Telangana data.")

tabs = st.tabs(["Prediction", "Prediction History", "Trend Analysis", "Officer Dashboard"])

with tabs[0]:
    st.subheader("Yield Prediction")

    crop_options = get_options(le_crop)
    district_options = get_options(le_district)
    soil_options = get_options(le_soil)
    water_options = get_options(le_water)

    selected_crop = st.selectbox("Select Crop", crop_options)
    selected_district = st.selectbox("Select Telangana Location", district_options)

    defaults = district_defaults.get(selected_district, {
        "rainfall": 800,
        "temperature": 30,
        "humidity": 60,
        "soil_type": "Loamy",
        "soil_ph": 6.5
    })

    col1, col2 = st.columns(2)

    with col1:
        rainfall = st.number_input("Rainfall", min_value=0.0, value=float(defaults["rainfall"]), step=10.0)
        temperature = st.number_input("Temperature", min_value=0.0, value=float(defaults["temperature"]), step=1.0)
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=float(defaults["humidity"]), step=1.0)
        soil_type = st.selectbox(
            "Soil Type",
            soil_options,
            index=soil_options.index(defaults["soil_type"]) if defaults["soil_type"] in soil_options else 0
        )

    with col2:
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=float(defaults["soil_ph"]), step=0.1)
        water_availability = st.selectbox("Water Availability", water_options)
        area = st.number_input("Area", min_value=0.1, value=2.0, step=0.1)
        fertilizer = st.number_input("Fertilizer", min_value=0.0, value=100.0, step=10.0)
        pesticide = st.number_input("Pesticide", min_value=0.0, value=50.0, step=5.0)
        acres = st.number_input("Land Size (Acres)", min_value=0.1, value=1.0, step=0.1)


    if st.button("Predict Yield"):
        try:
            input_df = pd.DataFrame([{
                "crop_type": le_crop.transform([selected_crop])[0],
                "district": le_district.transform([selected_district])[0],
                "rainfall": rainfall,
                "temperature": temperature,
                "humidity": humidity,
                "soil_type": le_soil.transform([soil_type])[0],
                "soil_ph": soil_ph,
                "water_availability": le_water.transform([water_availability])[0],
                "area": area,
                "fertilizer": fertilizer,
                "pesticide": pesticide
            }])

            prediction = model.predict(input_df)[0]

            total_yield = prediction * acres   # 🔥 MAIN LOGIC

            st.success(f"Predicted Yield: {total_yield:.2f}")

            save_prediction(
                selected_crop, selected_district, rainfall, temperature, humidity,
                soil_type, soil_ph, water_availability, area,
                fertilizer, pesticide, 0, float(prediction)
            )
            st.info("Prediction saved successfully.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tabs[1]:
    st.subheader("Prediction History")
    history_df = load_history()

    if history_df.empty:
        st.warning("No prediction history found.")
    else:
        st.dataframe(history_df, use_container_width=True)

with tabs[2]:
    st.subheader("Trend Analysis")

    history_df = load_history()

    # remove Telangana only if present
    if "Telangana" in history_df["district"].values:
        history_df = history_df[history_df["district"] != "Telangana"]

    if history_df.empty:
        st.warning("No data available for trend analysis.")
    else:
        # ✅ Convert to numeric (IMPORTANT FIX)
        history_df["predicted_yield"] = pd.to_numeric(history_df["predicted_yield"], errors="coerce")

        # 🌾 Crop Trend
        crop_avg = history_df.groupby("crop_type", as_index=False)["predicted_yield"].mean()

        fig1 = px.bar(
            crop_avg,
            x="crop_type",
            y="predicted_yield",
            title="🌾 Average Predicted Yield by Crop",
            text_auto=".2f",
            color="predicted_yield"
        )

        st.plotly_chart(fig1, use_container_width=True)

        # 📍 District Trend
        district_avg = history_df.groupby("district", as_index=False)["predicted_yield"].mean()

        fig2 = px.bar(
            district_avg,
            x="district",
            y="predicted_yield",
            title="📍 Average Predicted Yield by Telangana Location",
            text_auto=".2f",
            color="predicted_yield"
        )

        fig2.update_layout(xaxis_tickangle=45)

        st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader("Officer Dashboard")
    history_df = load_history()

    if history_df.empty:
        st.warning("No data available for dashboard.")
    else:
        total_predictions = len(history_df)
        avg_yield = history_df["predicted_yield"].mean()
        best_crop = history_df.groupby("crop_type")["predicted_yield"].mean().idxmax()
        best_location = history_df.groupby("district")["predicted_yield"].mean().idxmax()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", total_predictions)
        c2.metric("Average Yield", f"{avg_yield:.2f}")
        c3.metric("Best Crop", best_crop)
        c4.metric("Best Telangana Location", best_location)

        st.markdown("### Crop-wise Average Yield")
        st.dataframe(history_df.groupby("crop_type", as_index=False)["predicted_yield"].mean(), use_container_width=True)

        st.markdown("### Location-wise Average Yield")
        st.dataframe(history_df.groupby("district", as_index=False)["predicted_yield"].mean(), use_container_width=True)