# Crop Yield Prediction System - Telangana

## Problem Statement
Farmers in Telangana often estimate crop yield using guesswork. Crop production depends on rainfall, temperature, soil quality, water availability, and location. This leads to poor planning and financial losses.

## Solution
This project predicts crop yield using crop type, district, weather, soil, and water data. It helps farmers and government officers make better decisions.

## Features
- Multi-crop support
- Telangana district-wise prediction
- Yield prediction using Random Forest Regression
- Prediction history
- Trend analysis
- Officer dashboard
- SQLite database storage

## Tech Stack
- Python
- Streamlit
- SQLite
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Project Structure
- `app.py` - Streamlit application
- `train_model.py` - Model training script
- `data/raw/` - Raw dataset
- `data/processed/` - Processed Telangana dataset
- `model/` - Saved model and encoders
- `documentation/` - Project documents
- `architecture/` - Architecture diagram
- `slides/` - Presentation
- `screenshots/` - App screenshots

## How to Run
1. Create virtual environment
2. Install requirements
3. Place Kaggle CSV inside `data/raw/` as `india_crop_production.csv`
4. Run:
   ```bash
   python train_model.py
   https://codecurious-cropyieldprediction-telangana-dzrrluhsnkvwf7bbvkrd.streamlit.app/
