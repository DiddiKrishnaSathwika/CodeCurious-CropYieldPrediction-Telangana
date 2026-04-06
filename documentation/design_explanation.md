# Design Explanation

## System Architecture

The system consists of three main parts:

### 1. Frontend (User Interface)
- Built using Streamlit
- Allows users to enter input data
- Displays prediction results and charts

### 2. Backend (Processing Layer)
- Handles data preprocessing
- Loads trained machine learning model
- Performs prediction

### 3. Database
- SQLite database is used
- Stores prediction history

## Workflow

User Input → Data Preprocessing → Model Prediction → Result Display → Store in Database

## Machine Learning Model

We use Random Forest Regression because it:
- handles multiple input features
- works well with real-world data
- reduces overfitting
- gives stable predictions