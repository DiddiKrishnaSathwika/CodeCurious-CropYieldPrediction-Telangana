import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

RAW_FILE = "data/raw/india_crop_production.csv"
PROCESSED_FILE = "data/processed/telangana_crop_data.csv"

print("Step 1: Loading raw dataset...")

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}\n"
        f"Make sure the file is saved exactly as india_crop_production.csv inside data/raw/"
    )

df = pd.read_csv(RAW_FILE)
df.columns = [col.strip() for col in df.columns]

print("Available columns:", list(df.columns))

required_columns = ["Crop", "State", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

print("Step 2: Filtering Telangana data...")

telangana_df = df[df["State"].astype(str).str.strip().str.lower() == "telangana"].copy()

if telangana_df.empty:
    raise ValueError("No Telangana rows found in the dataset.")

# keep all crops
telangana_df["Crop"] = telangana_df["Crop"].astype(str).str.strip().str.title()
telangana_df = telangana_df[telangana_df["Crop"] != ""].copy()

if telangana_df.empty:
    raise ValueError("No crop records found in Telangana data.")

telangana_df = telangana_df.rename(columns={
    "Crop": "crop_type",
    "Annual_Rainfall": "rainfall",
    "Area": "area",
    "Production": "production",
    "Yield": "yield",
    "Fertilizer": "fertilizer",
    "Pesticide": "pesticide"
})

telangana_df = telangana_df[
    ["crop_type", "rainfall", "area", "production", "fertilizer", "pesticide", "yield"]
].copy()

telangana_df = telangana_df.dropna()

for col in ["rainfall", "area", "production", "fertilizer", "pesticide", "yield"]:
    telangana_df[col] = pd.to_numeric(telangana_df[col], errors="coerce")

telangana_df = telangana_df.dropna()

telangana_df = telangana_df[
    (telangana_df["area"] > 0) &
    (telangana_df["production"] > 0) &
    (telangana_df["yield"] > 0)
].copy()

telangana_districts = [
    "Adilabad",
    "Bhadradri Kothagudem",
    "Hanumakonda",
    "Hyderabad",
    "Jagtial",
    "Jangaon",
    "Jayashankar Bhupalpally",
    "Jogulamba Gadwal",
    "Kamareddy",
    "Karimnagar",
    "Khammam",
    "Kumuram Bheem Asifabad",
    "Mahabubabad",
    "Mahabubnagar",
    "Mancherial",
    "Medak",
    "Medchal-Malkajgiri",
    "Mulugu",
    "Nagarkurnool",
    "Nalgonda",
    "Narayanpet",
    "Nirmal",
    "Nizamabad",
    "Peddapalli",
    "Rajanna Sircilla",
    "Rangareddy",
    "Sangareddy",
    "Siddipet",
    "Suryapet",
    "Vikarabad",
    "Wanaparthy",
    "Warangal",
    "Yadadri Bhuvanagiri"
]

telangana_df["district"] = [
    telangana_districts[i % len(telangana_districts)]
    for i in range(len(telangana_df))
]

district_rainfall_map = {
    "Adilabad": 920, "Bhadradri Kothagudem": 1100, "Hanumakonda": 860, "Hyderabad": 700,
    "Jagtial": 850, "Jangaon": 800, "Jayashankar Bhupalpally": 980, "Jogulamba Gadwal": 720,
    "Kamareddy": 840, "Karimnagar": 830, "Khammam": 1000, "Kumuram Bheem Asifabad": 950,
    "Mahabubabad": 900, "Mahabubnagar": 730, "Mancherial": 890, "Medak": 780,
    "Medchal-Malkajgiri": 710, "Mulugu": 1150, "Nagarkurnool": 760, "Nalgonda": 770,
    "Narayanpet": 700, "Nirmal": 910, "Nizamabad": 880, "Peddapalli": 820,
    "Rajanna Sircilla": 810, "Rangareddy": 720, "Sangareddy": 760, "Siddipet": 790,
    "Suryapet": 750, "Vikarabad": 800, "Wanaparthy": 730, "Warangal": 870,
    "Yadadri Bhuvanagiri": 760
}

district_soil_map = {
    "Adilabad": "Black", "Bhadradri Kothagudem": "Clay", "Hanumakonda": "Black", "Hyderabad": "Sandy",
    "Jagtial": "Black", "Jangaon": "Loamy", "Jayashankar Bhupalpally": "Clay", "Jogulamba Gadwal": "Loamy",
    "Kamareddy": "Loamy", "Karimnagar": "Black", "Khammam": "Clay", "Kumuram Bheem Asifabad": "Black",
    "Mahabubabad": "Clay", "Mahabubnagar": "Loamy", "Mancherial": "Black", "Medak": "Black",
    "Medchal-Malkajgiri": "Sandy", "Mulugu": "Clay", "Nagarkurnool": "Loamy", "Narayanpet": "Sandy",
    "Nalgonda": "Clay", "Nirmal": "Black", "Nizamabad": "Loamy", "Peddapalli": "Black",
    "Rajanna Sircilla": "Black", "Rangareddy": "Sandy", "Sangareddy": "Black", "Siddipet": "Loamy",
    "Suryapet": "Clay", "Vikarabad": "Loamy", "Wanaparthy": "Loamy", "Warangal": "Black",
    "Yadadri Bhuvanagiri": "Clay"
}

district_temperature_map = {district: 30 for district in telangana_districts}
district_humidity_map = {district: 60 for district in telangana_districts}

for d in ["Adilabad", "Kumuram Bheem Asifabad", "Nirmal", "Mulugu", "Bhadradri Kothagudem"]:
    district_temperature_map[d] = 28
    district_humidity_map[d] = 70

for d in ["Hyderabad", "Mahabubnagar", "Narayanpet", "Jogulamba Gadwal", "Nalgonda"]:
    district_temperature_map[d] = 32
    district_humidity_map[d] = 55

def assign_soil_ph(soil):
    mapping = {"Black": 6.7, "Clay": 6.5, "Loamy": 6.4, "Sandy": 6.3}
    return mapping.get(soil, 6.5)

def assign_water(rainfall):
    if rainfall >= 950:
        return "High"
    elif rainfall >= 800:
        return "Medium"
    return "Low"

telangana_df["soil_type"] = telangana_df["district"].map(district_soil_map)
telangana_df["rainfall"] = telangana_df["district"].map(district_rainfall_map)
telangana_df["temperature"] = telangana_df["district"].map(district_temperature_map)
telangana_df["humidity"] = telangana_df["district"].map(district_humidity_map)
telangana_df["soil_ph"] = telangana_df["soil_type"].apply(assign_soil_ph)
telangana_df["water_availability"] = telangana_df["rainfall"].apply(assign_water)

telangana_df = telangana_df[
    [
        "crop_type",
        "district",
        "rainfall",
        "temperature",
        "humidity",
        "soil_type",
        "soil_ph",
        "water_availability",
        "area",
        "fertilizer",
        "pesticide",
        "production",
        "yield"
    ]
].copy()

os.makedirs("data/processed", exist_ok=True)
telangana_df.to_csv(PROCESSED_FILE, index=False)

print(f"Step 3: Telangana processed data saved to {PROCESSED_FILE}")
print(f"Total rows after filtering: {len(telangana_df)}")
print("Available crops in Telangana data:", sorted(telangana_df["crop_type"].unique().tolist()))

le_crop = LabelEncoder()
le_district = LabelEncoder()
le_soil = LabelEncoder()
le_water = LabelEncoder()

model_df = telangana_df.copy()
model_df["crop_type"] = le_crop.fit_transform(model_df["crop_type"])
model_df["district"] = le_district.fit_transform(model_df["district"])
model_df["soil_type"] = le_soil.fit_transform(model_df["soil_type"])
model_df["water_availability"] = le_water.fit_transform(model_df["water_availability"])

X = model_df.drop("yield", axis=1)
X = model_df.drop(columns=["yield", "production"])
y = model_df["yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Step 4: Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/crop_model.pkl")
joblib.dump(le_crop, "model/le_crop.pkl")
joblib.dump(le_district, "model/le_district.pkl")
joblib.dump(le_soil, "model/le_soil.pkl")
joblib.dump(le_water, "model/le_water.pkl")

print("Step 5: Model and encoders saved successfully!")
print("Model trained and saved successfully!")