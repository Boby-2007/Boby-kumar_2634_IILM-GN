# Inadequate Waste Management - Waste Prediction + Route Optimization
# Matches project file content exactly
# Run: pip install pandas numpy scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# ---------------------------------------------------------
# Step 1: Create Dummy Dataset (Based on your PDF features)
# ---------------------------------------------------------
np.random.seed(42)

areas = [f"Area_{i}" for i in range(1, 21)]
locality_type = ["Residential", "Commercial", "Industrial"]
weather = ["Sunny", "Rainy", "Cloudy"]

data = {
    "Area": np.random.choice(areas, 200),
    "Population_Density": np.random.randint(100, 5000, 200),
    "Locality_Type": np.random.choice(locality_type, 200),
    "Previous_Waste_kg": np.random.uniform(50, 500, 200),
    "Weather": np.random.choice(weather, 200),
    "Time_of_Collection": np.random.randint(1, 24, 200),
    "Season_Festival": np.random.randint(0, 2, 200),  # festival or not
    "Waste_Generated_kg": np.random.uniform(50, 600, 200)  # target variable
}

df = pd.DataFrame(data)
print("Sample Data:")
print(df.head())

# Encode categorical columns
df["Locality_Type"] = df["Locality_Type"].astype("category").cat.codes
df["Weather"] = df["Weather"].astype("category").cat.codes
df["Area_ID"] = df["Area"].astype("category").cat.codes

# Features + Target
X = df[[
    "Area_ID", "Population_Density", "Locality_Type",
    "Previous_Waste_kg", "Weather", "Time_of_Collection", "Season_Festival"
]]
y = df["Waste_Generated_kg"]

# ---------------------------------------------------------
# Step 2: Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# ---------------------------------------------------------
# Step 3: Model Training (Random Forest Regression)
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

# ---------------------------------------------------------
# Step 5: Predict Waste for Next-Day (example)
# ---------------------------------------------------------
new_area = X.sample(5)  # random 5 areas for demo
predicted_waste = model.predict(new_area)

print("\nPredicted Waste for next collection:")
print(predicted_waste)

# ---------------------------------------------------------
# Step 6: Simple Route Optimization Algorithm
# ---------------------------------------------------------
# Each area is assigned X/Y random coordinates for demo route planning
coords = {area: (np.random.randint(1, 50), np.random.randint(1, 50)) for area in areas}

def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

start = (0, 0)
route = []

remaining = list(coords.keys())
current = start

while remaining:
    nearest = min(remaining, key=lambda area: distance(current, coords[area]))
    route.append(nearest)
    current = coords[nearest]
    remaining.remove(nearest)

print("\nOptimized Truck Route (Nearest-First):")
print(route)