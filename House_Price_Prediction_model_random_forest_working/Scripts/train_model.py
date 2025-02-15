# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Cleaned Dataset
df = pd.read_csv("../data/cleaned_house_price_data.csv")

# Step 2: Remove Outliers (Keep only 1st to 99th percentile)
lower_bound = df['price'].quantile(0.01)
upper_bound = df['price'].quantile(0.99)
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Step 3: Apply Log Transformations to `sqft_living` and `sqft_lot`
df['sqft_living_log'] = np.log1p(df['sqft_living'])
df['sqft_lot_log'] = np.log1p(df['sqft_lot'])
df.drop(columns=['sqft_living', 'sqft_lot'], inplace=True)  # Drop original columns

# Step 4: Define Features (X) and Target (y)
X = df.drop(columns=['price'])  # Features (all columns except target)
y = df['price']  # Target variable

# Step 5: Split the Dataset into Training & Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the Features (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the Random Forest Model with Tuned Hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=150,     # Number of trees
    max_depth=20,         # Tree depth
    min_samples_split=5,  # Minimum samples to split a node
    min_samples_leaf=2,   # Minimum samples in a leaf
    max_features='sqrt',  # Number of features considered per split
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)  # Train the model

# Step 8: Evaluate Model Performance
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMSE: {mse:.2f}\nR² Score: {r2:.4f}")

# Step 9: Save the Trained Model & Scaler
joblib.dump(rf_model, "../models/random_forest_model.pkl")  # Save trained model
joblib.dump(scaler, "../models/scaler.pkl")  # Save feature scaler

print("✅ Model training completed and saved in '../models/' directory.")
