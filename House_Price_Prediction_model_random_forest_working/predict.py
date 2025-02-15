# predict.py

import numpy as np
import joblib

# Step 1: Load Trained Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Step 2: Example House Features (New Input Data)
new_house = np.array([[3, 2, np.log1p(1800), np.log1p(5000), 1, 0, 0, 4, 1000, 800, 2005, 0]])

# Step 3: Scale Input Features
new_house_scaled = scaler.transform(new_house)

# Step 4: Make Prediction
predicted_price = model.predict(new_house_scaled)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
