from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler
model = joblib.load("../models/random_forest_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Define House Features Schema
from pydantic import BaseModel

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int

@app.post("/predict/")
def predict_price(features: HouseFeatures):
    # Convert input data to NumPy array
    input_data = np.array([[features.bedrooms, features.bathrooms,
                            np.log1p(features.sqft_living), np.log1p(features.sqft_lot),
                            features.floors, features.waterfront, features.view,
                            features.condition, features.sqft_above, features.sqft_basement,
                            features.yr_built, features.yr_renovated]])

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_data_scaled)
    
    return {"predicted_price": round(predicted_price[0], 2)}

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
