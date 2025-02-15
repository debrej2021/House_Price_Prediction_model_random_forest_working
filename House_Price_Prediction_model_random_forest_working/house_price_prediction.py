# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the cleaned dataset
file_path = "cleaned_house_price_data.csv"  # Ensure this file is in the working directory
df = pd.read_csv(file_path)

# Step 2: Define Features (X) and Target Variable (y)
X = df.drop(columns=['price'])  # Features (all columns except target)
y = df['price']  # Target (house price)

# Step 3: Split Dataset into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Step 6: Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred_lr = lr_model.predict(X_test_scaled)  # Predictions from Linear Regression
y_pred_rf = rf_model.predict(X_test_scaled)  # Predictions from Random Forest

# Step 8: Evaluate Models
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Step 9: Print Model Performance
print("Model Performance Comparison:")
print(f"Linear Regression - Mean Squared Error: {mse_lr:.2f}, R² Score: {r2_lr:.4f}")
print(f"Random Forest - Mean Squared Error: {mse_rf:.2f}, R² Score: {r2_rf:.4f}")
