import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the dataset (use relative path for compatibility across environments)
df = pd.read_csv('Household energy bill data.csv')

# Display dataset info
print(df.info())  # Check columns and data types
print(df.head())  # Preview first 5 rows

# Define features & target (Corrected column names)
X = df.drop(columns=["amount_paid"])  # Features
y = df["amount_paid"]  # Target variable (Electricity Bill)

# Identify numerical and categorical columns
num_cols = ["num_rooms", "num_people", "housearea", "ave_monthly_income", "num_children"]
cat_cols = ["is_ac", "is_tv", "is_flat", "is_urban"]  # Binary categorical (0/1)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only numerical features (not categorical)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model for use in Streamlit
with open("randomforest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Model saved as randomforest_model.pkl")
