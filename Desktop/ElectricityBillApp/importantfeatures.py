import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = r"C:\Users\Neeraj kumar gothi\Desktop\ElectricityBillApp\Household energy bill data.csv"
df = pd.read_csv(file_path)

# Define Features & Target
X = df.drop(columns=["amount_paid"])  # Features (Modify column name if needed)
y = df["amount_paid"]  # Target variable

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get Feature Importance
feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
feature_importance = feature_importance.sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")

# Save the Plot
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")  # Save as PNG
plt.show()
