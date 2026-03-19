import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
# 1. Load Data
df = pd.read_csv("C:/Users/manis/OneDrive/Desktop/taxi project/data/taxipricing.csv")

# Remove rows with missing target
df = df.dropna(subset=["Trip_Price"])

# 2. Prepare Data
target_column = "Trip_Price"
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical → same as training
X = pd.get_dummies(X, drop_first=True)

# 3. Load Model + Columns

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

# Align columns
X = X.reindex(columns=columns, fill_value=0)

# 4. Prediction
y_pred = model.predict(X)

# 5. Evaluation Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("📊 Model Evaluation Results")
print("----------------------------")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")
