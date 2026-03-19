# =========================
# 1. Import Libraries
# =========================

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn - preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Models
from sklearn.ensemble import RandomForestRegressor

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Save model
import joblib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# =========================
# 2. Load Dataset
# =========================

df = pd.read_csv("C:/Users/manis/OneDrive/Desktop/taxi project/data/taxipricing.csv")

print(df.head())
print(df.info())



# =========================
# 3. Basic Data Analysis
# =========================

print("\nShape of dataset:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Fill numerical columns with mean
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(exclude=np.number).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

df = df.drop_duplicates()

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

target_column = "fare_amount"  # <-- your real column

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Correlation Heatmap
# =========================

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 5. Distribution Plots
# =========================

df.hist(figsize=(12, 10))
plt.show()

# =========================
# 6. Boxplots
# =========================

for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# =========================
# 7. Target Variable Distribution
# =========================

plt.figure(figsize=(6,4))
sns.histplot(y, kde=True)
plt.title("Target Variable Distribution")
plt.show()

# =========================
# 8. Feature vs Target
# =========================

for col in X.columns:
    plt.figure()
    sns.scatterplot(x=df[col], y=y)
    plt.title(f"{col} vs Target")
    plt.show()

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# Import our preprocessing function
from src.data_preprocessing import preprocess_data


# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("C:/Users/manis/OneDrive/Desktop/taxi project/data/taxipricing.csv")

# =========================
# 2. Preprocess Data
# =========================
X, y = preprocess_data(df, target_column="target_column")

# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Train Model
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# =========================
# 5. Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 6. Evaluation
# =========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# =========================
# 7. Save Model
# =========================
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")






from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("R2 Score:", r2)

joblib.dump(best_model, "models/model.pkl")
