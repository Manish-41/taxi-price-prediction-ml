import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df, target_column):
    
    # =========================
    # 1. Handle Missing Values
    # =========================
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # =========================
    # 2. Remove Duplicates
    # =========================
    df = df.drop_duplicates()

    # =========================
    # 3. Encode Categorical Data
    # =========================
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # =========================
    # 4. Feature Scaling
    # =========================
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # =========================
    # 5. Split Features & Target
    # =========================
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y

