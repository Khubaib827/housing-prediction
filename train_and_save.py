"""Train a model on Boston Housing data and save it as models/model.joblib

Usage:
  python train_and_save.py
Optionally replace data/boston_sample.csv with your full Boston Housing CSV.
Expected columns:
CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boston_sample.csv"  # change to your full dataset path if available
MODEL_PATH = ROOT / "models" / "model.joblib"

FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
TARGET = "MEDV"

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[Error] CSV not found at {path}. Place your dataset there.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        print(f"[Error] Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    return df

def main():
    print("[1/4] Loading data...")
    df = load_data(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print("[2/4] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[3/4] Building pipeline...")
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    print("[3/4] Training...")
    pipe.fit(X_train, y_train)

    print("[4/4] Evaluating...")
    preds = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    dump(pipe, MODEL_PATH)
    print(f"[Done] Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()