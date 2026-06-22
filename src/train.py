"""
train.py — Final training pipeline for medical insurance charge prediction.

Builds the best-performing model from our experimentation:
  Multiple Linear Regression + smoker x bmi interaction term
  (Test R^2 = 0.8658, Test RMSE = 4573 — see README for full comparison)

Run from repo root:
    python src/train.py
"""

import sys
sys.path.insert(0, "src")
import json
import numpy as np
import joblib

from preprocessing import load_and_encode, train_test_split_manual, standardize
from linear_reg_multi import add_intercept, gradient_descent, r_squared
from ply_reg import rmse


MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"


def build_features(df_encoded):
    """Adds the smoker x bmi interaction term — our best-performing feature."""
    df = df_encoded.copy()
    df["smoker_bmi"] = df["smoker_yes"] * df["bmi"]
    return df


def mae(X, y, beta):
    return np.mean(np.abs(X @ beta - y))


def main():
    print("Loading and encoding data...")
    df_encoded = load_and_encode("medical-insurance-prediction/data/insurance.csv")
    df_final = build_features(df_encoded)

    feature_cols = [c for c in df_final.columns if c != "charges"]
    X = df_final[feature_cols].values.astype(float)
    y = df_final["charges"].values.astype(float)

    print("Splitting train/test (80/20, seed=42)...")
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2, seed=42)

    print("Standardizing features (fit on train only)...")
    X_train_scaled, X_test_scaled, mean, std = standardize(X_train, X_test)
    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    print("Training via gradient descent...")
    beta, cost_history = gradient_descent(
        X_train_final, y_train,
        learning_rate=0.1, n_iterations=1500, verbose=False
    )

    metrics = {
        "train_r2": r_squared(X_train_final, y_train, beta),
        "test_r2": r_squared(X_test_final, y_test, beta),
        "test_rmse": rmse(X_test_final, y_test, beta),
        "test_mae": mae(X_test_final, y_test, beta),
        "final_cost": cost_history[-1],
    }

    print("\n=== Final Model Performance ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Save everything needed to predict on new patients later ---
    import os
    os.makedirs("models", exist_ok=True)

    artifact = {
        "beta": beta,
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()