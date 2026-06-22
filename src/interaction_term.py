import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from preprocessing import load_and_encode, train_test_split_manual, standardize
from linear_reg_multi import add_intercept, gradient_descent, r_squared
from ply_reg import rmse


if __name__ == "__main__":
    df_encoded = load_and_encode()

    # --- Add the interaction term: smoker_yes * bmi ---
    df_interact = df_encoded.copy()
    df_interact["smoker_bmi"] = df_interact["smoker_yes"] * df_interact["bmi"]

    feature_cols = [c for c in df_interact.columns if c != "charges"]
    X = df_interact[feature_cols].values.astype(float)
    y = df_interact["charges"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2, seed=42)
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)
    X_train_f = add_intercept(X_train_s)
    X_test_f = add_intercept(X_test_s)

    beta, _ = gradient_descent(X_train_f, y_train, learning_rate=0.1, n_iterations=1500, verbose=False)

    print("=== WITH smoker x bmi interaction term ===")
    print(f"Train R^2: {r_squared(X_train_f, y_train, beta):.4f}")
    print(f"Test  R^2: {r_squared(X_test_f, y_test, beta):.4f}")
    print(f"Test  RMSE: {rmse(X_test_f, y_test, beta):.2f}")

    print("\n=== All coefficients ===")
    for name, val in zip(["intercept"] + feature_cols, beta):
        print(f"  {name:<20}{val:>12.4f}")

    print("\n=== Comparison table ===")
    print(f"{'Model':<30}{'Test R2':>10}{'Test RMSE':>12}")
    print(f"{'Baseline (linear)':<30}{0.7837:>10.4f}{5805.15:>12.2f}")
    print(f"{'+ age^2, bmi^2':<30}{0.7848:>10.4f}{5790.35:>12.2f}")
    print(f"{'+ smoker x bmi':<30}{r_squared(X_test_f, y_test, beta):>10.4f}{rmse(X_test_f, y_test, beta):>12.2f}")