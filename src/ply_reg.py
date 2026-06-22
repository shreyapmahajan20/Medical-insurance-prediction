import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from preprocessing import load_and_encode, train_test_split_manual, standardize
from linear_reg_multi import add_intercept, gradient_descent, r_squared


def add_polynomial_features(df, columns, degree=2):
    """
    Adds squared (or higher degree) versions of specified columns.
    e.g. add_polynomial_features(df, ['age', 'bmi'], degree=2)
    creates 'age^2' and 'bmi^2' columns.
    """
    df = df.copy()
    for col in columns:
        for d in range(2, degree + 1):
            df[f"{col}^{d}"] = df[col] ** d
    return df


def rmse(X, y, beta):
    y_hat = X @ beta
    return np.sqrt(np.mean((y_hat - y) ** 2))


if __name__ == "__main__":
    df_encoded = load_and_encode()

    # --- BASELINE: plain multiple linear regression (no polynomial terms) ---
    feature_cols_base = [c for c in df_encoded.columns if c != "charges"]
    X_base = df_encoded[feature_cols_base].values.astype(float)
    y = df_encoded["charges"].values.astype(float)

    X_train_b, X_test_b, y_train, y_test = train_test_split_manual(X_base, y, test_ratio=0.2, seed=42)
    X_train_b_s, X_test_b_s, _, _ = standardize(X_train_b, X_test_b)
    X_train_b_f = add_intercept(X_train_b_s)
    X_test_b_f = add_intercept(X_test_b_s)

    beta_base, _ = gradient_descent(X_train_b_f, y_train, learning_rate=0.1, n_iterations=1000, verbose=False)

    print("=== BASELINE (linear only) ===")
    print(f"Train R^2: {r_squared(X_train_b_f, y_train, beta_base):.4f}")
    print(f"Test  R^2: {r_squared(X_test_b_f, y_test, beta_base):.4f}")
    print(f"Test  RMSE: {rmse(X_test_b_f, y_test, beta_base):.2f}")

    # --- POLYNOMIAL: add age^2 and bmi^2 ---
    df_poly = add_polynomial_features(df_encoded, ["age", "bmi"], degree=2)
    feature_cols_poly = [c for c in df_poly.columns if c != "charges"]
    X_poly = df_poly[feature_cols_poly].values.astype(float)

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split_manual(X_poly, y, test_ratio=0.2, seed=42)
    X_train_p_s, X_test_p_s, _, _ = standardize(X_train_p, X_test_p)
    X_train_p_f = add_intercept(X_train_p_s)
    X_test_p_f = add_intercept(X_test_p_s)

    beta_poly, _ = gradient_descent(X_train_p_f, y_train_p, learning_rate=0.1, n_iterations=1000, verbose=False)

    print("\n=== POLYNOMIAL (added age^2, bmi^2) ===")
    print(f"Train R^2: {r_squared(X_train_p_f, y_train_p, beta_poly):.4f}")
    print(f"Test  R^2: {r_squared(X_test_p_f, y_test_p, beta_poly):.4f}")
    print(f"Test  RMSE: {rmse(X_test_p_f, y_test_p, beta_poly):.2f}")

    print("\n=== Coefficients for the new polynomial terms ===")
    for name, val in zip(["intercept"] + feature_cols_poly, beta_poly):
        if "^2" in name:
            print(f"  {name}: {val:.4f}")