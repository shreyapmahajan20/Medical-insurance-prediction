import sys
sys.path.insert(0, "src")
import numpy as np
from preprocessing import load_and_encode, train_test_split_manual, standardize
from linear_reg_multi import add_intercept, gradient_descent, r_squared


def ols_normal_equation(X, y):
    """
    Closed-form solution: beta = (X^T X)^-1 X^T y
    X must already include the intercept column of 1s.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.inv(XtX) @ Xty
    return beta


if __name__ == "__main__":
    df_encoded = load_and_encode()
    feature_cols = [c for c in df_encoded.columns if c != "charges"]
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded["charges"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2, seed=42)
    X_train_scaled, X_test_scaled, mean, std = standardize(X_train, X_test)
    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    # --- OLS, closed form, no iteration ---
    beta_ols = ols_normal_equation(X_train_final, y_train)

    # --- Gradient descent, for comparison ---
    beta_gd, _ = gradient_descent(X_train_final, y_train, learning_rate=0.1,
                                   n_iterations=1000, verbose=False)

    print("Comparison of coefficients (OLS vs Gradient Descent):")
    print(f"{'feature':<20}{'OLS':>12}{'GD':>12}{'diff':>12}")
    for name, b_ols, b_gd in zip(["intercept"] + feature_cols, beta_ols, beta_gd):
        print(f"{name:<20}{b_ols:>12.4f}{b_gd:>12.4f}{abs(b_ols-b_gd):>12.6f}")

    print(f"\nOLS  Train R^2: {r_squared(X_train_final, y_train, beta_ols):.4f}")
    print(f"OLS  Test  R^2: {r_squared(X_test_final, y_test, beta_ols):.4f}")
    print(f"GD   Train R^2: {r_squared(X_train_final, y_train, beta_gd):.4f}")
    print(f"GD   Test  R^2: {r_squared(X_test_final, y_test, beta_gd):.4f}")