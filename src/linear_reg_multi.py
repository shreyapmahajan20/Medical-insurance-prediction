import numpy as np
import pandas as pd


def add_intercept(X):
    """
    Adds a column of 1s to X for the intercept (beta0).
    X shape (m, n) -> (m, n+1)
    """
    m = X.shape[0]
    ones = np.ones((m, 1))
    return np.hstack([ones, X])


def compute_cost(X, y, beta):
    """
    Vectorized MSE cost.
    X: (m, n+1) including intercept column
    y: (m,)
    beta: (n+1,)
    """
    m = len(y)
    y_hat = X @ beta
    error = y_hat - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost


def compute_gradients(X, y, beta):
    """
    Vectorized gradient: dJ/dbeta = (1/m) * X^T @ (y_hat - y)
    This single line replaces the separate d_beta0, d_beta1 formulas —
    it computes the gradient for EVERY beta (intercept + all features) at once.
    """
    m = len(y)
    y_hat = X @ beta
    error = y_hat - y
    gradient = (1 / m) * (X.T @ error)
    return gradient


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    """
    X: (m, n+1) including intercept column, already scaled
    y: (m,)
    """
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    cost_history = []

    for i in range(n_iterations):
        gradient = compute_gradients(X, y, beta)
        beta = beta - learning_rate * gradient

        cost = compute_cost(X, y, beta)
        cost_history.append(cost)

        if verbose and i % 100 == 0:
            print(f"Iteration {i}: cost={cost:.2f}")

    return beta, cost_history


def r_squared(X, y, beta):
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocessing import load_and_encode, train_test_split_manual, standardize

    df_encoded = load_and_encode()

    feature_cols = [c for c in df_encoded.columns if c != "charges"]
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded["charges"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2)

    X_train_scaled, X_test_scaled, mean, std = standardize(X_train, X_test)

    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    beta, history = gradient_descent(
        X_train_final, y_train,
        learning_rate=0.1,
        n_iterations=1000
    )

    print("\nFinal beta (scaled, includes intercept):")
    for name, val in zip(["intercept"] + feature_cols, beta):
        print(f"  {name}: {val:.4f}")

    train_r2 = r_squared(X_train_final, y_train, beta)
    test_r2 = r_squared(X_test_final, y_test, beta)

    print(f"\nTrain R^2: {train_r2:.4f}")
    print(f"Test R^2:  {test_r2:.4f}")