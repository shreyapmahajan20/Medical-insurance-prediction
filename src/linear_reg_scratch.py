import numpy as np
import pandas as pd

def compute_cost(X, y, beta0, beta1):
    m = len(y)
    y_hat = beta0 + beta1 * X
    error = y_hat - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost


def compute_gradients(X, y, beta0, beta1):
    m = len(y)
    y_hat = beta0 + beta1 * X
    error = y_hat - y
    d_beta0 = (1 / m) * np.sum(error)
    d_beta1 = (1 / m) * np.sum(error * X)
    return d_beta0, d_beta1


def gradient_descent(X, y, beta0_init=0.0, beta1_init=0.0,
                      learning_rate=0.01, n_iterations=1000):
    beta0, beta1 = beta0_init, beta1_init
    cost_history = []

    for i in range(n_iterations):
        d_beta0, d_beta1 = compute_gradients(X, y, beta0, beta1)
        beta0 = beta0 - learning_rate * d_beta0
        beta1 = beta1 - learning_rate * d_beta1

        cost = compute_cost(X, y, beta0, beta1)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: cost={cost:.2f}, beta0={beta0:.4f}, beta1={beta1:.4f}")

    return beta0, beta1, cost_history


if __name__ == "__main__":
    df = pd.read_csv("medical-insurance-prediction\data\insurance.csv")
    X_raw = df["age"].values.astype(float)
    y = df["charges"].values.astype(float)

    # --- Feature scaling (standardization) ---
    X_mean = X_raw.mean()
    X_std = X_raw.std()
    X_scaled = (X_raw - X_mean) / X_std

    beta0, beta1, history = gradient_descent(
        X_scaled, y,
        beta0_init=0.0, beta1_init=0.0,
        learning_rate=0.01,
        n_iterations=1000
    )

    print("\nFinal parameters (on SCALED age):")
    print(f"beta0 (intercept) = {beta0:.4f}")
    print(f"beta1 (slope, per std-dev of age) = {beta1:.4f}")

    # --- Convert back to original units for interpretability ---
    beta1_unscaled = beta1 / X_std
    beta0_unscaled = beta0 - beta1 * (X_mean / X_std)

    print("\nConverted back to ORIGINAL age units:")
    print(f"beta0 (intercept) = {beta0_unscaled:.4f}")
    print(f"beta1 (dollars per year of age) = {beta1_unscaled:.4f}")