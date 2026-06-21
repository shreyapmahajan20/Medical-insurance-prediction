import sys
sys.path.insert(0, "src")
import numpy as np
from preprocessing import load_and_encode, train_test_split_manual, standardize
from linear_reg_multi import add_intercept, gradient_descent, r_squared


def learning_curve(X, y, train_sizes, test_ratio=0.2, seed=42,
                    learning_rate=0.1, n_iterations=1000):
    """
    For each size in train_sizes, train on that many rows (subset of train set)
    and record train R^2 and test R^2 on a FIXED held-out test set.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split_manual(
        X, y, test_ratio=test_ratio, seed=seed
    )

    results = []
    for size in train_sizes:
        X_sub = X_train_full[:size]
        y_sub = y_train_full[:size]

        # scale using ONLY this subset's stats (mimics what you'd actually
        # have available if you'd only collected `size` data points)
        X_sub_scaled, X_test_scaled, mean, std = standardize(X_sub, X_test)

        X_sub_final = add_intercept(X_sub_scaled)
        X_test_final = add_intercept(X_test_scaled)

        beta, _ = gradient_descent(
            X_sub_final, y_sub,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            verbose=False
        )

        train_r2 = r_squared(X_sub_final, y_sub, beta)
        test_r2 = r_squared(X_test_final, y_test, beta)

        results.append((size, train_r2, test_r2))
        print(f"train_size={size:4d}  train_R2={train_r2:.4f}  test_R2={test_r2:.4f}")

    return results


if __name__ == "__main__":
    df_encoded = load_and_encode()
    feature_cols = [c for c in df_encoded.columns if c != "charges"]
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded["charges"].values.astype(float)

    train_sizes = [20, 50, 100, 200, 400, 700, 1070]
    learning_curve(X, y, train_sizes)