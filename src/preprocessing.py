import pandas as pd
import numpy as np


def load_and_encode(filepath="medical-insurance-prediction\data\insurance.csv"):
    """
    Loads the insurance dataset and one-hot encodes categorical columns.
    drop_first=True avoids the dummy variable trap.
    """
    df = pd.read_csv(filepath)
    df_encoded = pd.get_dummies(
        df,
        columns=["sex", "smoker", "region"],
        drop_first=True
    )
    # pd.get_dummies gives bool columns; convert to int for clean math
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    return df_encoded


def train_test_split_manual(X, y, test_ratio=0.2, seed=42):
    """
    Manual train/test split (no sklearn) so you see exactly what's happening.
    """
    np.random.seed(seed)
    m = len(y)
    indices = np.random.permutation(m)
    test_size = int(m * test_ratio)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """
    Standardize features using TRAIN set statistics only.
    Critical: test set must be scaled using train mean/std, never its own.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


if __name__ == "__main__":
    df_encoded = load_and_encode()
    print(df_encoded.head())
    print("\nColumns:", df_encoded.columns.tolist())
    print("\nShape:", df_encoded.shape)