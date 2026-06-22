"""
predict.py — Predict insurance charges for a new patient using the trained model.

Run from repo root:
    python src/predict.py
"""

import sys
sys.path.insert(0, "src")
import numpy as np
import joblib

MODEL_PATH = "models/model.joblib"


def encode_new_patient(age, sex, bmi, children, smoker, region, feature_cols):
    """
    Manually one-hot encodes a single new patient the SAME way training data
    was encoded (drop_first=True, same baseline categories dropped:
    sex_female, smoker_no, region_northeast).
    """
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    row["smoker_bmi"] = row["smoker_yes"] * row["bmi"]

    # Order must exactly match feature_cols used in training
    x = np.array([row[col] for col in feature_cols], dtype=float)
    return x


def predict(age, sex, bmi, children, smoker, region):
    artifact = joblib.load(MODEL_PATH)
    beta = artifact["beta"]
    feature_cols = artifact["feature_cols"]
    mean = artifact["mean"]
    std = artifact["std"]

    x = encode_new_patient(age, sex, bmi, children, smoker, region, feature_cols)

    # Apply TRAIN mean/std — never compute new stats for a single new patient
    x_scaled = (x - mean) / std
    x_final = np.concatenate([[1.0], x_scaled])  # add intercept term

    predicted_charge = x_final @ beta
    return predicted_charge


if __name__ == "__main__":
    examples = [
        dict(age=19, sex="female", bmi=27.9, children=0, smoker="yes", region="southwest"),
        dict(age=45, sex="male", bmi=29.0, children=2, smoker="no", region="northeast"),
        dict(age=50, sex="male", bmi=38.0, children=1, smoker="yes", region="southeast"),
    ]

    print("=== Predictions for sample patients ===\n")
    for p in examples:
        charge = predict(**p)
        print(f"Patient: {p}")
        print(f"  --> Predicted charge: ${charge:,.2f}\n")