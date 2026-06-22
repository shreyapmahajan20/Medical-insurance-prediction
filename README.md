# Medical Insurance Charge Prediction

A linear regression project predicting individual medical insurance charges from demographic and health data — built from scratch (no `sklearn` for the core model) to deeply understand the mechanics of linear regression: cost functions, gradient descent, OLS, and feature engineering.

## Dataset

[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) — 1,338 records with `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target variable `charges`.

## Project Goals

This project was built to deeply internalize core linear regression concepts by implementing everything from first principles, then validating each idea against real, measurable results rather than assuming textbook explanations apply blindly:

- Cost functions and why MSE is used over alternatives
- Gradient descent, including diagnosing and fixing real divergence
- Multiple linear regression with proper categorical encoding
- Evaluation metrics (R², MAE, MSE, RMSE) and what each one reveals
- Overfitting/underfitting via learning curves
- OLS (Normal Equation) derived by hand and verified against gradient descent
- Polynomial features and interaction terms — including a documented case where a reasonable hypothesis (age²) failed, and a different one (smoker × bmi) succeeded substantially

## Methodology & Results

### 1. Baseline: Simple Linear Regression (age only)

Fit `charges = β₀ + β₁ · age` using gradient descent implemented from scratch in NumPy.

**Result: R² = 0.089** — age alone explains only ~9% of charge variance.

Initial attempts diverged (cost exploded to `inf`/`nan`) due to feature scale mismatch between `age` (18–64) and `charges` (up to $63,770). Fixed by standardizing features before training — a direct, hands-on encounter with why gradient descent requires scaled inputs.

### 2. Multiple Linear Regression

Added `bmi`, `children`, `smoker`, `sex`, `region` (one-hot encoded, `drop_first=True` to avoid the dummy variable trap).

**Result: R² = 0.74 (train) / 0.78 (test)**

| Feature | Coefficient (scaled) |
|---|---|
| smoker_yes | +9,556 |
| age | +3,613 |
| bmi | +2,034 |
| children | +517 |
| region (all) | -158 to -353 |
| sex_male | -7 (negligible) |

`smoker` dominates; `sex` and `region` contribute almost nothing.

### 3. Evaluation Metrics

- **Test RMSE: 5,805** vs **Test MAE: 4,187** — the meaningful gap reveals a handful of large outlier errors (largest misses: $17,000–$23,000), not uniform error across all predictions. **Median absolute error was only $2,693**, much lower than the mean, confirming most predictions are accurate and a small subset of patients drive the larger average error — likely individuals with health-cost drivers not captured by this dataset.

### 4. Overfitting/Underfitting Check

Built a learning curve (training set size 20 → 1070 rows). Train and test R² converge to the same ~0.74–0.78 band as training size grows, with no widening gap — indicating a well-generalizing model, not overfitting. The curve flattening (rather than continuing to climb) shows this feature set has a performance ceiling around R² ≈ 0.78.

### 5. OLS (Normal Equation)

Derived β = (XᵀX)⁻¹Xᵀy directly from setting ∇J(β) = 0, then implemented and compared against gradient descent.

**Result: identical coefficients to 6 decimal places**, confirming both methods correctly solve the same optimization problem. Documented in code why gradient descent remains the standard choice for high-dimensional or non-closed-form problems (matrix inversion cost scales O(n³); gradient descent scales with data size and generalizes to models with no closed-form solution).

### 6. Feature Engineering: Polynomial vs. Interaction Terms

Two hypotheses were tested, with very different outcomes — both are documented because the negative result is as informative as the positive one:

| Model | Test R² | Test RMSE | Verdict |
|---|---|---|---|
| Baseline (linear) | 0.7837 | 5,805.15 | — |
| + age², bmi² | 0.7848 | 5,790.35 | **Negligible improvement.** Age's effect does accelerate in raw data, but this signal is dwarfed by the smoker effect once mixed across the full population. |
| + smoker × bmi interaction | **0.8658** | **4,573.22** | **Major improvement.** BMI's effect on charges is ~14x stronger for smokers than non-smokers in the raw data — a true interaction effect that additive linear terms cannot represent. |

**Why the difference matters:** a polynomial term (age²) lets one variable bend its own curve, the same way for everyone. An interaction term (smoker × bmi) lets the relationship between one variable and the target *change depending on another variable*. Testing the raw data first (average charges by BMI group, split by smoker status) showed the interaction hypothesis was far better supported before any model was even fit — feature engineering driven by data exploration outperformed engineering driven by assumption.

**Known limitation:** with the interaction term included, individual coefficients (`smoker_yes`, `smoker_bmi`) cannot be read in isolation — their *sum* gives the real effect, and that combined effect becomes unreliable at the sparse extremes of the smoker/low-BMI subgroup, where few training examples exist. This is flagged honestly rather than hidden: the model is meaningfully better in aggregate, but extrapolates unreliably at thin-data extremes.

## Final Model

Multiple Linear Regression with the `smoker × bmi` interaction term, trained via gradient descent.

- **Test R²: 0.8658**
- **Test RMSE: $4,573**
- **Test MAE: $2,772**

## Project Structure

```
medical-insurance-prediction/
├── data/
│   └── insurance.csv
├── src/
│   ├── preprocessing.py          # encoding, train/test split, scaling
│   ├── linear_reg_scratch.py     # simple linear regression (1 feature)
│   ├── linear_reg_multi.py       # vectorized multiple linear regression
│   ├── learning_curve.py         # overfitting/underfitting diagnostic
│   ├── ols.py                    # closed-form Normal Equation solution
│   ├── polynomial_reg.py         # age^2 / bmi^2 experiment
│   ├── interaction_term.py       # smoker x bmi experiment
│   ├── train.py                  # final unified training pipeline
│   └── predict.py                # predict charges for a new patient
├── models/                       # saved model artifacts (gitignored)
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone <your-repo-url>
cd medical-insurance-prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the final model
python src/train.py

# Predict on new patients (edit examples in src/predict.py, or import predict())
python src/predict.py
```

## Key Takeaways

1. **Feature scaling isn't optional with gradient descent** — unscaled features with very different ranges create an elongated cost surface that causes divergence, not just slow convergence.
2. **Scaling changes coefficient interpretation, not the underlying relationship** — coefficients on standardized features represent "effect per standard deviation," not "effect per raw unit," and must be converted back for reporting.
3. **A held-out test set is non-negotiable**, and any preprocessing statistics (mean, std) must be fit on the training set only, to avoid data leakage into evaluation.
4. **R² alone hides what kind of errors a model makes** — RMSE vs. MAE divergence reveals outlier-driven error even when overall fit looks reasonable.
5. **Domain-informed hypotheses beat blind feature engineering.** Exploring the raw data (charges by age group, charges by BMI-and-smoker group) before modeling correctly predicted which engineered feature (interaction term) would help, and which (polynomial term) would not.
6. **A better model isn't uniformly better everywhere.** The best-performing model here has a real, documented blind spot at sparse data extremes — worth stating plainly rather than overselling aggregate metrics.