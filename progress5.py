# ===============================
# Project Progress 5
# Model Evaluation & Refinement
# ===============================

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from warnings import filterwarnings
filterwarnings('ignore')
# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\bioinfo2\p4b\project\Pancrease_Cancer_Prediction\pancreatic_cancer_prediction_sample.csv")

# -------------------------------
# Select NUMERIC features only
# -------------------------------
features = [
    "Age",
    "Obesity",
    "Alcohol_Consumption"
]

X = df[features]
y = df["Survival_Time_Months"]

# Ensure numeric
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 1. LINEAR REGRESSION
# ===============================
linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

train_pred_lr = linear_model.predict(X_train)
test_pred_lr = linear_model.predict(X_test)

print("\nLinear Regression")
print("Train R2:", r2_score(y_train, train_pred_lr))
print("Test R2:", r2_score(y_test, test_pred_lr))
print("Train MSE:", mean_squared_error(y_train, train_pred_lr))
print("Test MSE:", mean_squared_error(y_test, test_pred_lr))

# ===============================
# 2. POLYNOMIAL REGRESSION
# ===============================
poly_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2)),
    ("model", LinearRegression())
])

poly_model.fit(X_train, y_train)

train_pred_poly = poly_model.predict(X_train)
test_pred_poly = poly_model.predict(X_test)

print("\nPolynomial Regression (Degree 2)")
print("Train R2:", r2_score(y_train, train_pred_poly))
print("Test R2:", r2_score(y_test, test_pred_poly))
print("Train MSE:", mean_squared_error(y_train, train_pred_poly))
print("Test MSE:", mean_squared_error(y_test, test_pred_poly))

# ===============================
# 3. RIDGE REGRESSION + GRID SEARCH
# ===============================
ridge_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", Ridge())
])

param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    scoring="r2",
    cv=5
)

grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_

train_pred_ridge = best_ridge.predict(X_train)
test_pred_ridge = best_ridge.predict(X_test)

print("\nRidge Regression (Tuned)")
print("Best alpha:", grid_search.best_params_)
print("Train R2:", r2_score(y_train, train_pred_ridge))
print("Test R2:", r2_score(y_test, test_pred_ridge))
print("Train MSE:", mean_squared_error(y_train, train_pred_ridge))
print("Test MSE:", mean_squared_error(y_test, test_pred_ridge))

# ===============================
# MODEL SELECTION SUMMARY
# ===============================
print("\nModel Selection Summary (Test R2)")
print("Linear:", r2_score(y_test, test_pred_lr))
print("Polynomial:", r2_score(y_test, test_pred_poly))
print("Ridge:", r2_score(y_test, test_pred_ridge))
