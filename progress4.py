# ===============================
# Project Progress 4
# Model Development
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\bioinfo2\p4b\project\pancreatic_cancer_prediction_sample.csv")

# -------------------------------
# Encode categorical levels
# -------------------------------
level_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Physical_Activity_Level"] = df["Physical_Activity_Level"].map(level_mapping)
df["Alcohol_Consumption"] = df["Alcohol_Consumption"].map(level_mapping)

# -------------------------------
# Select Features & Target
# -------------------------------
X_simple = df[["Age"]]
X_multiple = df[
    [
        "Age",
        "Obesity",
        "Alcohol_Consumption",
        "Physical_Activity_Level"
    ]
]

y = df["Survival_Time_Months"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

X_train_m, X_test_m, _, _ = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

# ===============================
# 1. SIMPLE LINEAR REGRESSION
# ===============================
simple_lr = LinearRegression()
simple_lr.fit(X_train_s, y_train)

y_pred_simple = simple_lr.predict(X_train_s)

mse_simple = mean_squared_error(y_train, y_pred_simple)
r2_simple = r2_score(y_train, y_pred_simple)

print("\nSimple Linear Regression")
print("MSE:", mse_simple)
print("R-squared:", r2_simple)

# Visualization
plt.scatter(X_train_s, y_train)
plt.plot(X_train_s, y_pred_simple)
plt.xlabel("Age")
plt.ylabel("Survival Time (Months)")
plt.title("Simple Linear Regression")
plt.show()

# ===============================
# 2. MULTIPLE LINEAR REGRESSION
# ===============================
multiple_lr = LinearRegression()
multiple_lr.fit(X_train_m, y_train)

y_pred_multiple = multiple_lr.predict(X_train_m)

mse_multiple = mean_squared_error(y_train, y_pred_multiple)
r2_multiple = r2_score(y_train, y_pred_multiple)

print("\nMultiple Linear Regression")
print("MSE:", mse_multiple)
print("R-squared:", r2_multiple)

# ===============================
# 3. POLYNOMIAL REGRESSION (PIPELINE)
# ===============================
poly_pipeline = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression())
    ]
)

poly_pipeline.fit(X_train_s, y_train)

y_pred_poly = poly_pipeline.predict(X_train_s)

mse_poly = mean_squared_error(y_train, y_pred_poly)
r2_poly = r2_score(y_train, y_pred_poly)

print("\nPolynomial Regression (Degree 2)")
print("MSE:", mse_poly)
print("R-squared:", r2_poly)

# Visualization
X_sorted = X_train_s.sort_values(by="Age")
y_poly_sorted = poly_pipeline.predict(X_sorted)

plt.scatter(X_train_s, y_train)
plt.plot(X_sorted, y_poly_sorted)
plt.xlabel("Age")
plt.ylabel("Survival Time (Months)")
plt.title("Polynomial Regression")
plt.show()

# ===============================
# 4. PREDICTION & DECISION MAKING
# ===============================
sample_patient = pd.DataFrame(
    {
        "Age": [65],
        "Obesity": [1],
        "Alcohol_Consumption": [2],
        "Physical_Activity_Level": [1]
    }
)

predicted_survival = multiple_lr.predict(sample_patient)

print("\nPredicted Survival Time (Months):", predicted_survival[0])
