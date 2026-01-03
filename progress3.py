# ===============================
# Project Progress 3
# Exploratory Data Analysis (FIXED)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

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
# Descriptive Statistics
# -------------------------------
numerical_cols = [
    "Age",
    "Survival_Time_Months",
    "Obesity",
    "Alcohol_Consumption",
    "Physical_Activity_Level"
]

print("\nDescriptive Statistics:")
print(df[numerical_cols].describe())

# -------------------------------
# Basic Grouping
# -------------------------------
print("\nMean Age and Survival Time by Stage at Diagnosis:")
print(df.groupby("Stage_at_Diagnosis")[["Age", "Survival_Time_Months"]].mean())

# -------------------------------
# ANOVA (Adaptive & Safe)
# -------------------------------
anova_groups_age = []
anova_groups_survival = []

for stage in df["Stage_at_Diagnosis"].unique():
    age_values = df[df["Stage_at_Diagnosis"] == stage]["Age"]
    survival_values = df[df["Stage_at_Diagnosis"] == stage]["Survival_Time_Months"]

    if len(age_values) >= 2:
        anova_groups_age.append(age_values)

    if len(survival_values) >= 2:
        anova_groups_survival.append(survival_values)

if len(anova_groups_age) >= 2:
    f_age, p_age = f_oneway(*anova_groups_age)
    print("\nANOVA: Age vs Stage at Diagnosis")
    print("F-statistic:", f_age)
    print("P-value:", p_age)
else:
    print("\nANOVA: Age vs Stage at Diagnosis — Not enough data")

if len(anova_groups_survival) >= 2:
    f_survival, p_survival = f_oneway(*anova_groups_survival)
    print("\nANOVA: Survival Time vs Stage at Diagnosis")
    print("F-statistic:", f_survival)
    print("P-value:", p_survival)
else:
    print("\nANOVA: Survival Time vs Stage at Diagnosis — Not enough data")

# -------------------------------
# Correlation Analysis
# -------------------------------
correlation_data = df[
    [
        "Age",
        "Survival_Time_Months",
        "Obesity",
        "Alcohol_Consumption",
        "Physical_Activity_Level"
    ]
]

correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8, 6))
plt.matshow(correlation_matrix, fignum=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns)
plt.title("Correlation Matrix", pad=20)
plt.show()
