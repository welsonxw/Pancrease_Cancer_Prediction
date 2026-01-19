# For EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler

# For Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.exceptions import FitFailedWarning

# import dataset from csv to python 
df = pd.read_csv(r"C:\bioinfo2\p4b\project\Pancrease_Cancer_Prediction\pancreatic_cancer_prediction_sample.csv")
df.head()

# export dataset 
df.to_csv("pancreatic_cancer_data_processed.csv", index=False)

df.describe().T

df.describe(include='object').T

# missing value and duplicate value check
print('Missing Value (%)')
print(df.isna().mean()*100)
print('\nDuplicate Row (%)')
print(df.duplicated().mean())

print("Number of duplicate rows BEFORE removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of duplicate rows AFTER removal:", df.duplicated().sum())

# standardise column names and text values for stage_at_diagnosis
df.columns = df.columns.str.lower()
df['gender'] = df['gender'].str.lower()
df['stage_at_diagnosis'] = df['stage_at_diagnosis'].str.replace(' ', '_')

print(df.columns)
print(df['gender'].unique())
print(df['stage_at_diagnosis'].unique())

# data normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age', 'survival_time_months']] = scaler.fit_transform(
    df[['age', 'survival_time_months']]
)

# actual
print(df[['age', 'survival_time_months']].describe())

# rounded
desc = df[['age', 'survival_time_months']].describe()
desc.loc['count'] = desc.loc['count'].astype(int)
desc.round(10)

# binning age column
#  avg = 64.54
df['age_group'] = pd.cut(
    df['age'],
    bins=[-3, -1, 0, 3],   # scaled values after StandardScaler
    labels=['Below-Average', 'Average', 'Above-Average']
)

print(df[['age', 'age_group']].head(10))
print(df['age_group'].value_counts())

# indicator variable 
df_encoded = pd.get_dummies(
    df,
    columns=['gender', 'treatment_type', 'stage_at_diagnosis', 'urban_vs_rural',
             'country', 'physical_activity_level', 'diet_processed_food', 
             'access_to_healthcare', 'economic_status', 'age_group'],
    drop_first=True
)

print(df_encoded.columns)
print(df_encoded.head())

# descriptive statistic

desc_numeric = df[['age', 'survival_time_months']].describe()
print(desc_numeric)

#binary variable 
categorical_cols = [
    'gender', 
    'stage_at_diagnosis',
    'treatment_type',
    'smoking_history',
    'survival_status'
]

for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())

    # basic grouping
# Survival (months)=(z-score×σ (std))+ μ(mean)

age_group_survival = df.groupby('age_group', observed=True)['survival_time_months'].mean()
print(age_group_survival)

print("\n")
stage_survival = df.groupby('stage_at_diagnosis')['survival_time_months'].mean()
print(stage_survival)

print("\n")
treatment_survival = df.groupby('treatment_type')['survival_time_months'].mean()
print(treatment_survival)

print("\n")
smoking_survival = df.groupby('smoking_history')['survival_time_months'].mean()
print(smoking_survival)

#  anova 

from scipy.stats import f_oneway

stage_groups = [
    df[df['stage_at_diagnosis'] == stage]['survival_time_months']
    for stage in df['stage_at_diagnosis'].unique()
]

f_stat_stage, p_value_stage = f_oneway(*stage_groups)

print("ANOVA: Survival Time vs Stage at Diagnosis")
print("F-statistic:", f_stat_stage)
print("p-value:", p_value_stage)


# correlation analysis

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

original_objects_cols = list(label_encoders.keys())
num_col = df.select_dtypes(include=['int64','float64']).columns

num_col_scale = [col for col in num_col if col not in original_objects_cols and col != 'Survival_Status']

scaler = StandardScaler()
df[num_col_scale] = scaler.fit_transform(df[num_col_scale])

df[num_col_scale].head()

# correlation analysis

label_encoders = {}
for column in df.select_dtypes(include=['object', 'category']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

correlation_matrix = df.corr(method='pearson')

plt.figure(figsize=(18, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

#  prepare data regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[['age']]   # simple regression
y = df['survival_time_months']

# simple linear regression 

lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred = lin_reg.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Simple Linear Regression")
print("MSE:", mse)
print("R²:", r2)

# visualization

plt.scatter(X, y, alpha=0.3)
plt.plot(X, y_pred, color='red')
plt.xlabel("Age (scaled)")
plt.ylabel("Survival Time (scaled)")
plt.title("Simple Linear Regression")
plt.show()

#  multiple linear regression

X_multi = df[['age', 'smoking_history', 'diabetes', 'obesity']]

multi_reg = LinearRegression()
multi_reg.fit(X_multi, y)

y_pred_multi = multi_reg.predict(X_multi)

print("Multiple Linear Regression")
print("MSE:", mean_squared_error(y, y_pred_multi))
print("R²:", r2_score(y, y_pred_multi))

# polynomial regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_model.fit(X, y)
y_pred_poly = poly_model.predict(X)

print("Polynomial Regression (degree=2)")
print("MSE:", mean_squared_error(y, y_pred_poly))
print("R²:", r2_score(y, y_pred_poly))

# prediction and decision making

example_age = pd.DataFrame([[0.5]], columns=['age'])  # above-average age (scaled)
predicted_survival = lin_reg.predict(example_age)

print("Predicted survival time (scaled):", predicted_survival)

# prepare data and train test split

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['survival_status'])
y = df_encoded['survival_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check for non-numeric columns in X_train
print("Non-numeric columns in X_train:")
print(X_train.select_dtypes(include=['object', 'category']).columns.tolist())

# logistic regression 

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

print("Logistic Regression (Train)")
print(classification_report(y_train, y_train_pred))

print("Logistic Regression (Test)")
print(classification_report(y_test, y_test_pred))

# overfitting and underfitting 

from sklearn.metrics import accuracy_score

print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Test accuracy :", accuracy_score(y_test, y_test_pred))

# ridge regression

from sklearn.linear_model import RidgeClassifier

ridge = RidgeClassifier(alpha=1.0)
ridge.fit(X_train, y_train)

y_train_pred_ridge = ridge.predict(X_train)
y_test_pred_ridge = ridge.predict(X_test)

print("Ridge Classifier (Train)")
print(classification_report(y_train, y_train_pred_ridge))

print("Ridge Classifier (Test)")
print(classification_report(y_test, y_test_pred_ridge))

# random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # reduce from 500/1000
    max_depth=15,          # limit tree depth
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

print("Random Forest (Train)")
print(classification_report(y_train, y_train_pred_rf))

print("Random Forest (Test)")
print(classification_report(y_test, y_test_pred_rf))

# grid seacrh

from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    RidgeClassifier(),
    param_grid,
    cv=5,
    scoring='f1'
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# model refinement 

best_ridge = grid.best_estimator_

y_test_pred_best = best_ridge.predict(X_test)

print("Tuned Ridge (Test)")
print(classification_report(y_test, y_test_pred_best))

# comparison between models

X = df_encoded.drop('survival_status', axis=1)
y = df_encoded['survival_status'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42)
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

results = []

for name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

results_df = pd.DataFrame(results)
results_df