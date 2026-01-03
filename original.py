import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from warnings import filterwarnings
filterwarnings('ignore')

# 1. Load the dataset
df = pd.read_csv(r"C:\bioinfo2\p4b\project\pancreatic_cancer_prediction_sample.csv")
print(df.info())
print(df.describe())

# 2. Analyze missing values
missing_values = df.isnull().sum()
if missing_values.any():
    print("Missing values per column:")
    print(missing_values[missing_values > 0])

stage_survival = df.groupby('Stage_at_Diagnosis')['Survival_Time_Months'].mean()

plt.figure(figsize=(10, 6))
stage_survival.plot(kind='bar', color='skyblue')
plt.title('Average Survival Time by Stage at Diagnosis')
plt.xlabel('Stage at Diagnosis')
plt.ylabel('Average Survival Time (Months)')
plt.show()
plt.figure(figsize=(10, 6))
plt.hist(df['Survival_Time_Months'], bins=20, color='purple', alpha=0.7)
plt.title('Distribution of Survival Time')
plt.xlabel('Survival Time (Months)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Family_History', y='Survival_Time_Months', data=df, inner='quartile')
plt.title('Violin Plot of Survival Time by Family History')
plt.xlabel('Family History (1=Yes, 0=No)')
plt.ylabel('Survival Time (Months)')
plt.show()

plt.figure(figsize=(10, 6))
df['Age'].plot(kind='hist', bins=15, color='teal', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
df['Treatment_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Distribution of Treatment Types')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Survival_Time_Months'], bins=20, color='orange', alpha=0.7)
plt.title('Distribution of Survival Time')
plt.xlabel('Survival Time (Months)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
df['Family_History'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Family History Distribution')
plt.ylabel('')
plt.show()

plt.figure(figsize=(8, 8))
df['Alcohol_Consumption'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Alcohol Consumption Distribution')
plt.ylabel('')
plt.show()

plt.figure(figsize=(10, 6))
age_mean_survival = df.groupby('Age')['Survival_Time_Months'].mean()
plt.plot(age_mean_survival.index, age_mean_survival.values, color='blue', linestyle='-', marker='o')
plt.title('Change in Survival Time by Age')
plt.xlabel('Age')
plt.ylabel('Average Survival Time (Months)')
plt.grid()
plt.show()

fig = px.pie(df, names='Treatment_Type', title='Distribution of Treatment Types',
             labels={'Treatment_Type': 'Treatment Type'}, hole=0.3)
fig.show()

fig = px.box(df, x='Obesity', y='Survival_Time_Months',
             title='Box Plot of Survival Time by Obesity',
             labels={'Obesity': 'Obesity (1=Yes, 0=No)', 'Survival_Time_Months': 'Survival Time (Months)'})
fig.show()

fig = px.histogram(df, x='Alcohol_Consumption', color='Gender',
                   title='Alcohol Consumption Distribution by Gender',
                   labels={'Alcohol_Consumption': 'Alcohol Consumption'})
fig.update_traces(opacity=0.75)
fig.show()

# 3. Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# For numeric columns, impute with median
for col in numeric_columns:
    df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, impute with mode
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

    # 4. Map ordinal categorical columns
ordinal_columns = [
    'Physical_Activity_Level',
    'Diet_Processed_Food',
    'Access_to_Healthcare'
]
ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
for col in ordinal_columns:
    df[col] = df[col].map(ordinal_mapping)

# 5. One-hot encode nominal categorical columns
nominal_columns = [
    'Gender', 'Country', 'Stage_at_Diagnosis',
    'Treatment_Type', 'Urban_vs_Rural', 'Economic_Status'
]
df = pd.get_dummies(df, columns=nominal_columns)

# 6. Define features and target variable
X = df.drop(['Survival_Status', 'Survival_Time_Months'], axis=1)
y = df['Survival_Status']

# 7. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9. Optimize model hyperparameters using RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=5, random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# 10. Make predictions
y_pred = best_model.predict(X_test)

# 11. Evaluate the model
print("\nModel Performance:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Perform cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 13. Analyze feature importance
importance = best_model.feature_importances_
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# 14. Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()

# 15. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

import joblib
joblib.dump(best_model, 'random_forest_model.pkl')
loaded_model = joblib.load('random_forest_model.pkl')