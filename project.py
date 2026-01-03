
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv(r"C:\bioinfo2\p4b\project\pancreatic_cancer_prediction_sample.csv")
print(df.info())
print(df.describe())

stage_survival = df.groupby('Stage_at_Diagnosis')['Survival_Time_Months'].mean()
stage_survival.plot(kind='bar', color='skyblue', figsize=(8,5))
plt.title('Average Survival Time by Stage at Diagnosis')
plt.xlabel('Stage at Diagnosis')
plt.ylabel('Average Survival Time (Months)')
plt.show()

plt.hist(df['Survival_Time_Months'], bins=20, color='purple', alpha=0.7)
plt.title('Distribution of Survival Time')
plt.xlabel('Survival Time (Months)')
plt.ylabel('Frequency')
plt.show()

sns.violinplot(x='Family_History', y='Survival_Time_Months', data=df, inner='quartile')
plt.title('Survival Time by Family History')
plt.xlabel('Family History (1=Yes, 0=No)')
plt.ylabel('Survival Time (Months)')
plt.show()

plt.hist(df['Age'], bins=15, color='teal', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

df['Treatment_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, figsize=(6,6))
plt.title('Treatment Type Distribution')
plt.ylabel('')
plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

ordinal_cols = ['Physical_Activity_Level', 'Diet_Processed_Food', 'Access_to_Healthcare']
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
for col in ordinal_cols:
    df[col] = df[col].map(ordinal_map)

nominal_cols = ['Gender', 'Country', 'Stage_at_Diagnosis', 'Treatment_Type', 'Urban_vs_Rural', 'Economic_Status']
df = pd.get_dummies(df, columns=nominal_cols)

X = df.drop(['Survival_Status', 'Survival_Time_Months'], axis=1)
y = df['Survival_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("Cross-validation scores:", cv_scores)
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()

joblib.dump(best_model, 'random_forest_model.pkl')
loaded_model = joblib.load('random_forest_model.pkl')
