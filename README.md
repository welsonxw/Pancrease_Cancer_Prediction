# 🔬 Pancreatic Cancer Survival Prediction
### *A Machine Learning Approach to Cancer Prognosis*

---

## 👥 Group 10 - Team Members

| Name | Matric Number |
|------|---------------|
| Welson Woong Lu Bin | A23CS0196 |
| Ravinesh A/L Maran | A23CS0175 |
| Bernice Lou Min Yun | A23CS0056 |

---

## 🎯 Mission: Predicting Pancreatic Cancer Outcomes

Pancreatic cancer remains one of the **deadliest cancers worldwide**, with survival rates heavily influenced by early detection and accurate prognosis. Misclassification of survival outcomes can lead to inappropriate treatment plans and missed intervention opportunities.

This project, developed for the **Programming for Bioinformatics (SECB3203)** course, tackles this critical healthcare challenge. Using machine learning on clinical patient data, we build a robust, data-driven classifier to predict survival status.

**The Goal:** To accurately predict patient survival outcomes based on clinical features, demographics, and lifestyle factors.

| Feature Category | Examples |
|-----------------|----------|
| Demographics | Age, Gender, Country |
| Medical History | Diabetes, Obesity, Smoking History |
| Diagnosis | Stage at Diagnosis |
| Treatment | Treatment Type |
| Lifestyle | Physical Activity, Diet, Healthcare Access |

---

## 🔧 The Pipeline: From Raw Data to Predictive Model

This repository contains the complete machine learning pipeline for binary classification:

### 📥 Progress 2: Data Acquisition & Wrangling
- **Importing Data**: Loading CSV dataset into Python environment
- **Data Cleaning**: Handling missing values and removing duplicates
- **Standardization**: Column names and text normalization
- **Feature Engineering**: Data normalization using `StandardScaler`
- **Binning**: Age grouping (Below-Average, Average, Above-Average)
- **Encoding**: Creating indicator variables with `get_dummies()`

```
pancreatic_cancer_prediction_sample.csv → Data Wrangling → pancreatic_cancer_data_processed.csv
```

### 📊 Progress 3: Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Understanding data distributions
- **Grouping Analysis**: Survival patterns by age, stage, treatment, smoking
- **ANOVA Testing**: Statistical significance of survival time vs diagnosis stage
- **Correlation Analysis**: Heatmap visualization of feature relationships

### 🤖 Progress 4: Model Development
- **Simple Linear Regression**: Age → Survival Time
- **Multiple Linear Regression**: Age + Smoking + Diabetes + Obesity → Survival
- **Polynomial Regression**: Non-linear relationships with degree=2
- **Visualization**: Scatter plots with regression lines
- **Evaluation Metrics**: MSE and R² for in-sample evaluation

### 🎯 Progress 5: Model Evaluation & Selection
- **Train-Test Split**: 80/20 split with stratification
- **Models Evaluated**:

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline classifier |
| Ridge Classifier | Regularization to prevent overfitting |
| Random Forest | Ensemble learning |
| Decision Tree | Interpretable model |
| KNN | Instance-based learning |
| XGBoost | Gradient boosting |

- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Model Comparison**: Accuracy, Precision, Recall, F1-Score

---

## 📁 Repository Structure

```
Pancrease_Cancer_Prediction/
│
├── 📓 all_progress.ipynb              # Complete Jupyter notebook
├── 📄 pancreatic_cancer_prediction_sample.csv   # Raw dataset
├── 📄 pancreatic_cancer_data_processed.csv      # Processed dataset
├── 🐍 progress3.py                    # EDA scripts
├── 🐍 progress4.py                    # Model development
├── 🐍 progress5.py                    # Model evaluation
├── 🐍 original.py                     # Original code
└── 📖 README.md                       # Project documentation
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Persistence | Joblib |

---

## 🚀 Quick Start

```python
# Load the trained model
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
features = joblib.load("features.pkl")

# Make predictions
prediction = model.predict(new_patient_data)
```

---

## 📈 Key Findings

- Multiple machine learning models were trained and compared
- Feature importance analysis reveals key predictors of survival
- Model persistence enables deployment for real-world predictions

---

## 📚 Course Information

**Course**: SECB3203 - Programming for Bioinformatics  
**Semester**: 2526/1  
**Institution**: Universiti Teknologi Malaysia

---

<p align="center">
  <i>Built with 💻 and ☕ by Group 10</i>
</p>
