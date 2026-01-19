import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(r"C:\bioinfo2\p4b\project\Pancrease_Cancer_Prediction\pancreatic_cancer_prediction_sample.csv")

df.columns = df.columns.str.lower()
df['gender'] = df['gender'].str.lower()
df['stage_at_diagnosis'] = df['stage_at_diagnosis'].str.replace(' ', '_')

# -----------------------------
# Encode categorical variables
# -----------------------------
df_encoded = pd.get_dummies(
    df,
    columns=[
        'gender', 'treatment_type', 'stage_at_diagnosis',
        'urban_vs_rural', 'country', 'physical_activity_level',
        'diet_processed_food', 'access_to_healthcare',
        'economic_status'
    ],
    drop_first=True
)

# -----------------------------
# Split features & target
# -----------------------------
X = df_encoded.drop(columns=['survival_status'])
y = df_encoded['survival_status']

joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# -----------------------------
# Scale data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("✅ Training completed. Model saved.")
