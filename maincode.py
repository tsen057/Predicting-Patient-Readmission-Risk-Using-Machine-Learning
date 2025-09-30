#!/usr/bin/env python
# coding: utf-8

# In[1]:




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
diabetic_data = pd.read_csv("C:/Users/tejas/Downloads/Diabetics-dataset/dataset_diabetes/diabetic_data.csv")
id_mapping = pd.read_csv("C:/Users/tejas/Downloads/Diabetics-dataset/dataset_diabetes/IDs_mapping.csv")

# Ensure both columns have the same type
diabetic_data["admission_type_id"] = diabetic_data["admission_type_id"].astype(str)
id_mapping["admission_type_id"] = id_mapping["admission_type_id"].astype(str)

# Merge datasets
diabetic_data = diabetic_data.merge(id_mapping, on="admission_type_id", how="left")

# Drop unnecessary columns
diabetic_data.drop(columns=["encounter_id", "patient_nbr"], inplace=True)

# Handle missing values
diabetic_data.replace("?", np.nan, inplace=True)
diabetic_data.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = diabetic_data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col])
    label_encoders[col] = le

# Define features and target variable
X = diabetic_data.drop(columns=["readmitted"])
y = diabetic_data["readmitted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# In[8]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve,
    f1_score, recall_score
)
from sklearn.calibration import calibration_curve
from joblib import dump

# -----------------------------
# CONFIG: set your CSV paths
# -----------------------------
DIABETIC_CSV = r"C:/Users/tejas/Downloads/Diabetics-dataset/dataset_diabetes/diabetic_data.csv"
ID_MAP_CSV   = r"C:/Users/tejas/Downloads/Diabetics-dataset/dataset_diabetes/IDs_mapping.csv"

OUT_DIR = "assets"           # where plots/metrics will be written
MODEL_OUT = "readmission_model.joblib"  # saved pipeline

# -----------------------------
# LOAD & MERGE
# -----------------------------
diabetic = pd.read_csv(DIABETIC_CSV)
id_mapping = pd.read_csv(ID_MAP_CSV)

# Ensure same dtype for merge key
diabetic["admission_type_id"] = diabetic["admission_type_id"].astype(str)
id_mapping["admission_type_id"] = id_mapping["admission_type_id"].astype(str)

# Merge (optional but matches your original intent)
diabetic = diabetic.merge(id_mapping, on="admission_type_id", how="left")

# -----------------------------
# TARGET: binary (1 = <30 days, 0 = NO or >30)
# -----------------------------
diabetic["readmitted"] = diabetic["readmitted"].astype(str).str.strip()
y = (diabetic["readmitted"] == "<30").astype(int)

# -----------------------------
# CLEANUP / MISSING:
# - Replace '?' with 'Unknown' in object columns (don’t drop rows)
# - Keep patient_nbr for splitting; drop it AFTER split
# -----------------------------
obj_cols = diabetic.select_dtypes(include=["object"]).columns
for c in obj_cols:
    diabetic[c] = diabetic[c].replace("?", "Unknown")

# We will split by patient_nbr to avoid leakage. Keep encounter for ref (not used).
if "patient_nbr" not in diabetic.columns:
    raise ValueError("Expected 'patient_nbr' column is missing from the dataset.")

# -----------------------------
# FEATURE FRAME (keep patient_nbr for split, drop readmitted only)
# -----------------------------
X_all = diabetic.drop(columns=["readmitted"], errors="ignore")
groups = diabetic["patient_nbr"]

# -----------------------------
# PATIENT-LEVEL SPLIT
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_all, y, groups=groups))

X_train_raw = X_all.iloc[train_idx].copy()
y_train     = y.iloc[train_idx].copy()
X_test_raw  = X_all.iloc[test_idx].copy()
y_test      = y.iloc[test_idx].copy()

# Now it is safe to drop patient_nbr / encounter_id so they don't leak into the model
for col in ["patient_nbr", "encounter_id"]:
    if col in X_train_raw.columns:
        X_train_raw.drop(columns=[col], inplace=True)
    if col in X_test_raw.columns:
        X_test_raw.drop(columns=[col], inplace=True)

# -----------------------------
# COLUMN TYPES
# -----------------------------
cat_cols = [c for c in X_train_raw.columns if X_train_raw[c].dtype == "object"]
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

# -----------------------------
# PREPROCESSOR (impute + encode/scale)
# -----------------------------
categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
])
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))  # keep sparse compatibility
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipe, cat_cols),
        ("num", numeric_pipe, num_cols),
    ]
)

# -----------------------------
# MODELS
# - You can switch to LogisticRegression baseline if you want
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
# logreg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")

model = rf  # or logreg

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", model)
])

# -----------------------------
# FIT
# -----------------------------
pipe.fit(X_train_raw, y_train)

# -----------------------------
# PREDICT & METRICS
# -----------------------------
proba = pipe.predict_proba(X_test_raw)[:, 1]
y_pred = (proba >= 0.5).astype(int)

ap = average_precision_score(y_test, proba)       # PR-AUC
f1m = f1_score(y_test, y_pred, average="macro")   # macro-F1
rec_pos = recall_score(y_test, y_pred, pos_label=1)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Average Precision (PR-AUC): {ap:.4f}")
print(f"F1-macro: {f1m:.4f}")
print(f"Recall (positive/readmit=1): {rec_pos:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))

# -----------------------------
# PLOTS 
# -----------------------------
import sys

os.makedirs(OUT_DIR, exist_ok=True)

# Text metrics
ap = average_precision_score(y_test, proba)       # PR-AUC
f1m = f1_score(y_test, y_pred, average="macro")   # macro-F1
rec_pos = recall_score(y_test, y_pred, pos_label=1)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Average Precision (PR-AUC): {ap:.4f}")
print(f"F1-macro: {f1m:.4f}")
print(f"Recall (positive/readmit=1): {rec_pos:.4f}\n")

report = classification_report(y_test, y_pred, digits=3)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
sys.stdout.flush()  # make sure output appears in some IDEs/Jupyter setups

# Confusion Matrix (save + show)
plt.figure(figsize=(5.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.show()  # <-- show

# Precision-Recall Curve (save + show)
precision, recall, _ = precision_recall_curve(y_test, proba)
plt.figure()
plt.step(recall, precision, where="post")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve | AP={ap:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"))
plt.show()  # <-- show

# Calibration Curve (save + show)
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calibration_curve.png"))
plt.show()  # <-- show

# Save metrics to text
with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Average Precision (PR-AUC): {ap:.4f}\n")
    f.write(f"F1-macro: {f1m:.4f}\n")
    f.write(f"Recall (positive): {rec_pos:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))

# -----------------------------
# OPTIONAL: SHAP EXPLANATIONS
# -----------------------------
try:
    import shap
    # Only meaningful for tree models; skip if not RandomForest
    if isinstance(model, RandomForestClassifier):
        pre = pipe.named_steps["preprocessor"]
        clf = pipe.named_steps["clf"]

        # Transform a sample of training data to keep SHAP tractable
        Xt = pre.transform(X_train_raw)
        sample_n = min(2000, Xt.shape[0])
        rng = np.random.RandomState(42)
        if hasattr(Xt, "toarray"):
            idx = rng.choice(Xt.shape[0], sample_n, replace=False)
            Xt_sample = Xt[idx].toarray()
        else:
            idx = rng.choice(Xt.shape[0], sample_n, replace=False)
            Xt_sample = Xt[idx]

        # Feature names
        cat_features = pre.transformers_[0][2]
        num_features = pre.transformers_[1][2]
        ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(cat_features)
        feature_names = list(ohe_names) + list(num_features)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Xt_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv = shap_values

        shap.summary_plot(sv, features=Xt_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"))
        plt.close()
        print("Saved SHAP summary to assets/shap_summary.png")
except Exception as e:
    print(f"[SHAP] Skipping SHAP due to: {e}")

# -----------------------------
# SAVE MODEL (full pipeline)
# -----------------------------
dump(pipe, MODEL_OUT)
print(f"Saved pipeline model to: {MODEL_OUT}")


# 
