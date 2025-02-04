#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[ ]:




