# Predicting Patient Readmission Risk (30‑Day) — Machine Learning

> **Purpose**: Build an interpretable, reproducible model to estimate the probability of 30‑day hospital readmission using the UCI diabetic hospital dataset (130 US hospitals, 1999–2008). Emphasis on clinical plausibility, fairness checks, and deployment readiness (monitoring + thresholding).

---

##  Problem Statement

Hospitals aim to identify patients with elevated risk of 30‑day readmission so care teams can target follow‑ups, discharge planning, or education. We model this as **binary classification** (readmitted within 30 days vs. not), optimizing for **precision‑recall** and **calibration** rather than raw accuracy.

---

##  Dataset

* **Source**: UCI “Diabetes 130‑US hospitals for years 1999–2008” (aka *diabetic\_data.csv*).
* **Target**: `readmitted` recoded to `1` for "<30" and `0` for "NO"/">30" (configurable).
* **ID columns**: `patient_nbr` (patient identifier), `encounter_id` (visit/encounter).
* **Typical features**: demographics, diagnoses (ICD‑9), procedures, medications, lab/procedure counts, prior utilization.
* **Ethics**: data are de‑identified; still handle sensitive attributes responsibly; avoid proxies and document limitations.

> **Leakage guard**: **Split by patient**, not by individual encounters. Use `patient_nbr` only for splitting, then drop it.

---

##  Project Goals

1. **Reproducible pipeline** from raw CSV → cleaned features → trained model → metrics & plots.
2. **Clinician‑friendly** insights: feature importance, SHAP explanations, error analysis (false positives/negatives).
3. **Production‑minded** evaluation: class imbalance strategies, calibration, drift monitoring, and threshold tuning for different capacities.

---

##  Repository Structure

```
readmission-risk/
├── diabetic_data.csv                                             #main data 
├── IDs_mapping.csv                                               #ID mapping data
├── Predicting Patient Readmission Risk Using Machine Learning.py #main code
├── README.md
└── LICENSE
```

---

##  Quickstart

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt (suggested)**

```
pandas
numpy
scikit-learn
imblearn
matplotlib
shap
mlflow  # optional, for experiment tracking
joblib
pyyaml
```

### 2) Data

Place `diabetic_data.csv` under `data/raw/`.

### 3) Train & Evaluate

```bash
# 1) Prepare data (handles missing/unknown, encodes categoricals, derives features)
python -m src.prepare_data --input data/raw/diabetic_data.csv --output data/interim/processed.parquet

# 2) Train models (logreg baseline + random forest)
python -m src.train --data data/interim/processed.parquet --outdir models/

# 3) Evaluate (PR-AUC, F1-macro, recall; calibration & confusion matrix plots)
python -m src.evaluate --data data/interim/processed.parquet --model models/model.joblib --report reports/

# 4) Explain (SHAP summary & top features)
python -m src.explain --data data/interim/processed.parquet --model models/model.joblib --out reports/figures/
```

> Defaults: patient‑level split (80/20) with stratification; class weights enabled. Override via CLI flags.

---

##  Evaluation & What to Look For

* **Class Imbalance**: majority class is "not readmitted"; use `class_weight='balanced'` or resampling (SMOTE).
* **Primary Metrics**: **PR‑AUC** and **F1‑macro**, plus **Recall (positive/readmit)** at chosen threshold.
* **Calibration**: reliability diagram, Brier score.
* **Confusion Matrix**: inspect errors; correlate with capacity constraints to pick thresholds.
* **Explainability**: SHAP summary & dependence plots; highlight comorbidity/utilization signals.

---

##  Feature Engineering (examples)

* Map ICD‑9 codes to high‑level comorbidity groups (Charlson‑style).
* Prior utilization counts: previous admissions, ER visits, inpatient days.
* Medication & lab counts; change indicators (e.g., med changes during encounter).
* Age binning with clinical input; handle *Unknown/Other* explicitly rather than dropping.

---

##  Responsible Use & Limitations

* **Not medical advice**. For demonstration only. Any clinical deployment requires local validation, governance, and ethics review.
* Dataset is historical US data; patterns may not transfer to other settings.
* Potential **bias** across demographic groups; include fairness checks (e.g., group PR‑AUC/recall).
* Missingness and coding practices vary; avoid over‑interpreting individual features without clinical oversight.

---

##  Example Results (placeholders)

* Logistic Regression (baseline): PR‑AUC \~ 0.21; F1‑macro \~ 0.49 (illustrative).
* Random Forest: PR‑AUC \~ 0.26; F1‑macro \~ 0.53; better recall at similar precision (illustrative).
* Calibrated probabilities improve decision thresholding for given outreach capacity.

> Replace with your actual metrics once you run the pipeline on your machine.


---

##  Acknowledgements

* UCI Machine Learning Repository — Diabetes 130‑US hospi
