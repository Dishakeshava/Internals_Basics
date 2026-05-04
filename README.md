# Internals_Basics — MLOps Lab CIE

**USN:** 1BM23AI048  
**Question Paper Code:** mlops-cie-040  
**Experiment Name:** biomotion-injury-risk-score

---

## Scenario

BioMotion provides injury risk assessment for athletes. This project builds an MLOps pipeline to predict `injury_risk_score` from gait analysis data using the following features:

| Feature | Range |
|---|---|
| stride_length_cm | 80–160 |
| ground_contact_ms | 150–350 |
| hip_drop_degrees | 2–15 |
| fatigue_index | 1–10 |

---

## Repository Structure

```
Internals_Basics/
└── MLOps_Lab_CIE/
    ├── data/
    │   ├── training_data.csv
    │   └── new_data.csv
    ├── src/
    │   ├── train.py
    │   ├── tune.py
    │   ├── register_model.py
    │   └── retrain.py
    ├── models/
    │   └── best_model.pkl
    ├── results/
    │   ├── step1_s1.json
    │   ├── step2_s2.json
    │   ├── step3_s6.json
    │   └── step4_s8.json
    ├── requirements.txt
    └── README.md
```

---

## Tasks

### Task 1 — Experiment Tracking & Model Comparison (`train.py`)
- Trains **Lasso** and **RandomForest** on `training_data.csv`
- Logs MAE, RMSE, R², MAPE and tag `priority=high` to MLflow
- Experiment name: `biomotion-injury-risk-score`
- Selects best model by RMSE (lower is better)
- Saves best model to `models/best_model.pkl`
- Output: `results/step1_s1.json`

### Task 2 — Hyperparameter Tuning (`tune.py`)
- Tunes the best model from Task 1 using grid search
- Parameter grid: `n_estimators` [50, 100, 200], `max_depth` [5, 10, None], `min_samples_split` [2, 5]
- 5-fold cross-validation, each trial logged as a nested MLflow run under parent `tuning-biomotion`
- Output: `results/step2_s2.json`

### Task 3 — Model Versioning (`register_model.py`)
- Registers the Task 1 best run in the MLflow Model Registry
- Registered model name: `biomotion-injury-risk-score-predictor`
- Links version to the original Task 1 run ID
- Output: `results/step3_s6.json`

### Task 4 — Retraining Pipeline (`retrain.py`)
- Combines `training_data.csv` + `new_data.csv`
- Retrains the same model type that won Task 1
- Compares retrained vs champion on the same test set
- Promotes if MAE improves by at least 0.5
- Output: `results/step4_s8.json`

---

## Results Summary

| Task | Output file | Key result |
|---|---|---|
| Task 1 | step1_s1.json | Best model: RandomForest (RMSE 8.366) |
| Task 2 | step2_s2.json | Best params: n_estimators=200, max_depth=5, min_samples_split=2 |
| Task 3 | step3_s6.json | Registered version 3, run linked to Task 1 |
| Task 4 | step4_s8.json | Action: promoted (improvement 4.1918 > 0.5) |

---

## How to Run

> Run scripts in order — each builds on the previous.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start MLflow UI (optional, in a separate terminal)
mlflow ui

# 3. Run tasks in sequence
python src/train.py
python src/tune.py
python src/register_model.py
python src/retrain.py
```

Results are saved to the `results/` folder as JSON files.

---

## Requirements

```
pandas
numpy
scikit-learn
mlflow
joblib
```

---

## Notes

- `random_state=42` and `test_size=0.2` used for all train/test splits
- MLflow tracking uses local file store (`mlruns/`)
- Do not modify the provided CSV files in `data/`
