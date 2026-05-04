import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, "data", "training_data.csv")
new_path = os.path.join(BASE_DIR, "data", "new_data.csv")
step1_path = os.path.join(BASE_DIR, "results", "step1_s1.json")
result_path = os.path.join(BASE_DIR, "results", "step4_s8.json")

# ---------------- READ BEST MODEL TYPE FROM TASK 1 ----------------  ← FIX
if not os.path.exists(step1_path):
    raise FileNotFoundError(
        f"step1_s1.json not found at {step1_path}. Run train.py first."
    )

with open(step1_path, "r") as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]
print(f"Champion model type from Task 1: {best_model_name}")

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(train_path)
new_df = pd.read_csv(new_path)

original_rows = len(train_df)
new_rows = len(new_df)

# ---------------- COMBINE ----------------
combined_df = pd.concat([train_df, new_df], ignore_index=True)
combined_rows = len(combined_df)

# ---------------- SPLIT (SAME TEST SET — from original data only) ----------------
X = train_df.drop("injury_risk_score", axis=1)
y = train_df["injury_risk_score"]

X_train_old, X_test, y_train_old, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- HELPER: build model by name ----------------  ← FIX
def build_model(name):
    if name == "Lasso":
        return Lasso(alpha=0.1)
    elif name == "RandomForest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {name}")

# ---------------- CHAMPION MODEL ----------------
champion_model = build_model(best_model_name)   # ← FIX: dynamic, not hardcoded
champion_model.fit(X_train_old, y_train_old)
champion_preds = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

# ---------------- RETRAIN MODEL ON COMBINED DATA ----------------
X_new = combined_df.drop("injury_risk_score", axis=1)
y_new = combined_df["injury_risk_score"]

retrained_model = build_model(best_model_name)   # ← FIX: same model type
retrained_model.fit(X_new, y_new)
retrained_preds = retrained_model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, retrained_preds)

# ---------------- DECISION ----------------
improvement = champion_mae - retrained_mae
threshold = 0.5

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

# ---------------- SAVE OUTPUT ----------------
output = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_mae": round(champion_mae, 4),
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "mae"
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
with open(result_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nTask 4 completed")
print("Champion MAE:", round(champion_mae, 4))
print("Retrained MAE:", round(retrained_mae, 4))
print("Improvement:", round(improvement, 4))
print("Action:", action)
print("Saved at:", result_path)