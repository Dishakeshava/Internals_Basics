import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "training_data.csv")
new_path = os.path.join(BASE_DIR, "data", "new_data.csv")
result_path = os.path.join(BASE_DIR, "results", "step4_s8.json")

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(train_path)
new_df = pd.read_csv(new_path)

# counts
original_rows = len(train_df)
new_rows = len(new_df)

# ---------------- COMBINE ----------------
combined_df = pd.concat([train_df, new_df], ignore_index=True)
combined_rows = len(combined_df)

# ---------------- SPLIT (SAME TEST SET) ----------------
X = train_df.drop("injury_risk_score", axis=1)
y = train_df["injury_risk_score"]

X_train_old, X_test, y_train_old, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- CHAMPION MODEL ----------------
champion_model = RandomForestRegressor(n_estimators=100, random_state=42)
champion_model.fit(X_train_old, y_train_old)

champion_preds = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

# ---------------- RETRAIN MODEL ----------------
X_new = combined_df.drop("injury_risk_score", axis=1)
y_new = combined_df["injury_risk_score"]

retrained_model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "mae"
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

with open(result_path, "w") as f:
    json.dump(output, f, indent=4)

print("\n✅ Task 4 completed")
print("Action:", action)
print("Saved at:", result_path)