import pandas as pd
import numpy as np
import mlflow
import json
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "training_data.csv")
results_path = os.path.join(BASE_DIR, "results", "step2_s2.json")

# ---------------- LOAD DATA ----------------
data = pd.read_csv(data_path)
X = data.drop("injury_risk_score", axis=1)
y = data["injury_risk_score"]

# ---------------- PARAM GRID ----------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_rmse = float("inf")
best_mae = float("inf")   # ← FIX: track real MAE
best_params = None
trial_count = 0

# ---------------- MLflow PARENT RUN ----------------
mlflow.set_experiment("biomotion-injury-risk-score")

with mlflow.start_run(run_name="tuning-biomotion"):

    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            for s in param_grid["min_samples_split"]:
                trial_count += 1
                rmse_scores = []
                mae_scores = []   # ← FIX: collect MAE per fold

                with mlflow.start_run(nested=True, run_name=f"trial_{trial_count}"):
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                        model = RandomForestRegressor(
                            n_estimators=n,
                            max_depth=d,
                            min_samples_split=s,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)

                        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
                        mae_scores.append(mean_absolute_error(y_val, preds))  # ← FIX

                    avg_rmse = np.mean(rmse_scores)
                    avg_mae = np.mean(mae_scores)   # ← FIX

                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", d)
                    mlflow.log_param("min_samples_split", s)
                    mlflow.log_metric("rmse", avg_rmse)
                    mlflow.log_metric("mae", avg_mae)   # ← FIX: log MAE too

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_mae = avg_mae   # ← FIX: save corresponding MAE
                        best_params = {
                            "n_estimators": n,
                            "max_depth": d,
                            "min_samples_split": s
                        }

# ---------------- SAVE RESULTS ----------------
output = {
    "search_type": "grid",
    "n_folds": 5,
    "total_trials": trial_count,
    "best_params": best_params,
    "best_mae": round(best_mae, 4),          # ← FIX: real MAE value
    "best_cv_mae": round(best_mae, 4),       # ← FIX: correct key name (was best_cv_rmse)
    "parent_run_name": "tuning-biomotion"
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
with open(results_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nTask 2 completed")
print("Best Params:", best_params)
print("Best RMSE:", round(best_rmse, 4))
print("Best MAE:", round(best_mae, 4))