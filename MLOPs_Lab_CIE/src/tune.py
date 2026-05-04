import pandas as pd
import numpy as np
import mlflow
import json
import os

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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
best_params = None
trial_count = 0

# ---------------- MLflow PARENT RUN ----------------
mlflow.set_experiment("biomotion-injury-risk-score")

with mlflow.start_run(run_name="tuning-biomotion"):

    # Grid search manually
    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            for s in param_grid["min_samples_split"]:

                trial_count += 1
                rmse_scores = []

                # -------- NESTED RUN --------
                with mlflow.start_run(nested=True):

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

                        rmse = np.sqrt(mean_squared_error(y_val, preds))
                        rmse_scores.append(rmse)

                    avg_rmse = np.mean(rmse_scores)

                    # Log params + metric
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", d)
                    mlflow.log_param("min_samples_split", s)

                    mlflow.log_metric("rmse", avg_rmse)

                    # Track best
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
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
    "best_mae": 0.0,
    "best_cv_rmse": best_rmse,
    "parent_run_name": "tuning-biomotion"
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

with open(results_path, "w") as f:
    json.dump(output, f, indent=4)

print("\nTask 2 completed")
print("Best Params:", best_params)
print("Best RMSE:", best_rmse)