import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- SAFE PATH HANDLING ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "training_data.csv")
results_path = os.path.join(BASE_DIR, "results", "step1_s1.json")
models_dir = os.path.join(BASE_DIR, "models")

print("Reading data from:", data_path)

# ---------------- LOAD DATA ----------------
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

data = pd.read_csv(data_path)

# ---------------- FEATURES & TARGET ----------------
if "injury_risk_score" not in data.columns:
    raise ValueError("Column 'injury_risk_score' not found in dataset")

X = data.drop("injury_risk_score", axis=1)
y = data["injury_risk_score"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------- METRICS FUNCTION ----------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Avoid division by zero in MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return mae, rmse, r2, mape


# ---------------- MLflow SETUP ----------------
mlflow.set_experiment("biomotion-injury-risk-score")

results = []

# ---------------- LASSO ----------------
with mlflow.start_run(run_name="Lasso"):
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    preds = lasso.predict(X_test)
    mae, rmse, r2, mape = calculate_metrics(y_test, preds)

    mlflow.log_param("model", "Lasso")
    mlflow.log_param("alpha", 0.1)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    mlflow.set_tag("priority", "high")

    mlflow.sklearn.log_model(lasso, "model")

    results.append({
        "name": "Lasso",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    })


# ---------------- RANDOM FOREST ----------------
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae, rmse, r2, mape = calculate_metrics(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    mlflow.set_tag("priority", "high")

    mlflow.sklearn.log_model(rf, "model")

    results.append({
        "name": "RandomForest",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    })


# ---------------- SELECT BEST MODEL ----------------
best_model = min(results, key=lambda x: x["rmse"])


# ---------------- SAVE OUTPUT ----------------
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

output = {
    "experiment_name": "biomotion-injury-risk-score",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

with open(results_path, "w") as f:
    json.dump(output, f, indent=4)


print("\nTask 1 completed")
print("Best model:", best_model["name"])
print("Results saved at:", results_path)