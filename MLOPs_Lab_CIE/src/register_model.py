import mlflow
import mlflow.sklearn
import pandas as pd
import json
import os

from sklearn.ensemble import RandomForestRegressor

# ---------------- CONFIG ----------------
MODEL_NAME = "biomotion-injury-risk-score-predictor"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULT_PATH = os.path.join(BASE_DIR, "results", "step3_s6.json")

# ---------------- LOAD DATA ----------------
data = pd.read_csv(data_path)

X = data.drop("injury_risk_score", axis=1)
y = data["injury_risk_score"]

# ---------------- TRAIN MODEL ----------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- MLflow ----------------
mlflow.set_experiment("biomotion-injury-risk-score")

with mlflow.start_run(run_name="Final-Registry-Run") as run:

    # 🔥 IMPORTANT: log under "model"
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = run.info.run_id
    print("Run ID:", run_id)

    # ---------------- REGISTER MODEL ----------------
    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    version = registered_model.version

# ---------------- SAVE JSON ----------------
output = {
    "registered_model_name": MODEL_NAME,
    "version": version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": 0.0
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("\n✅ Task 3 completed")
print("Version:", version)
print("Saved at:", RESULT_PATH)