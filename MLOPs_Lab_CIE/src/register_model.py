import mlflow
import mlflow.sklearn
import json
import os

# ---------------- CONFIG ----------------
MODEL_NAME = "biomotion-injury-risk-score-predictor"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP1_RESULT = os.path.join(BASE_DIR, "results", "step1_s1.json")
RESULT_PATH = os.path.join(BASE_DIR, "results", "step3_s6.json")

# ---------------- READ TASK 1 RESULTS ----------------  ← FIX: use Task 1 run, not a new one
if not os.path.exists(STEP1_RESULT):
    raise FileNotFoundError(
        f"step1_s1.json not found at {STEP1_RESULT}. Run train.py first."
    )

with open(STEP1_RESULT, "r") as f:
    step1 = json.load(f)

best_run_id = step1["best_model_run_id"]         # ← the winning run from Task 1
best_metric_value = step1["best_metric_value"]   # ← real RMSE, not 0.0

print(f"Linking to Task 1 best run: {best_run_id}")
print(f"Best RMSE from Task 1: {best_metric_value}")

# ---------------- MLflow: register the Task 1 model ----------------  ← FIX: no retraining
mlflow.set_experiment("biomotion-injury-risk-score")

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

version = registered_model.version
print(f"Registered as version: {version}")

# ---------------- SAVE JSON ----------------
output = {
    "registered_model_name": MODEL_NAME,
    "version": int(version),
    "run_id": best_run_id,
    "source_metric": "rmse",
    "source_metric_value": round(best_metric_value, 4)   # ← FIX: real value
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print("\nTask 3 completed")
print("Version:", version)
print("Saved at:", RESULT_PATH)