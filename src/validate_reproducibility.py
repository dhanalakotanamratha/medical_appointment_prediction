import os
import sys
import json
import numpy as np
import mlflow
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

os.chdir(Path(__file__).resolve().parents[1])

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Medical_Appointment_NoShow"
RUN_NAME = "RandomForest"
PARAMS = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}
TOLERANCE = 1e-9


def run_validation():
    print("Starting Reproducibility Validation...")
    errors = []

    # 1. The processed arrays must match the hashes recorded during preprocessing
    from preprocess import array_hash
    with open("data/processed/dataset_metadata.json") as f:
        meta = json.load(f)
    for name, expected in meta["sha256"].items():
        actual = array_hash(np.load(f"data/processed/{name}.npy"))
        if actual != expected:
            errors.append(f"{name}.npy changed since preprocessing")

    # 2. Retrain with identical parameters and compare with the latest tracked run
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = RandomForestClassifier(**PARAMS).fit(X_train, y_train)
    new_metrics = {
        "f1_score": f1_score(y_test, model.predict(X_test), zero_division=0),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        errors.append(f"No tracked run named '{RUN_NAME}' found. Run train_mlflow.py first.")
    else:
        for metric, value in new_metrics.items():
            logged = float(runs.iloc[0][f"metrics.{metric}"])
            print(f"{metric}: logged={logged:.6f} retrained={value:.6f}")
            if abs(logged - value) > TOLERANCE:
                errors.append(f"{metric} differs: logged={logged} retrained={value}")

    report = {"errors": errors, "status": "FAILED" if errors else "PASSED"}
    os.makedirs("logs", exist_ok=True)
    with open("logs/reproducibility_validation.json", "w") as f:
        json.dump(report, f, indent=4)

    for e in errors:
        print(f"[ERROR] {e}")
    print(f"Reproducibility validation {report['status']}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
