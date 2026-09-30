import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

REQUIRED_FILES = [
    "data/processed/X_train_final.npy",
    "data/processed/X_test_final.npy",
    "data/processed/y_train.npy",
    "data/processed/y_test.npy",
    "data/processed/dataset_metadata.json",
    "preprocess_pipeline.pkl",
    "models/random_forest_model.pkl",
    "artifacts/confusion_matrix.png",
    "artifacts/roc_curve.png",
    "mlflow.db",
]


def run_validation():
    print("Starting Output Validation...")
    errors = []

    for path in REQUIRED_FILES:
        if not os.path.exists(path):
            errors.append(f"Missing output: {path}")

    if not errors:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
        with open("data/processed/dataset_metadata.json") as f:
            meta = json.load(f)

        if list(X_train.shape) != meta["train_shape"] or list(X_test.shape) != meta["test_shape"]:
            errors.append("Array shapes do not match dataset_metadata.json")
        if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
            errors.append("Feature / label row counts differ")
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            errors.append("NaN values found in processed features")
        if not set(np.unique(y_train)) <= {0, 1} or not set(np.unique(y_test)) <= {0, 1}:
            errors.append("Target labels are not binary")

        model = joblib.load("models/random_forest_model.pkl")
        preds = model.predict(X_test[:100])
        if len(preds) != min(100, len(X_test)):
            errors.append("Model prediction length mismatch")

    report = {"errors": errors, "status": "FAILED" if errors else "PASSED"}
    os.makedirs("logs", exist_ok=True)
    with open("logs/output_validation.json", "w") as f:
        json.dump(report, f, indent=4)

    for e in errors:
        print(f"[ERROR] {e}")
    print(f"Output validation {report['status']}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
