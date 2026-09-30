import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.chdir(Path(__file__).resolve().parents[1])


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_baseline.pkl")

    # 2. Make predictions
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    # 3. Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("-------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_metrics.json", "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=4)

    # 4. Error Analysis Export (original appointment rows saved by preprocess.py)
    errors_df = pd.read_csv("data/processed/X_test_raw.csv")
    errors_df["Actual_NoShow"] = y_test
    errors_df["Predicted_NoShow"] = y_pred

    false_negatives = errors_df[(errors_df["Actual_NoShow"] == 1) & (errors_df["Predicted_NoShow"] == 0)]
    false_positives = errors_df[(errors_df["Actual_NoShow"] == 0) & (errors_df["Predicted_NoShow"] == 1)]

    false_negatives.to_csv("outputs/false_negatives.csv", index=False)
    false_positives.to_csv("outputs/false_positives.csv", index=False)

    print(f"False negatives: {len(false_negatives)} | False positives: {len(false_positives)}")
    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
