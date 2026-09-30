import os
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.chdir(Path(__file__).resolve().parents[1])

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Medical_Appointment_NoShow_Registry"
REGISTERED_MODEL_NAME = "medical_appointment_noshow_model"

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
    ),
}


def run_registry_training():
    print("Starting Model Registry Training...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for name, model in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_family", name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
            mlflow.log_metrics(metrics)

            # Every candidate becomes a new version of the same registered model
            mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
)
            print(f"Registered {name}: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

    print("Model registry training complete!")


if __name__ == "__main__":
    run_registry_training()
