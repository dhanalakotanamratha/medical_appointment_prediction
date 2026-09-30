import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "medical_appointment_noshow_model"
CHAMPION_ALIAS = "champion"

# Promotion gates: a version must pass all of them to be eligible
MIN_ROC_AUC = 0.65
MIN_RECALL = 0.40


def run_lifecycle():
    print("Starting Model Lifecycle Automation...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"[ERROR] No versions found for '{REGISTERED_MODEL_NAME}'. Run train_registry.py first.")
        sys.exit(1)

    eligible = []
    for mv in versions:
        metrics = client.get_run(mv.run_id).data.metrics
        passed = metrics.get("roc_auc", 0) >= MIN_ROC_AUC and metrics.get("recall", 0) >= MIN_RECALL
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, mv.version, "validation_status", "passed" if passed else "failed"
        )
        print(
            f"Version {mv.version}: F1={metrics.get('f1_score', 0):.4f} "
            f"Recall={metrics.get('recall', 0):.4f} ROC-AUC={metrics.get('roc_auc', 0):.4f} -> "
            f"{'PASSED' if passed else 'FAILED'} gates"
        )
        if passed:
            eligible.append((metrics.get("f1_score", 0), int(mv.version)))

    if not eligible:
        print("[WARN] No version passed the promotion gates. Champion alias unchanged.")
        return

    best_f1, best_version = max(eligible)
    for mv in versions:
        status = "champion" if int(mv.version) == best_version else "archived"
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "lifecycle_status", status)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(best_version))
    print(f"Version {best_version} promoted to alias '{CHAMPION_ALIAS}' (F1 = {best_f1:.4f})")


if __name__ == "__main__":
    run_lifecycle()
