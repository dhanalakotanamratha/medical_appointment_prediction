import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "medical_appointment_noshow_model"


def generate_report():
    print("Generating Model Registry Report...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = sorted(
        client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"), key=lambda v: int(v.version)
    )
    rows = []
    for mv in versions:
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        rows.append(
            {
                "version": int(mv.version),
                "model_family": run.data.params.get("model_family", ""),
                "aliases": list(mv.aliases),
                "lifecycle_status": mv.tags.get("lifecycle_status", "unassigned"),
                "validation_status": mv.tags.get("validation_status", "unassigned"),
                "accuracy": round(m.get("accuracy", 0), 4),
                "precision": round(m.get("precision", 0), 4),
                "recall": round(m.get("recall", 0), 4),
                "f1_score": round(m.get("f1_score", 0), 4),
                "roc_auc": round(m.get("roc_auc", 0), 4),
            }
        )

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/registry_report.json", "w") as f:
        json.dump({"registered_model": REGISTERED_MODEL_NAME, "versions": rows}, f, indent=4)

    lines = [
        f"# Model Registry Report: {REGISTERED_MODEL_NAME}",
        "",
        "| Version | Model | Aliases | Lifecycle | Validation | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['version']} | {r['model_family']} | {', '.join(r['aliases']) or '-'} | "
            f"{r['lifecycle_status']} | {r['validation_status']} | {r['accuracy']} | "
            f"{r['precision']} | {r['recall']} | {r['f1_score']} | {r['roc_auc']} |"
        )
    with open("outputs/registry_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Registry report saved to outputs/registry_report.md and outputs/registry_report.json")


if __name__ == "__main__":
    generate_report()
