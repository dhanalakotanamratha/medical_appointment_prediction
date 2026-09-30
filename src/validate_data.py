import os
import sys
import json
import pandas as pd
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

RAW_PATH = "data/raw/KaggleV2-May-2016.csv"
REQUIRED_COLUMNS = [
    "patientid", "appointmentid", "gender", "scheduledday", "appointmentday",
    "age", "neighbourhood", "scholarship", "hipertension", "diabetes",
    "alcoholism", "handcap", "sms_received", "no_show",
]
BINARY_COLUMNS = ["scholarship", "hipertension", "diabetes", "alcoholism", "sms_received"]


def run_validation():
    print("Starting Raw Data Validation...")
    errors, warnings = [], []

    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] {RAW_PATH} not found. Copy the Kaggle file into data/raw/ first.")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")

    # Schema
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    else:
        if len(df) < 1000:
            errors.append(f"Only {len(df)} rows found (expected at least 1000)")
        if df[REQUIRED_COLUMNS].isnull().any().any():
            errors.append("Null values found in required columns")
        if not set(df["no_show"].unique()) <= {"Yes", "No"}:
            errors.append(f"Unexpected no_show values: {df['no_show'].unique().tolist()}")
        if not set(df["gender"].unique()) <= {"F", "M"}:
            errors.append(f"Unexpected gender values: {df['gender'].unique().tolist()}")
        for col in BINARY_COLUMNS:
            if not set(df[col].unique()) <= {0, 1}:
                errors.append(f"{col} is not binary: {df[col].unique().tolist()}")

        # Issues that preprocess.py fixes automatically -> warnings only
        age = pd.to_numeric(df["age"], errors="coerce")
        bad_age = int(((age < 0) | (age > 100) | age.isna()).sum())
        if bad_age:
            warnings.append(f"{bad_age} rows with invalid age (removed in preprocessing)")
        sched = pd.to_datetime(df["scheduledday"], errors="coerce")
        appt = pd.to_datetime(df["appointmentday"], errors="coerce")
        if sched.isna().any() or appt.isna().any():
            errors.append("Unparseable scheduledday / appointmentday values")
        else:
            neg = int(((appt.dt.normalize() - sched.dt.normalize()).dt.days < 0).sum())
            if neg:
                warnings.append(f"{neg} rows where appointment is before scheduling (removed in preprocessing)")
        dups = int(df.duplicated().sum())
        if dups:
            warnings.append(f"{dups} duplicate rows (removed in preprocessing)")

    report = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "errors": errors,
        "warnings": warnings,
        "status": "FAILED" if errors else "PASSED",
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/data_validation.json", "w") as f:
        json.dump(report, f, indent=4)

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print(f"Data validation {report['status']} ({len(df)} rows)")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
