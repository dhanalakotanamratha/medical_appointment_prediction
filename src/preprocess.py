import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Always run relative to the project root, no matter where the script is launched from
os.chdir(Path(__file__).resolve().parents[1])

RAW_PATH = "data/raw/KaggleV2-May-2016.csv"
TARGET = "no_show"

NUMERICAL_FEATURES = [
    "gender", "age", "scholarship", "hipertension", "diabetes", "alcoholism",
    "handcap", "sms_received", "appointment_year", "appointment_month",
    "appointment_day", "appointment_dayofweek", "waiting_days",
]
CATEGORICAL_FEATURES = ["neighbourhood"]
DROP_COLUMNS = ["patientid", "appointmentid", "scheduledday", "appointmentday"]


def clean_data(df):
    """Cleaning steps taken from the notebook (sections 4 - 19)."""
    # Clean column names
    df.columns = (
        df.columns.str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")
    )

    # Duplicates and age validation
    df = df.drop_duplicates()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"] >= 0].copy()

    # Date features
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], errors="coerce")
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], errors="coerce")
    df["appointment_year"] = df["appointmentday"].dt.year
    df["appointment_month"] = df["appointmentday"].dt.month
    df["appointment_day"] = df["appointmentday"].dt.day
    df["appointment_dayofweek"] = df["appointmentday"].dt.dayofweek
    df["waiting_days"] = (
        df["appointmentday"].dt.normalize() - df["scheduledday"].dt.normalize()
    ).dt.days

    # Invalid values
    df = df.drop_duplicates()
    df = df[(df["age"] >= 0) & (df["age"] <= 100)]
    df = df[df["waiting_days"] >= 0]

    # Encode target and gender
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    df["gender"] = df["gender"].map({"F": 0, "M": 1})
    df = df.dropna(subset=[TARGET, "gender"]).copy()
    df[TARGET] = df[TARGET].astype(int)

    # Remove columns that are not features
    return df.drop(columns=DROP_COLUMNS)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def array_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")

    # 2. Data Cleaning
    df = clean_data(df)
    print(f"Clean shape: {df.shape}")

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 3. Train-Test Split (before fitting anything, to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale & Encode
    preprocessor = build_preprocessor()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    X_train_final = X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final
    X_test_final = X_test_final.toarray() if hasattr(X_test_final, "toarray") else X_test_final

    # 5. Save Artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)  # used for error analysis

    joblib.dump(preprocessor, "preprocess_pipeline.pkl")

    # 6. Save Metadata
    metadata = {
        "dataset_name": "Medical Appointment No-Show",
        "target": TARGET,
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "no_show_rate_train": round(float(y_train.mean()), 4),
        "no_show_rate_test": round(float(y_test.mean()), 4),
        "sha256": {
            "X_train_final": array_hash(X_train_final),
            "X_test_final": array_hash(X_test_final),
            "y_train": array_hash(y_train.to_numpy(dtype=np.int64)),
            "y_test": array_hash(y_test.to_numpy(dtype=np.int64)),
        },
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
