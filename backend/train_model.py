"""
train_model.py
----------------
Trains two multi-class Random Forest classifiers — one for Dyslexia, one for
Dysgraphia — on the real labeled datasets provided
(data/final_dyslexia_dataset_1000.xlsx, data/final_dysgraphia_dataset_1000.xlsx).

Each dataset has 1000 rows, 11 numeric features, and a Risk_Label in
{Normal, Mild, Moderate, Severe} — this is the actual "Multi-Class
Classification" the project title refers to.

Run: python train_model.py
Outputs (per disorder): models/<name>_rf.joblib, models/<name>_scaler.joblib,
models/<name>_features.joblib, models/<name>_label_encoder.joblib
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ordered so the label encoder produces 0=Normal < 1=Mild < 2=Moderate < 3=Severe
# (an ordinal-ish ordering, useful for risk-level display later)
CLASS_ORDER = ["Normal", "Mild", "Moderate", "Severe"]

DATASETS = {
    "dyslexia": {
        "file": "final_dyslexia_dataset_1000.xlsx",
        "features": [
            "Age", "Reading_Time_sec", "Reading_Speed_WPM", "Reading_Accuracy",
            "Word_Error_Count", "Letter_Reversal_Count", "Spelling_Accuracy",
            "Comprehension_Score", "Avg_Response_Time_ms", "Hesitation_Count",
            "Confidence_Score",
        ],
    },
    "dysgraphia": {
        "file": "final_dysgraphia_dataset_1000.xlsx",
        "features": [
            "Age", "Writing_Time_sec", "Writing_Speed_CPS", "Stroke_Count",
            "Pen_Lift_Count", "Avg_Stroke_Speed", "Stroke_Smoothness",
            "Letter_Size_SD", "Word_Spacing_SD", "Baseline_Deviation",
            "Correction_Count",
        ],
    },
}


def train_and_save(name, cfg):
    df = pd.read_excel(os.path.join(DATA_DIR, cfg["file"]))
    feature_cols = cfg["features"]

    # Encode labels with a fixed, known order so class indices are stable
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    df = df[df["Risk_Label"].isin(CLASS_ORDER)]  # guard against stray labels
    y = le.transform(df["Risk_Label"])
    X = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        random_state=7, class_weight="balanced"
    )
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ({len(df)} rows) ===")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_test, preds), index=le.classes_, columns=le.classes_))

    importances = sorted(zip(feature_cols, clf.feature_importances_.round(3)), key=lambda x: -x[1])
    print("Feature importances (ranked):", importances)

    joblib.dump(clf, os.path.join(MODELS_DIR, f"{name}_rf.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{name}_scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{name}_features.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, f"{name}_label_encoder.joblib"))
    return acc


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        train_and_save(name, cfg)
    print("\nModels saved to:", MODELS_DIR)
