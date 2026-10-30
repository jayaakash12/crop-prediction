"""
train_model.py
---------------
Trains several scikit-learn classifiers on data/Crop_recommendation.csv,
evaluates them on a held-out split + 5-fold CV, picks the best model by
macro F1, and writes everything the Flask backend needs to models/:

  models/model.joblib            -> best fitted pipeline (scaler + classifier)
  models/label_encoder.joblib    -> crop name <-> class index
  models/metrics.json            -> per-model accuracy/precision/recall/F1/macro-F1/CV
  models/feature_importance.json -> feature importances of the best model
  models/meta.json               -> best model name, version, trained_at, dataset info
"""
import json
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "Crop_recommendation.csv")
MODELS_DIR = os.path.join(HERE, "models")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CANDIDATES = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Real dataset not found at {DATA_PATH}. "
            "Place Crop_recommendation.csv in data/ and rerun training."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + ["label"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required_columns].dropna().copy()
    if df.empty:
        raise ValueError("Dataset is empty after removing rows with missing values.")
    X = df[FEATURES].values
    y_raw = df["label"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    fitted_pipelines = {}

    for name, clf in CANDIDATES.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1w = f1_score(y_test, preds, average="weighted", zero_division=0)
        f1m = f1_score(y_test, preds, average="macro", zero_division=0)

        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)

        results[name] = {
            "name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1w), 4),
            "macro_f1": round(float(f1m), 4),
            "cross_validation_f1_macro": round(float(cv_scores.mean()), 4),
        }
        fitted_pipelines[name] = pipe
        print(f"{name:20s} acc={acc:.3f}  macroF1={f1m:.3f}  CV_macroF1={cv_scores.mean():.3f}")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    # refit best model on ALL data for the deployed artifact
    best_pipe.fit(X, y)

    # feature importance (best-effort across model types)
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        # permutation importance fallback
        from sklearn.inspection import permutation_importance
        r = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = r.importances_mean

    importances = np.array(importances, dtype=float)
    importances = importances / (importances.sum() if importances.sum() != 0 else 1)
    feature_importance = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in sorted(zip(FEATURES, importances), key=lambda t: -t[1])
    ]

    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"models": list(results.values()), "best_model": best_name}, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump({"model": best_name, "features": feature_importance}, f, indent=2)

    meta = {
        "model_name": best_name,
        "version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "features": FEATURES,
        "dataset_file": os.path.basename(DATA_PATH),
        "dataset_source": "real Crop Recommendation dataset",
        "dataset_size": int(len(df)),
        "num_classes": int(len(le.classes_)),
        "classes": le.classes_.tolist(),
        **results[best_name],
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved model + metrics to", MODELS_DIR)


if __name__ == "__main__":
    main()
