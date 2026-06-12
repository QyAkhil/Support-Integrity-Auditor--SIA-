"""
Stage 4 — Run inference on all tickets.

Loads the trained LightGBM classifier and TF-IDF vectorizer,
predicts severity on all tickets, and derives mismatch labels.
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


SEVERITY_LABELS   = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
PRIORITY_MAP      = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
METADATA_FEATURES = [
    "channel_encoded",
    "category_encoded",
    "resolution_time_z",
    "satisfaction_norm",
    "template_severity",
]


def load_artifacts(model_dir: str | None = None):
    model_dir = model_dir or str(C.MODELS_DIR)
    model = joblib.load(os.path.join(model_dir, "severity_classifier.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    return model, tfidf


def build_features(df: pd.DataFrame, tfidf):
    X_text = tfidf.transform(df["combined_text"])
    X_meta = csr_matrix(df[METADATA_FEATURES].values.astype(float))
    return hstack([X_text, X_meta])


def predict(df: pd.DataFrame, model, tfidf) -> pd.DataFrame:
    """
    Predict severity → derive mismatch by comparing to assigned priority.
    Returns df with prediction columns added.
    """
    X = build_features(df, tfidf)

    df = df.copy()
    df["predicted_severity_encoded"] = model.predict(X)
    df["predicted_severity"]         = df["predicted_severity_encoded"].map(SEVERITY_LABELS)
    df["confidence"]                 = model.predict_proba(X).max(axis=1).round(4)

    # Derive mismatch from severity comparison
    df["severity_delta"] = df["predicted_severity_encoded"] - df["priority_encoded"]
    df["is_mismatch"]    = (df["severity_delta"] != 0).astype(int)
    df["mismatch_type"]  = df["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )
    return df


def run(
    input_path          : str | None = None,
    template_scores_path: str | None = None,
    output_path         : str | None = None,
    model_dir           : str | None = None,
) -> pd.DataFrame:

    input_path           = input_path           or str(C.FEATURES_CSV)
    template_scores_path = template_scores_path or str(C.TEMPLATE_SCORES_CSV)
    output_path          = output_path          or str(C.PREDICTIONS_CSV)
    model_dir            = model_dir            or str(C.MODELS_DIR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("STAGE 4 — Inference")
    print("=" * 60)

    model, tfidf = load_artifacts(model_dir)
    df = pd.read_csv(input_path)

    # Merge template_severity
    template = pd.read_csv(template_scores_path)[[C.COL_ID, "template_severity", "subject_template"]]
    df = df.merge(template, on=C.COL_ID, how="left")

    # Ensure priority_encoded exists
    if "priority_encoded" not in df.columns:
        df["priority_encoded"] = df[C.COL_PRIORITY].map(PRIORITY_MAP)

    print(f"Running inference on {len(df)} tickets...")
    df = predict(df, model, tfidf)

    print(f"\nMismatch summary:")
    print(df["mismatch_type"].value_counts())
    print(f"\nMismatch rate: {df['is_mismatch'].mean():.2%}")

    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved → {output_path}")
    return df


if __name__ == "__main__":
    run()