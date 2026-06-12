"""
Stage 3 — Train the LightGBM severity classifier.

Uses TF-IDF text features + structured metadata to predict
inferred severity (4-class). Mismatch is derived post-hoc by
comparing predicted severity to assigned priority.
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix
)
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


SEVERITY_LABELS   = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
METADATA_FEATURES = [
    "channel_encoded",
    "category_encoded",
    "resolution_time_z",
    "satisfaction_norm",
    "template_severity",
]
TEXT_COL    = "combined_text"
TARGET_COL  = "inferred_severity_encoded"


def load_data(pseudo_labels_path: str):
    df = pd.read_csv(pseudo_labels_path)
    print(f"Loaded {len(df)} tickets")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts().sort_index()}")
    return df


def build_features(df_train, df_test):
    """
    Two-head feature construction:
    - Text head  : TF-IDF on combined_text (5000 features)
    - Meta head  : structured metadata features
    Combined via sparse hstack.
    """
    # Text head
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_text_train = tfidf.fit_transform(df_train[TEXT_COL])
    X_text_test  = tfidf.transform(df_test[TEXT_COL])

    # Meta head
    X_meta_train = csr_matrix(df_train[METADATA_FEATURES].values.astype(float))
    X_meta_test  = csr_matrix(df_test[METADATA_FEATURES].values.astype(float))

    # Combined
    X_train = hstack([X_text_train, X_meta_train])
    X_test  = hstack([X_text_test,  X_meta_test])

    return X_train, X_test, tfidf


def compute_class_weights(y):
    """Inverse frequency class weights for imbalance handling."""
    counts = np.bincount(y)
    total  = len(y)
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    return weights


def train_classifier(X_train, y_train):
    class_weights = compute_class_weights(y_train)
    sample_weights = np.array([class_weights[label] for label in y_train])

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        class_weight=class_weights,
        random_state=C.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def derive_mismatch(df_test: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Core decomposed approach:
    Predict severity → compare to assigned priority → derive mismatch.
    No direct mismatch prediction.
    """
    df_test = df_test.copy()
    df_test["predicted_severity_encoded"] = y_pred
    df_test["predicted_severity"]         = df_test["predicted_severity_encoded"].map(SEVERITY_LABELS)

    df_test["severity_delta"] = (
        df_test["predicted_severity_encoded"] - df_test["priority_encoded"]
    )
    df_test["is_mismatch"] = (df_test["severity_delta"] != 0).astype(int)
    df_test["mismatch_type"] = df_test["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )
    return df_test


def evaluate(df_test: pd.DataFrame, y_pred_severity: np.ndarray):
    """
    Evaluate on binary mismatch task (derived from severity prediction).
    Print verification metrics.
    """
    df_result = derive_mismatch(df_test, y_pred_severity)

    y_true_mismatch = df_test["mismatch_label"].values
    y_pred_mismatch = df_result["is_mismatch"].values

    acc      = accuracy_score(y_true_mismatch, y_pred_mismatch)
    macro_f1 = f1_score(y_true_mismatch, y_pred_mismatch, average="macro")
    recalls  = recall_score(y_true_mismatch, y_pred_mismatch, average=None)

    print("\n" + "="*50)
    print("VERIFICATION METRICS")
    print("="*50)
    print(f"Binary Accuracy       : {acc:.4f}  {'PASS' if acc >= 0.83 else 'FAIL'}")
    print(f"Macro F1              : {macro_f1:.4f}  {'PASS' if macro_f1 >= 0.82 else 'FAIL'}")
    print(f"Recall (Consistent)   : {recalls[0]:.4f}  {'PASS' if recalls[0] >= 0.78 else 'FAIL'}")
    print(f"Recall (Mismatch)     : {recalls[1]:.4f}  {'PASS' if recalls[1] >= 0.78 else 'FAIL'}")
    print("="*50)

    print("\nSeverity Classification Report:")
    print(classification_report(
        df_test[TARGET_COL].values,
        y_pred_severity,
        target_names=["Low", "Medium", "High", "Critical"]
    ))

    print("\nMismatch Confusion Matrix:")
    print(confusion_matrix(y_true_mismatch, y_pred_mismatch))

    return acc, macro_f1, recalls, df_result


def save_artifacts(model, tfidf, model_dir: str | None = None):
    model_dir = model_dir or str(C.MODELS_DIR)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "severity_classifier.pkl")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    print(f"\nModel saved → {model_path}")
    print(f"TF-IDF saved → {tfidf_path}")


def run(pseudo_labels_path: str | None = None,
        model_dir: str | None = None,
        test_size: float = 0.2) -> dict:

    pseudo_labels_path = pseudo_labels_path or str(C.PSEUDO_LABELS_CSV)
    model_dir          = model_dir          or str(C.MODELS_DIR)

    print("=" * 60)
    print("STAGE 3 — Model Training")
    print("=" * 60)

    df = load_data(pseudo_labels_path)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=C.SEED,
        stratify=df[TARGET_COL]
    )
    print(f"\nTrain: {len(df_train)} | Test: {len(df_test)}")

    print("\nBuilding features...")
    X_train, X_test, tfidf = build_features(df_train, df_test)

    print("Training LightGBM severity classifier...")
    model = train_classifier(X_train, df_train[TARGET_COL].values)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc, macro_f1, recalls, df_result = evaluate(df_test, y_pred)

    save_artifacts(model, tfidf, model_dir)

    # Save test predictions
    test_pred_path = str(C.TEST_PREDICTIONS_CSV)
    df_result.to_csv(test_pred_path, index=False)
    print(f"Test predictions saved → {test_pred_path}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "recall_consistent": recalls[0],
        "recall_mismatch": recalls[1],
    }


if __name__ == "__main__":
    run()