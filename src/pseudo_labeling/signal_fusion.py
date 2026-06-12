"""
Stage 2c — Signal fusion and pseudo-label generation.

Merges template scores, LLM scores, and resolution time into a single
inferred severity score, then generates mismatch labels.
"""
import os, sys
import pandas as pd
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


PRIORITY_MAP    = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

# Signal weights
WEIGHTS = {"template": 0.50, "llm": 0.35, "resolution": 0.15}


def compute_resolution_score(df: pd.DataFrame) -> pd.DataFrame:
    """Invert resolution time, rank-normalize to 0-1."""
    inverted = 1.0 / (df[C.COL_RES_HRS] + 1e-8)
    df["resolution_score"] = np.round(rankdata(inverted) / len(inverted), 4)
    return df


def load_and_merge(
    features_path       : str,
    template_scores_path: str,
    llm_scores_path     : str,
) -> pd.DataFrame:
    df       = pd.read_csv(features_path)
    template = pd.read_csv(template_scores_path)[[C.COL_ID, "template_score", "template_severity", "subject_template"]]
    llm      = pd.read_csv(llm_scores_path)[[C.COL_ID, "llm_score"]]

    df = df.merge(template, on=C.COL_ID, how="left")
    df = df.merge(llm,      on=C.COL_ID, how="left")

    print(f"Total tickets     : {len(df)}")
    print(f"With LLM score    : {df['llm_score'].notna().sum()}")
    print(f"Without LLM score : {df['llm_score'].isna().sum()}")
    return df


def fuse_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted fusion → continuous inferred_score (0-1).
    Tickets without LLM score: redistribute LLM weight to template.
    """
    has_llm  = df["llm_score"].notna()
    w_t, w_l, w_r = WEIGHTS["template"], WEIGHTS["llm"], WEIGHTS["resolution"]

    # With LLM
    df.loc[has_llm, "inferred_score"] = np.round(
        w_t * df.loc[has_llm, "template_score"] +
        w_l * df.loc[has_llm, "llm_score"] +
        w_r * df.loc[has_llm, "resolution_score"],
        4
    )

    # Without LLM — redistribute LLM weight to template
    df.loc[~has_llm, "inferred_score"] = np.round(
        (w_t + w_l) * df.loc[~has_llm, "template_score"] +
        w_r         * df.loc[~has_llm, "resolution_score"],
        4
    )
    return df


def assign_severity_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert inferred_score → 4-class severity label.
    Thresholds tuned to match original priority distribution (~38/38/17/7).
    """
    q1 = df["inferred_score"].quantile(0.38)
    q2 = df["inferred_score"].quantile(0.76)
    q3 = df["inferred_score"].quantile(0.93)

    print(f"Severity thresholds: Low<{q1:.3f} | Medium<{q2:.3f} | High<{q3:.3f} | Critical above")

    def to_severity(s):
        if s < q1:   return 0
        elif s < q2: return 1
        elif s < q3: return 2
        else:        return 3

    df["inferred_severity_encoded"] = df["inferred_score"].apply(to_severity)
    df["inferred_severity"]         = df["inferred_severity_encoded"].map(SEVERITY_LABELS)
    return df


def generate_mismatch_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary mismatch by comparing inferred vs assigned severity.
    This is the training signal for the severity classifier indirectly —
    but the classifier will be trained on inferred_severity_encoded directly.
    """
    df["severity_delta"] = df["inferred_severity_encoded"] - df["priority_encoded"]
    df["mismatch_label"] = (df["severity_delta"] != 0).astype(int)
    df["mismatch_type"]  = df["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )
    return df


def ablation_report(df: pd.DataFrame):
    t_bin = pd.cut(df["template_score"],   bins=4, labels=[0,1,2,3]).astype(int)
    r_bin = pd.cut(df["resolution_score"], bins=4, labels=[0,1,2,3]).astype(int)
    print(f"\nAblation — Template vs Resolution agreement : {(t_bin == r_bin).mean():.4f}")

    llm_sub = df[df["llm_score"].notna()].copy()
    l_bin   = pd.cut(llm_sub["llm_score"],      bins=4, labels=[0,1,2,3]).astype(int)
    t_sub   = pd.cut(llm_sub["template_score"], bins=4, labels=[0,1,2,3]).astype(int)
    print(f"Ablation — Template vs LLM agreement       : {(t_sub == l_bin).mean():.4f}")

    print(f"\nInferred severity distribution:")
    print(df["inferred_severity"].value_counts())
    print(f"\nMismatch label distribution:")
    print(df["mismatch_label"].value_counts())
    print(f"\nMismatch type distribution:")
    print(df["mismatch_type"].value_counts())


def run(
    features_path        : str | None = None,
    template_scores_path : str | None = None,
    llm_scores_path      : str | None = None,
    output_path          : str | None = None,
) -> pd.DataFrame:
    features_path        = features_path        or str(C.FEATURES_CSV)
    template_scores_path = template_scores_path or str(C.TEMPLATE_SCORES_CSV)
    llm_scores_path      = llm_scores_path      or str(C.LLM_SCORES_CSV)
    output_path          = output_path          or str(C.PSEUDO_LABELS_CSV)

    print("=" * 60)
    print("STAGE 2c — Signal Fusion & Pseudo Labels")
    print("=" * 60)

    df = load_and_merge(features_path, template_scores_path, llm_scores_path)
    df = compute_resolution_score(df)
    df = fuse_signals(df)
    df = assign_severity_label(df)
    df = generate_mismatch_labels(df)
    ablation_report(df)

    df.to_csv(output_path, index=False)
    print(f"\nPseudo labels saved → {output_path}")
    return df


if __name__ == "__main__":
    run()