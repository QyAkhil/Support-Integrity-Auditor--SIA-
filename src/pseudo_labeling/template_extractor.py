"""
Stage 2a — Template-based severity scoring.

Maps each ticket's subject template to an inferred severity level
based on domain knowledge of issue types.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


# ── Template → inferred severity mapping ──────────────────────────────────────
# Derived from data: dominant priority + logical severity adjustment
# Key insight: technical failure templates (App crashing, API Error 500 etc.)
# are assigned Medium by agents but logically warrant High severity
# This deliberate upward adjustment is what generates Hidden Crisis labels

TEMPLATE_SEVERITY_MAP = {
    # Critical — security/fraud, zero ambiguity
    "Account hacked"       : 3,
    "Alert notification"   : 3,
    "Suspicious activity"  : 3,
    "Stolen card"          : 3,
    "Phishing attempt"     : 3,
    "Unrecognized login"   : 3,

    # High — technical failures with real operational impact
    "App crashing"         : 2,
    "API Error 500"        : 2,
    "Login failed"         : 2,

    # Medium — some urgency but not critical
    "Screen freezes"       : 1,
    "Data not syncing"     : 1,
    "Installation issue"   : 1,
    "Charged twice"        : 1,
    "Invoice discrepancy"  : 1,
    "2FA issues"           : 1,

    # Low — informational, billing queries, no urgency
    "Demo request"         : 0,
    "Feature request"      : 0,
    "Hours of operation"   : 0,
    "Office location"      : 0,
    "Password reset"       : 0,
    "Pricing tiers"        : 0,
    "Product question"     : 0,
    "Profile update"       : 0,
    "Refund status"        : 0,
    "Subscription upgrade" : 0,
    "Update credit card"   : 0,
    "Change email"         : 0,
    "Delete account"       : 0,
    "Payment failed"       : 0,
    "Suspicious charge"    : 0,
}

SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def extract_template(subject: str) -> str:
    """Extract prefix before ' - ' as the template key."""
    return subject.split(" - ")[0].strip() if " - " in subject else subject.strip()


def compute_template_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map each ticket's subject template to inferred severity.
    Returns normalized 0-1 score and raw severity int.
    Unknown templates default to Medium (1).
    """
    df["subject_template"] = df[C.COL_SUBJECT].apply(extract_template)

    df["template_severity"] = df["subject_template"].map(TEMPLATE_SEVERITY_MAP).fillna(1).astype(int)

    # Normalize to 0-1
    df["template_score"] = df["template_severity"] / 3.0

    # Coverage report
    known   = df["subject_template"].isin(TEMPLATE_SEVERITY_MAP).sum()
    unknown = len(df) - known
    print(f"Template coverage: {known}/{len(df)} tickets ({unknown} unknown → defaulted to Medium)")
    print("\nTemplate severity distribution:")
    print(df["template_severity"].value_counts().sort_index())

    return df


def run(input_path: str | None = None, output_path: str | None = None) -> pd.DataFrame:
    input_path  = input_path  or str(C.FEATURES_CSV)
    output_path = output_path or str(C.TEMPLATE_SCORES_CSV)

    print("=" * 60)
    print("STAGE 2a — Template Scoring")
    print("=" * 60)

    df = pd.read_csv(input_path)
    df = compute_template_score(df)

    df[[C.COL_ID, "subject_template", "template_severity", "template_score"]].to_csv(
        output_path, index=False
    )
    print(f"\nTemplate scores saved → {output_path}")
    return df


if __name__ == "__main__":
    run()