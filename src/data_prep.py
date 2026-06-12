"""
Stage 0 — Data preparation for the Support Integrity Auditor.

Responsibilities:
1. Load the raw CRM ticket CSV.
2. Extract the *real issue sentence* from each description (the dataset pads
   a genuine leading sentence with random faker words).
3. Build the text the classifier will see + structured metadata features.
4. Create a FIXED, stratified train/test split that is independent of any
   pseudo-label, so labels can never leak into the split.
"""
from __future__ import annotations
import os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────────────────────────────────────────
_GREETING = re.compile(r"^\s*(hi|hello|hey|dear)\b[^,.:;!?]*[,:]?\s*", re.IGNORECASE)
_FIRST_SENT = re.compile(r"(.+?[.?!])(?:\s|$)")


def extract_leading_sentence(desc: str) -> str:
    """Strip the greeting and return the first sentence (the genuine issue).

    The trailing faker words form a second 'sentence'; we deliberately drop
    them because raw keyword counts over the filler are misleading and are
    exactly the surface an adversarial ticket would attack.
    """
    t = str(desc).strip()
    t = _GREETING.sub("", t, count=1)
    m = _FIRST_SENT.search(t)
    lead = (m.group(1) if m else t).strip()
    return lead or t


def extract_subject_template(subject: str) -> str:
    """Extract the template prefix before ' - '."""
    return subject.split(" - ")[0].strip() if " - " in subject else subject.strip()


def build_model_text(row: pd.Series) -> str:
    """Text fed to the classifier: assigned priority + structured tags + issue.

    Including the *assigned priority* is intentional and correct — the task is
    to decide whether the assigned priority is consistent with the ticket's
    content, so the model must see both.
    """
    parts = [
        f"priority: {row[C.COL_PRIORITY]}",
        f"category: {row[C.COL_CATEGORY]}",
        f"channel: {row[C.COL_CHANNEL]}",
        f"tier: {row['customer_tier']}",
        f"issue: {row['lead_sentence']}",
    ]
    return " | ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run(raw_csv: str | None = None, output_path: str | None = None) -> pd.DataFrame:
    raw_csv = raw_csv or str(C.RAW_CSV)
    output_path = output_path or str(C.PROCESSED_CSV)

    print("=" * 60)
    print("STAGE 0 — Data Preparation")
    print("=" * 60)

    df = pd.read_csv(raw_csv)
    print(f"Loaded {len(df)} tickets from {raw_csv}")

    # ── Extract text features ──
    df["lead_sentence"] = df[C.COL_DESC].apply(extract_leading_sentence)
    df["subject_template"] = df[C.COL_SUBJECT].apply(extract_subject_template)

    # ── Structured metadata ──
    df["priority_score"] = df[C.COL_PRIORITY].map(C.PRIORITY_TO_SCORE)
    df["priority_ord"] = df[C.COL_PRIORITY].map(C.PRIORITY_ORD)
    df["customer_tier"] = df[C.COL_EMAIL].apply(C.domain_tier)
    df["category_prior"] = df[C.COL_CATEGORY].map(C.CATEGORY_SEVERITY_PRIOR)

    # ── Build classifier input text ──
    df["model_text"] = df.apply(build_model_text, axis=1)

    # ── Fixed stratified split ──
    # Stratify by priority to ensure balanced representation
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.15, random_state=C.SEED,
        stratify=df[C.COL_PRIORITY],
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Test:  {(df['split'] == 'test').sum()}")
    print(f"  Priority distribution (train):")
    print(df[df["split"] == "train"][C.COL_PRIORITY].value_counts().to_string())

    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved → {output_path}")
    return df


if __name__ == "__main__":
    run()
