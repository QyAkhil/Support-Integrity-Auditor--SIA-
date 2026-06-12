"""
Stage 1 — Feature engineering for the Support Integrity Auditor.

Responsibilities:
1. Clean and combine text fields.
2. Encode categorical features (priority, channel, category).
3. Extract email domain tier proxy.
4. Normalize resolution time and satisfaction score.
"""
import os, sys
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
CHANNEL_MAP  = {"Web Form": 0, "Chat": 1, "Email": 2, "Phone": 3, "Social Media": 4}


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip subject + description, combine into one field."""
    df["clean_subject"] = df[C.COL_SUBJECT].fillna("").str.lower().str.strip()
    df["clean_desc"]    = df[C.COL_DESC].fillna("").str.lower().str.strip()
    df["combined_text"] = df["clean_subject"] + " " + df["clean_desc"]
    return df


def encode_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Map Priority_Level string → integer."""
    df["priority_encoded"] = df[C.COL_PRIORITY].map(PRIORITY_MAP)
    return df


def encode_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Map Ticket_Channel string → integer."""
    df["channel_encoded"] = df[C.COL_CHANNEL].map(CHANNEL_MAP).fillna(0).astype(int)
    return df


def encode_category(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode Issue_Category."""
    df["category_encoded"] = df[C.COL_CATEGORY].astype("category").cat.codes
    return df


def extract_email_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract domain suffix from email as a rough customer tier proxy.
    .com → 0, .org → 1, .net → 2, other → 3
    (All are example.* here, but keeps the feature for real data too)
    """
    domain_map = {"example.com": 0, "example.org": 1, "example.net": 2}
    df["email_domain"] = df[C.COL_EMAIL].str.split("@").str[-1].str.lower()
    df["domain_tier"]  = df["email_domain"].map(domain_map).fillna(3).astype(int)
    df.drop(columns=["email_domain"], inplace=True)
    return df


def normalize_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize Resolution_Time_Hours. Used as severity proxy in pseudo-labeling."""
    df["resolution_time_z"] = zscore(df[C.COL_RES_HRS].fillna(df[C.COL_RES_HRS].median()))
    return df


def normalize_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Satisfaction_Score to 0-1. Low score = more severe in practice."""
    min_s = df[C.COL_SAT].min()
    max_s = df[C.COL_SAT].max()
    df["satisfaction_norm"] = (df[C.COL_SAT] - min_s) / (max_s - min_s + 1e-8)
    return df


def run(input_path: str | None = None, output_path: str | None = None) -> pd.DataFrame:
    input_path  = input_path  or str(C.ENHANCED_CSV)
    output_path = output_path or str(C.FEATURES_CSV)

    print("=" * 60)
    print("STAGE 1 — Feature Engineering")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    df = clean_text(df)
    df = encode_priority(df)
    df = encode_channel(df)
    df = encode_category(df)
    df = extract_email_domain(df)
    df = normalize_resolution_time(df)
    df = normalize_satisfaction(df)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows → {output_path}")
    return df


if __name__ == "__main__":
    run()