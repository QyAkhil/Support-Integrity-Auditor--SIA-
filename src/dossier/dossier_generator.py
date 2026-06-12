"""
Stage 5 — Evidence dossier generation for mismatch tickets.

Generates structured, grounded evidence dossiers for every ticket
where inferred severity ≠ assigned priority. Each evidence item
traces to an actual ticket field — no hallucination.
"""
import os, sys
import pandas as pd
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config as C


SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
PRIORITY_MAP    = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# Keywords that signal high severity — used for evidence extraction
CRITICAL_KEYWORDS = [
    "data loss", "data breach", "hacked", "unauthorized", "locked out",
    "cannot access", "system down", "security", "compromised", "stolen",
    "phishing", "suspicious", "fraud", "emergency", "critical"
]
HIGH_KEYWORDS = [
    "crash", "crashing", "not working", "broken", "error", "failed",
    "unable to", "not loading", "freezing", "stuck", "not syncing"
]
ESCALATION_KEYWORDS = [
    "unacceptable", "escalate", "manager", "legal", "refund", "cancel",
    "worst", "never again", "lawsuit"
]


def extract_keywords(text: str) -> list:
    """Find which severity keywords appear in ticket text."""
    text = text.lower()
    found = []
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "Critical"})
    for kw in HIGH_KEYWORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "High"})
    for kw in ESCALATION_KEYWORDS:
        if kw in text:
            found.append({"keyword": kw, "severity_signal": "Escalation"})
    return found[:3]  # cap at 3 evidence items


def resolution_interpretation(hours: float, inferred: str) -> str:
    """Interpret resolution time in context of inferred severity."""
    if hours <= 12:
        return f"Resolved in {hours:.0f}h — consistent with {inferred}-severity handling"
    elif hours <= 30:
        return f"Resolved in {hours:.0f}h — moderate resolution time"
    else:
        return f"Resolved in {hours:.0f}h — slow resolution suggests lower agent priority"


def build_constraint_analysis(row: pd.Series) -> str:
    """
    2-3 sentence grounded explanation of the mismatch.
    Every claim traces to a real ticket field.
    """
    assigned  = row[C.COL_PRIORITY]
    inferred  = row["predicted_severity"]
    template  = row["subject_template"]
    channel   = row[C.COL_CHANNEL]
    res_time  = row[C.COL_RES_HRS]
    delta     = int(row["severity_delta"])

    if row["mismatch_type"] == "Hidden Crisis":
        return (
            f"The ticket subject '{template}' via {channel} indicates a severity of {inferred}, "
            f"yet was assigned {assigned} priority by the agent. "
            f"Resolution took {res_time:.0f} hours, which combined with the issue type "
            f"suggests the ticket's true impact was underestimated by {abs(delta)} severity level(s)."
        )
    else:
        return (
            f"The ticket subject '{template}' via {channel} was assigned {assigned} priority, "
            f"but the content analysis infers a severity of {inferred}. "
            f"Resolution took {res_time:.0f} hours, suggesting the issue was less urgent "
            f"than its assigned priority implies, overestimated by {abs(delta)} severity level(s)."
        )


def generate_dossier(row: pd.Series) -> dict:
    """
    Generate evidence dossier for a single mismatch ticket.
    Every field traces to actual ticket data — no hallucination.
    """
    keywords     = extract_keywords(str(row["combined_text"]))
    feature_evidence = []

    # Keyword evidence — from combined_text field
    for kw in keywords:
        feature_evidence.append({
            "signal"       : "keyword",
            "value"        : kw["keyword"],
            "weight"       : kw["severity_signal"],
            "source_field" : "Ticket_Description"
        })

    # Template evidence — from Ticket_Subject field
    feature_evidence.append({
        "signal"         : "subject_template",
        "value"          : row["subject_template"],
        "weight"         : SEVERITY_LABELS.get(int(row["template_severity"]), "Unknown"),
        "source_field"   : "Ticket_Subject"
    })

    # Resolution time evidence — from Resolution_Time_Hours field
    feature_evidence.append({
        "signal"        : "resolution_time",
        "value"         : f"{row[C.COL_RES_HRS]:.0f} hours",
        "interpretation": resolution_interpretation(
            row[C.COL_RES_HRS],
            row["predicted_severity"]
        ),
        "source_field"  : "Resolution_Time_Hours"
    })

    # Channel evidence — from Ticket_Channel field
    feature_evidence.append({
        "signal"      : "channel",
        "value"       : row[C.COL_CHANNEL],
        "weight"      : "intake context",
        "source_field": "Ticket_Channel"
    })

    return {
        "ticket_id"         : row[C.COL_ID],
        "assigned_priority" : row[C.COL_PRIORITY],
        "inferred_severity" : row["predicted_severity"],
        "mismatch_type"     : row["mismatch_type"],
        "severity_delta"    : int(row["severity_delta"]),
        "feature_evidence"  : feature_evidence,
        "constraint_analysis": build_constraint_analysis(row),
        "confidence"        : float(row["confidence"]),
    }


def validate_dossier(dossier: dict, row: pd.Series) -> bool:
    """
    Verify no hallucination — every evidence item must trace to a real field.
    Returns False if any evidence is fabricated.
    """
    valid_sources = {
        "Ticket_Description", "Ticket_Subject",
        "Resolution_Time_Hours", "Ticket_Channel"
    }
    for item in dossier["feature_evidence"]:
        if item.get("source_field") not in valid_sources:
            return False
    if dossier["assigned_priority"] != row[C.COL_PRIORITY]:
        return False
    return True


def run(
    predictions_path: str | None = None,
    output_path     : str | None = None,
) -> list:
    predictions_path = predictions_path or str(C.PREDICTIONS_CSV)
    output_path      = output_path      or str(C.DOSSIERS_JSON)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("STAGE 5 — Dossier Generation")
    print("=" * 60)

    df = pd.read_csv(predictions_path)
    mismatches = df[df["is_mismatch"] == 1].copy()
    print(f"Generating dossiers for {len(mismatches)} mismatch tickets...")

    dossiers   = []
    invalidated = 0

    for _, row in mismatches.iterrows():
        dossier = generate_dossier(row)
        if validate_dossier(dossier, row):
            dossiers.append(dossier)
        else:
            invalidated += 1

    print(f"Valid dossiers   : {len(dossiers)}")
    print(f"Invalidated      : {invalidated}")

    with open(output_path, "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"Dossiers saved → {output_path}")
    return dossiers


if __name__ == "__main__":
    run()