"""
Central configuration for the Support Integrity Auditor (SIA).

Every constant here is grounded in the dataset profiling pass.
Keeping all design decisions in one place makes the pipeline reproducible
and the ablation/threshold choices auditable.
"""
from __future__ import annotations
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

RAW_CSV = RAW_DIR / "customer_support_tickets.csv"
ENHANCED_CSV = RAW_DIR / "enhanced_customer_support_data.csv"

# Processed data paths
PROCESSED_CSV = PROC_DIR / "processed.csv"
FEATURES_CSV = PROC_DIR / "tickets_with_features.csv"
TEMPLATE_SCORES_CSV = PROC_DIR / "template_scores.csv"
LLM_SCORES_CSV = PROC_DIR / "llm_scores.csv"
PSEUDO_LABELS_CSV = PROC_DIR / "pseudo_labels.csv"
TEST_PREDICTIONS_CSV = PROC_DIR / "test_predictions.csv"

# Output paths
PREDICTIONS_CSV = OUTPUTS_DIR / "predictions.csv"
DOSSIERS_JSON = OUTPUTS_DIR / "dossiers.json"

# Model artifact paths
CLASSIFIER_PKL = MODELS_DIR / "severity_classifier.pkl"
TFIDF_PKL = MODELS_DIR / "tfidf_vectorizer.pkl"

for _d in (RAW_DIR, PROC_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Schema — actual column names in the CSV
# ──────────────────────────────────────────────────────────────────────────────
COL_ID       = "Ticket_ID"
COL_NAME     = "Customer_Name"
COL_EMAIL    = "Customer_Email"
COL_SUBJECT  = "Ticket_Subject"
COL_DESC     = "Ticket_Description"
COL_CATEGORY = "Issue_Category"
COL_PRIORITY = "Priority_Level"
COL_CHANNEL  = "Ticket_Channel"
COL_DATE     = "Submission_Date"
COL_RES_HRS  = "Resolution_Time_Hours"
COL_AGENT    = "Assigned_Agent"
COL_SAT      = "Satisfaction_Score"

# ──────────────────────────────────────────────────────────────────────────────
# Priority → continuous severity axis [0, 1]
# Anchors are hand-chosen to spread 4 ordinal levels across [0,1].
# ──────────────────────────────────────────────────────────────────────────────
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_ORD = {lvl: i for i, lvl in enumerate(PRIORITY_LEVELS)}
PRIORITY_TO_SCORE = {"Low": 0.12, "Medium": 0.40, "High": 0.70, "Critical": 0.92}

def score_to_priority(score: float) -> str:
    """Map a continuous inferred-severity score back to an ordinal label."""
    if score < 0.26:  return "Low"
    if score < 0.55:  return "Medium"
    if score < 0.81:  return "High"
    return "Critical"

# ──────────────────────────────────────────────────────────────────────────────
# Category severity prior — from the Priority × Category crosstab.
#
# Fraud: only High/Critical in the data → very strong high-severity prior
# Technical: spans all levels but includes Critical → moderate-high
# Billing: skews Low/Medium → moderate
# Account: skews Low/Medium → moderate-low
# General Inquiry: almost all Low/Medium → low
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_SEVERITY_PRIOR = {
    "Fraud":           0.88,
    "Technical":       0.55,
    "Billing":         0.38,
    "Account":         0.35,
    "General Inquiry": 0.22,
}

# ──────────────────────────────────────────────────────────────────────────────
# Customer tier proxy via email domain
# ──────────────────────────────────────────────────────────────────────────────
BUSINESS_DOMAINS = {"enterprise.org", "company.com", "tech.io"}
CONSUMER_DOMAINS = {"example.com", "example.org", "example.net"}

def domain_tier(email: str) -> str:
    dom = email.split("@")[-1].strip().lower() if "@" in str(email) else ""
    return "business" if dom in BUSINESS_DOMAINS else "consumer"

# ──────────────────────────────────────────────────────────────────────────────
# Channels (only 3 exist in the data, balanced ~6,650 each)
# ──────────────────────────────────────────────────────────────────────────────
CHANNELS = ["Chat", "Email", "Web Form"]

# ──────────────────────────────────────────────────────────────────────────────
# Resolution time → severity proxy boundaries (from profiling)
# Critical ≈ 12h, High ≈ 25h, Medium ≈ 44h, Low ≈ 45h
# ──────────────────────────────────────────────────────────────────────────────
RT_BOUNDARIES = [0, 18, 35, 60, 200]   # → Critical / High / Medium / Low
RT_SCORES     = [0.90, 0.65, 0.40, 0.15]

# ──────────────────────────────────────────────────────────────────────────────
# Signal fusion weights — text + embedding = 85%, resolution = 15% (corroborator)
# ──────────────────────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "text_rule":       0.45,
    "embedding":       0.40,
    "resolution_time": 0.15,
}

# ──────────────────────────────────────────────────────────────────────────────
# Mismatch thresholds
# ──────────────────────────────────────────────────────────────────────────────
MISMATCH_TAU = 0.30                     # default; auto-calibrated on train
TARGET_POSITIVE_RATE = (0.15, 0.25)     # target mismatch proportion

# ──────────────────────────────────────────────────────────────────────────────
# Keyword dictionaries for rule-based NLP severity signal
# ──────────────────────────────────────────────────────────────────────────────

# Critical severity — financial loss, security breach, data corruption
CRITICAL_TERMS = [
    "data loss", "data corruption", "data breach", "security breach",
    "account hacked", "hacked", "stolen card", "stolen",
    "unauthorized access", "unauthorized", "phishing",
    "identity theft", "fraud", "fraudulent",
    "system down", "outage", "service down",
    "charged twice", "double charged", "overcharged",
    "crash", "crashing", "app crash",
]

# High severity — functional failures, significant user impact
HIGH_TERMS = [
    "not working", "doesn't work", "does not work",
    "cannot access", "can't access", "unable to access",
    "login failed", "login failure", "authentication failed",
    "error", "bug", "broken",
    "payment failed", "payment issue", "payment problem",
    "sync issue", "not syncing", "data not syncing",
    "screen freezes", "freezing", "frozen",
    "api error", "500 error", "server error",
    "suspicious", "suspicious activity", "suspicious charge",
    "unrecognized", "unrecognized login",
]

# Urgency modifiers — weak weight (adversarial robustness: stuffing these
# should NOT inflate severity of a trivial request)
URGENCY_TERMS = [
    "urgent", "asap", "immediately", "emergency", "critical",
    "right away", "as soon as possible",
]

# Trivial / routine — pull severity DOWN
TRIVIAL_TERMS = [
    "feature request", "demo request", "question",
    "how to", "how do i", "information",
    "hours of operation", "office location", "office hours",
    "pricing", "pricing tiers", "subscription",
    "upgrade", "downgrade",
    "profile update", "change email", "update email",
    "password reset", "reset password",
    "refund status", "refund update",
    "general inquiry", "just wondering", "curious",
]

# Negation cues — "cannot" / "not working" nudge severity up slightly
NEGATIONS = [
    "not", "cannot", "can't", "unable", "failed", "failure",
    "doesn't", "does not", "won't", "will not",
]

# ──────────────────────────────────────────────────────────────────────────────
# Embedding model for semantic severity signal
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

# Anchor phrases for cosine-similarity based severity estimation
SEVERE_ANCHORS = [
    "Critical system failure causing data loss and service outage",
    "Account has been hacked and unauthorized transactions detected",
    "Payment processing error resulting in double charges",
    "Application crashing repeatedly with data corruption",
    "Security breach with stolen credentials and fraud detected",
]

TRIVIAL_ANCHORS = [
    "General question about product features and pricing",
    "Request for a demo or information about office hours",
    "How to update profile settings and change email address",
    "Inquiry about subscription upgrade options",
    "Password reset request for routine account maintenance",
]

# ──────────────────────────────────────────────────────────────────────────────
# Classifier hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_EPOCHS = 3
TRAIN_BATCH = 16
EVAL_BATCH = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# Verification thresholds from the problem statement
VERIFY_ACCURACY = 0.83
VERIFY_F1       = 0.82
VERIFY_RECALL   = 0.78
