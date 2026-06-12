# Support Integrity Auditor (SIA)

**MARS Open Projects 2026 — AI/ML Problem Statement 1**
Semantics-driven, evidence-grounded auditor that detects **Priority Mismatch** in customer support tickets — cases where a ticket's true severity (inferred from its content and metadata) conflicts with its human-assigned priority.

---

## 1. Problem Overview

Manual support-ticket triage suffers from agent fatigue, customer favoritism, and keyword anchoring, which can cause critical issues to be mislabeled as "Low" (**Hidden Crisis**) or trivial issues inflated to "Critical" (**False Alarm**). SIA bootstraps its own supervision signal (no pre-labeled mismatch data exists) and learns to flag these mismatches, producing a structured, hallucination-free **Evidence Dossier** for every flagged ticket.

**Dataset:** [Customer Support Tickets — CRM Dataset](https://www.kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset/data)

---

## 2. Pipeline Architecture

The project is split into a **training pipeline** and an **inference pipeline**, orchestrated by `train_pipeline.py` and `predict.py` respectively.

```
Raw Tickets
   │
   ▼
Stage 0 — Data Preparation        (src/data_prep.py)
   │
   ▼
Stage 1 — Feature Engineering     (src/features/feature_engg.py)
   │
   ├── Rule-based NLP features (keyword density, negation, escalation phrases)
   ├── Resolution-time based severity proxy
   └── Embedding-based semantic urgency features (sentence-transformers)
   │
   ▼
Stage 2a — Template / Rule Scoring (src/pseudo_labeling/template_extractor.py)
   │
   ▼
Stage 2b — LLM Zero-Shot Severity Scoring  (run separately — see Section 5)
   │            (Mistral-7B-Instruct / Phi-3-mini, executed on Google Colab)
   │
   ▼
Stage 2c — Signal Fusion & Pseudo-Label Generation (src/pseudo_labeling/signal_fusion.py)
   │            → combines Stage 2a + Stage 2b signals into a binary
   │              mismatch pseudo-label (Consistent / Mismatched)
   │
   ▼
Stage 3 — Classifier Training      (src/model/train.py)
   │            → fine-tuned / adapter-based model on pseudo-labels
   │              (text + structured metadata, class-imbalance handled)
   │
   ▼
Stage 4 — Inference                (src/model/predict.py)
   │
   ▼
Stage 5 — Evidence Dossier Generation (src/dossier/dossier_generator.py)
   │
   ▼
predict.py  →  predictions + per-ticket Evidence Dossiers
```

---

## 3. ⚠️ Important Note on Stage 2b (LLM Scoring)

Stage 2b (LLM-based zero-shot severity scoring using mistralai/Mistral-7B-Instruct-v0.3) is **computationally heavy** and was **not run as part of the local `train_pipeline.py` execution**, since it requires a GPU not available on the development laptop.

Instead, Stage 2b was executed **separately on Google Colab** (GPU runtime). The notebook used for this step is provided in:

```
notebooks/
```

The **output of Stage 2b** (per-ticket LLM severity scores) is exported as a CSV and placed at:

```
<path-to-csv-in-repo>      # e.g. data/processed/llm_severity_scores.csv
```

`src/pseudo_labeling/signal_fusion.py` (Stage 2c) reads this CSV directly and fuses it with the Stage 2a rule-based signal to produce the final pseudo-labels — so `train_pipeline.py` can be re-run end-to-end **as long as this CSV is present**, without needing a GPU.

> **To reproduce Stage 2b from scratch:** open the notebook in `notebooks/` on Google Colab, run it on the prepared ticket dataset, and export the resulting CSV to the path above before running `train_pipeline.py`.

---

## 4. Ablation: Signal Fusion Justification

| Signal | Source | Individual Contribution (Pseudo-label Agreement / Accuracy) |
|---|---|---|
| Rule-based NLP (keyword density, negation, escalation phrases) | Stage 2a | *fill in from experiment results* |
| LLM zero-shot severity (Mistral-7B-Instruct) | Stage 2b (Colab) | *fill in from experiment results* |
| Fused signal (Stages 2a + 2b) | Stage 2c | *fill in from experiment results* |

---

## 5. Repository Structure

```
.
├── .venv/                      # Local Python virtual environment (Add to .gitignore)
├── app/
│   └── app.py                  # Main application entry point (e.g., Streamlit, FastAPI, or Flask)
├── data/
│   ├── processed/              # Cleaned, transformed, and intermediate datasets
│   │   ├── llm_scores.csv
│   │   ├── processed.csv
│   │   ├── pseudo_labels.csv
│   │   ├── template_scores.csv
│   │   ├── test_predictions.csv
│   │   └── tickets_with_features.csv
│   └── raw/                    # Immutable, original source data
│       ├── customer_support_tickets.csv
│       └── enhanced_customer_support_data.csv
├── models/                     # Serialized machine learning artifacts
│   ├── severity_classifier.pkl # Trained classification model
│   └── tfidf_vectorizer.pkl    # Text vectorization model
├── notebooks/                  # Jupyter notebooks for exploration and pipeline prototyping
│   └── SIA_Pipeline.ipynb
├── outputs/                    # Final generated artifacts from the pipeline
│   ├── dossiers.json
│   └── predictions.csv
└── src/                        # Core source code modules
    ├── dossier/                # Logic for report/dossier generation
    │   ├── __init__.py
    │   └── dossier_generator.py
    ├── features/               # Feature engineering and extraction scripts
    │   ├── __init__.py
    │   └── feature_engg.py
    ├── model/                  # Core model training and inference pipelines
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    ├── pseudo_labeling/        # Weak supervision and LLM-assisted labeling logic
    │   ├── __init__.py
    │   ├── llm_scorer.ipynb
    │   ├── signal_fusion.py
    │   └── template_extractor.py
    ├── __init__.py
    ├── config.py               # Global configuration and environment settings
    └── data_prep.py            # Initial data loading and preprocessing routines
```

---

## 6. Setup & Usage

### Installation
```bash
git clone https://github.com/QyAkhil/Support-Integrity-Auditor--SIA-.git
cd Support-Integrity-Auditor--SIA-
pip install -r requirements.txt
```

### Training
```bash
python train_pipeline.py
```
This runs Stages 0, 1, 2a, 2c, and 3. **Requires the Stage 2b LLM-score CSV (see Section 3) to be present** for Stage 2c to run correctly.

### Inference
```bash
python predict.py
```
Runs Stages 4–5: produces severity predictions and Evidence Dossiers for flagged tickets.

### Streamlit App
```bash
streamlit run app.py
```
- Accepts single-ticket form input or batch CSV upload
- Returns binary mismatch judgment + full Evidence Dossier
- Displays Priority Mismatch Dashboard (distribution, mismatch types, top signals)
- Severity-delta heatmap across ticket categories/channels

**Hosted demo:** *https://my8nvylwymy39gkcgjyrapp.streamlit.app/

---

## 7. Evidence Dossier Schema

For every ticket classified as a mismatch, the system outputs:

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm",
  "severity_delta": "",
  "feature_evidence": [
    { "signal": "keyword", "value": "...", "weight": "..." },
    { "signal": "resolution_time", "value": "...", "interpretation": "..." }
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>",
  "confidence": ""
}
```

**Hard Rule:** Every `feature_evidence` item is traceable to a specific field in the input ticket — no fabricated/unverifiable claims.

---

## 8. Evaluation Results

| Metric | Threshold | Result | Status |
|---|---|---|---|
| Binary Classification Accuracy | ≥ 83% | **91.63%** | ✅ Pass |
| Macro F1 Score | ≥ 0.82 | **0.9151** | ✅ Pass |
| Per-Class Recall (Consistent) | ≥ 0.78 | **0.9008** | ✅ Pass |
| Per-Class Recall (Mismatched) | ≥ 0.78 | **0.9285** | ✅ Pass |

### Pipeline Run Summary

| | |
|---|---|
| Total tickets processed | 20,000 |
| Mismatches detected | 11,098 (55.49%) |
| Hidden Crises | 5,564 |
| False Alarms | 5,534 |
| Dossiers generated | 11,098 |

**Output files:**
- Predictions → `outputs/predictions.csv`
- Dossiers → `outputs/dossiers.json`

All three verification thresholds (Accuracy, Macro F1, Per-Class Recall on both classes) are met.

---

## 9. Demo Video

🔗 *https://drive.google.com/drive/folders/1Mu9pYWhoZAenGWyifw4Y_3H9DMjPJyPh*

---

## 10. Disclaimer

This project was built for the MARS Open Projects 2026 evaluation. All implementation logic is original; open-source libraries and pretrained models (sentence-transformers, Mistral-7B-Instruct ) were used as permitted.
