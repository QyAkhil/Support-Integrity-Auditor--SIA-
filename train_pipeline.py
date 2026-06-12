"""
End-to-end training pipeline for the Support Integrity Auditor.

Orchestrates all stages:
  Stage 0 — Data Preparation
  Stage 1 — Feature Engineering
  Stage 2a — Template Scoring
  Stage 2c — Signal Fusion & Pseudo Labels
  Stage 3 — Model Training

Usage:
    python train_pipeline.py
"""
import sys, os

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_prep import run as run_data_prep
from src.features.feature_engg import run as run_features
from src.pseudo_labeling.template_extractor import run as run_templates
from src.pseudo_labeling.signal_fusion import run as run_fusion
from src.model.train import run as run_train


def main():
    print("\n" + "█" * 60)
    print("  SUPPORT INTEGRITY AUDITOR — TRAINING PIPELINE")
    print("█" * 60 + "\n")

    # Stage 0: Data Preparation
    run_data_prep()

    # Stage 1: Feature Engineering
    run_features()

    # Stage 2a: Template-based severity scoring
    run_templates()

    # Stage 2c: Signal fusion + pseudo-label generation
    # (Stage 2b LLM scoring is done separately via notebook)
    run_fusion()

    # Stage 3: Train the severity classifier
    metrics = run_train()

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60)
    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Recall(C) : {metrics['recall_consistent']:.4f}")
    print(f"  Recall(M) : {metrics['recall_mismatch']:.4f}")
    print()

    return metrics


if __name__ == "__main__":
    main()
