"""
End-to-end prediction pipeline for the Support Integrity Auditor.

Orchestrates:
  Stage 4 — Inference (predict severity on all tickets)
  Stage 5 — Dossier Generation (evidence for mismatches)

Usage:
    python predict.py
"""
import sys, os

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.model.predict import run as run_predict
from src.dossier.dossier_generator import run as run_dossiers


def main():
    print("\n" + "█" * 60)
    print("  SUPPORT INTEGRITY AUDITOR — PREDICTION PIPELINE")
    print("█" * 60 + "\n")

    # Stage 4: Inference
    df = run_predict()

    # Stage 5: Dossier Generation
    dossiers = run_dossiers()

    print("\n" + "█" * 60)
    print("  PREDICTION COMPLETE")
    print("█" * 60)
    print(f"\n  Total tickets    : {len(df)}")
    print(f"  Mismatches       : {df['is_mismatch'].sum()}")
    print(f"  Mismatch rate    : {df['is_mismatch'].mean():.2%}")
    print(f"  Dossiers created : {len(dossiers)}")
    print()

    return df, dossiers


if __name__ == "__main__":
    main()
