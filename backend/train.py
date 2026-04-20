"""
train.py  –  Training Orchestrator (Python 3.14 Compatible)
  1. Download dataset (or use a custom one)
  2. Run preprocessing pipeline (binary + multiclass)
  3. Train stacked ensemble models (XGBoost + DNN + RF → SVC)
  4. Evaluate & print metrics
  5. Save models, scalers, and metadata to backend/saved_model/

Usage:
  # Train with default Cleveland dataset
  python -m backend.train

  # Train with a custom dataset
  python -m backend.train --dataset "path/to/your/dataset.csv"

  # Train only binary or multiclass
  python -m backend.train --mode binary
  python -m backend.train --mode multi

  # Custom PCC threshold
  python -m backend.train --threshold 0.15
"""

import os, sys, json, argparse
import numpy as np
import joblib

# Ensure the project root is on the path for sibling imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.download_dataset import download
from backend.preprocess import run_pipeline
from backend.model import StackedEnsemble

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")


def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)


def train_and_save(mode: str = "binary", dataset_path: str = None,
                   pcc_threshold: float = 0.2):
    """Train one stacked ensemble (binary or multiclass) and persist it."""
    is_binary = mode == "binary"
    num_classes = 2 if is_binary else 5
    print(f"\n{'='*60}")
    print(f"  Training {mode.upper()} model  (classes={num_classes})")
    if dataset_path:
        print(f"  Dataset: {dataset_path}")
    print(f"  PCC Threshold: {pcc_threshold}")
    print(f"{'='*60}")

    # Run full preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, features, corr_scores = \
        run_pipeline(binary=is_binary, dataset_path=dataset_path,
                     pcc_threshold=pcc_threshold)

    # Build and train stacked ensemble
    ensemble = StackedEnsemble(num_classes=num_classes)
    ensemble.train(X_train, y_train)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    print(f"\n[train] {mode} evaluation:")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  Confusion :\n{np.array(results['confusion_matrix'])}")

    # Feature importance
    feat_imp = ensemble.feature_importance(features)
    print(f"\n[train] Feature importance:")
    for fi in feat_imp:
        print(f"  {fi['feature']:>10s}  {fi['importance']:.4f}")

    # ── Persist ──
    prefix = "binary" if is_binary else "multi"
    ensure_dirs()

    # Save the entire stacked ensemble
    joblib.dump(ensemble, os.path.join(SAVE_DIR, f"{prefix}_ensemble.pkl"))

    # Save scaler
    joblib.dump(scaler, os.path.join(SAVE_DIR, f"{prefix}_scaler.pkl"))

    # Save metadata
    meta = {
        "mode": mode,
        "features": features,
        "num_classes": num_classes,
        "input_dim": len(features),
        "correlation_scores": corr_scores,
        "metrics": {k: v for k, v in results.items() if k != "report"},
        "feature_importance": feat_imp,
        "dataset": dataset_path or "Cleveland (default)",
        "pcc_threshold": pcc_threshold,
    }
    with open(os.path.join(SAVE_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] {mode} model saved to {SAVE_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Heart Disease Prediction Models"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to custom CSV dataset. Must have columns: "
             "age, sex, cp, trestbps, chol, fbs, restecg, thalach, "
             "exang, oldpeak, slope, ca, thal, target"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["binary", "multi", "both"],
        help="Training mode: 'binary', 'multi', or 'both' (default: both)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="PCC correlation threshold for feature selection (default: 0.2)"
    )
    args = parser.parse_args()

    # Step 1 – download default dataset if no custom one provided
    if args.dataset is None:
        download()

    # Step 2 – train
    results = {}
    if args.mode in ("binary", "both"):
        results["binary"] = train_and_save(
            "binary", dataset_path=args.dataset,
            pcc_threshold=args.threshold
        )

    if args.mode in ("multi", "both"):
        results["multi"] = train_and_save(
            "multi", dataset_path=args.dataset,
            pcc_threshold=args.threshold
        )

    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE")
    if "binary" in results:
        print(f"  Binary accuracy  : {results['binary']['accuracy']:.4f}")
    if "multi" in results:
        print(f"  Multi  accuracy  : {results['multi']['accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
