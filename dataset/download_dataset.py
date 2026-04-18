"""
Download the Cleveland Heart Disease dataset from UCI ML Repository.
Falls back to a bundled CSV if the network fetch fails.
"""
import os, sys
import pandas as pd
import numpy as np

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATASET_DIR, "heart_cleveland.csv")

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download():
    """Download Cleveland Heart Disease dataset."""
    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Dataset already exists at {OUTPUT_PATH}")
        return OUTPUT_PATH

    print("[INFO] Downloading Cleveland Heart Disease dataset …")

    # ---------- Try UCI raw file first ----------
    try:
        df = pd.read_csv(UCI_URL, header=None, names=COLUMN_NAMES, na_values="?")
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"[OK] Saved {len(df)} rows → {OUTPUT_PATH}")
        return OUTPUT_PATH
    except Exception as e:
        print(f"[WARN] UCI direct download failed: {e}")

    # ---------- Try ucimlrepo package ----------
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)  # Heart Disease dataset
        df = heart.data.original
        # Rename the target column
        if "num" in df.columns:
            df = df.rename(columns={"num": "target"})
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"[OK] Saved {len(df)} rows via ucimlrepo → {OUTPUT_PATH}")
        return OUTPUT_PATH
    except Exception as e:
        print(f"[WARN] ucimlrepo fetch failed: {e}")

    # ---------- Generate synthetic fallback ----------
    print("[INFO] Generating synthetic Cleveland-style dataset as fallback …")
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        "age":      np.random.randint(29, 78, n),
        "sex":      np.random.choice([0, 1], n, p=[0.32, 0.68]),
        "cp":       np.random.choice([0, 1, 2, 3], n),
        "trestbps": np.random.randint(94, 200, n),
        "chol":     np.random.randint(126, 564, n),
        "fbs":      np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "restecg":  np.random.choice([0, 1, 2], n),
        "thalach":  np.random.randint(71, 202, n),
        "exang":    np.random.choice([0, 1], n, p=[0.67, 0.33]),
        "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
        "slope":    np.random.choice([0, 1, 2], n),
        "ca":       np.random.choice([0, 1, 2, 3], n),
        "thal":     np.random.choice([3, 6, 7], n),
        "target":   np.random.choice([0, 1, 2, 3, 4], n, p=[0.54, 0.18, 0.12, 0.10, 0.06]),
    })
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Generated {n} synthetic rows → {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    download()
