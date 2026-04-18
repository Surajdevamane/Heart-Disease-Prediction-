"""
preprocess.py  –  Complete data preprocessing pipeline (Python 3.14 Compatible)
  1. Load & clean Cleveland Heart Disease data
  2. Label-encode categoricals
  3. Pearson Correlation feature selection (threshold = 0.2)
  4. SMOTE class balancing (applied AFTER train/test split – no leakage)
  5. StandardScaler normalisation
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "heart_cleveland.csv"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Columns that should be treated as categorical for Label Encoding
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

CORRELATION_THRESHOLD = 0.2


# ──────────────────────────────────────────────────────────────
#  1. Load & Clean
# ──────────────────────────────────────────────────────────────
def load_and_clean(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset, remove rows with missing values (paper drops 6 → 297)."""
    df = pd.read_csv(path, na_values="?")

    # Ensure column names match
    if list(df.columns) != COLUMN_NAMES:
        if df.shape[1] == len(COLUMN_NAMES):
            df.columns = COLUMN_NAMES
        else:
            if "num" in df.columns:
                df = df.rename(columns={"num": "target"})

    # Convert everything to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing values (paper: 6 rows removed → 297 used)
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)

    print(f"[preprocess] Loaded {before} rows, removed {removed} with missing values → {len(df)} clean rows")
    return df


# ──────────────────────────────────────────────────────────────
#  2. Encode categorical features
# ──────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply LabelEncoder to categorical columns."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(int))
    return df


# ──────────────────────────────────────────────────────────────
#  3. Feature Selection via Pearson Correlation
# ──────────────────────────────────────────────────────────────
def pearson_feature_selection(df: pd.DataFrame, target_col: str = "target",
                              threshold: float = CORRELATION_THRESHOLD):
    """
    Return list of features with |correlation| >= threshold w.r.t. target.

    Paper Table 13 (Cleveland):
      ca: 0.5177, oldpeak: 0.4985, thal: 0.4846, slope: 0.4055,
      cp: 0.3946, thalach: 0.3549, restecg: 0.3127, exang: 0.2650,
      age: 0.2501, trestbps: 0.2325, sex: 0.2286
      chol: 0.0244 → DROP,  fbs: 0.0234 → DROP
    """
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    correlations = corr.abs().sort_values(ascending=False)

    selected = correlations[correlations >= threshold].index.tolist()
    removed = correlations[correlations < threshold].index.tolist()

    print(f"\n[preprocess] PCC Feature Correlations (threshold={threshold}):")
    for feat in correlations.index:
        val = correlations[feat]
        status = "KEPT" if val >= threshold else "DROPPED"
        print(f"  {feat:>10s}: {val:.4f}  [{status}]")

    print(f"\n[preprocess] Kept {len(selected)} features, dropped {removed}")
    return selected, {feat: round(float(correlations[feat]), 4) for feat in correlations.index}


# ──────────────────────────────────────────────────────────────
#  4. SMOTE Balancing (training data only)
# ──────────────────────────────────────────────────────────────
def apply_smote(X, y):
    """Apply SMOTE oversampling. Only call on training data."""
    print(f"\n[preprocess] Class distribution BEFORE SMOTE:")
    dist_before = pd.Series(y).value_counts().sort_index()
    for cls, count in dist_before.items():
        print(f"  Class {cls}: {count}")

    # Adjust k_neighbors for small classes
    class_counts = pd.Series(y).value_counts()
    min_count = class_counts.min()
    k = min(5, max(1, min_count - 1))

    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)

    print(f"\n[preprocess] Class distribution AFTER SMOTE:")
    dist_after = pd.Series(y_res).value_counts().sort_index()
    for cls, count in dist_after.items():
        print(f"  Class {cls}: {count}")
    print(f"  Total: {len(y_res)} (was {len(y)})")

    return X_res, y_res


# ──────────────────────────────────────────────────────────────
#  Master pipeline
# ──────────────────────────────────────────────────────────────
def run_pipeline(binary: bool = True):
    """
    Full preprocessing pipeline.

    Steps (per paper):
    1. Load & clean (303 → 297)
    2. Label encode categoricals
    3. Binarize target if binary mode
    4. PCC feature selection (threshold=0.2) → drop chol, fbs
    5. 80:20 train/test split (stratified)
    6. SMOTE on training data ONLY (no data leakage)
    7. StandardScaler: fit on train, transform both

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, selected_features, correlation_scores
    """
    df = load_and_clean()
    df = encode_categoricals(df)

    if binary:
        df["target"] = (df["target"] > 0).astype(int)

    selected_features, correlation_scores = pearson_feature_selection(
        df, threshold=CORRELATION_THRESHOLD
    )

    # Ensure we keep at least 5 features regardless of threshold
    if len(selected_features) < 5:
        corr = df.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
        selected_features = corr.head(5).index.tolist()
        print(f"[preprocess] Fallback: using top-5 features: {selected_features}")

    X = df[selected_features]
    y = df["target"]

    # Step 5: Train/test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 6: SMOTE on training data only (prevents data leakage)
    X_train_sm, y_train_sm = apply_smote(X_train.values, y_train.values)

    # Step 7: Scale — fit on training, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test.values)

    print(f"\n[preprocess] Pipeline complete:")
    print(f"  Training samples: {len(X_train_scaled)} (after SMOTE)")
    print(f"  Testing samples:  {len(X_test_scaled)}")
    print(f"  Features used:    {len(selected_features)} → {selected_features}")

    return (X_train_scaled, X_test_scaled, y_train_sm, y_test.values,
            scaler, selected_features, correlation_scores)


if __name__ == "__main__":
    print("=" * 60)
    print("  BINARY MODE")
    print("=" * 60)
    Xtr, Xte, ytr, yte, sc, feats, corrs = run_pipeline(binary=True)
    print(f"Binary — Train: {Xtr.shape}, Test: {Xte.shape}, classes: {np.unique(ytr)}")

    print("\n" + "=" * 60)
    print("  MULTICLASS MODE")
    print("=" * 60)
    Xtr, Xte, ytr, yte, sc, feats, corrs = run_pipeline(binary=False)
    print(f"Multi  — Train: {Xtr.shape}, Test: {Xte.shape}, classes: {np.unique(ytr)}")
