"""
model.py  –  Stacked Ensemble Model Architecture (Python 3.14 Compatible)
  Base models  : XGBoost + MLP (DNN) + Random Forest
  Meta-learner : SVC (Support Vector Classifier)

  No TensorFlow dependency – uses scikit-learn MLPClassifier for DNN.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier


# ──────────────────────────────────────────────────────────────
#  1. MLP (Deep Neural Network) Builder
# ──────────────────────────────────────────────────────────────
def build_mlp(hidden_layers=(128, 64, 32), learning_rate=0.001,
              max_iter=300, random_state=42):
    """Build a scikit-learn MLPClassifier (drop-in DNN replacement)."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=1e-4,              # L2 regularisation
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        batch_size=32,
        random_state=random_state,
        verbose=False,
    )


# ──────────────────────────────────────────────────────────────
#  2. XGBoost with GridSearch
# ──────────────────────────────────────────────────────────────
def build_xgboost(X_train, y_train, num_classes=2, cv_folds=5):
    """Train XGBoost with hyper-parameter tuning via GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }
    xgb = XGBClassifier(
        eval_metric='mlogloss' if num_classes > 2 else 'logloss',
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring='accuracy',
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"[model] XGBoost best params: {grid.best_params_}  "
          f"CV acc={grid.best_score_:.4f}")
    return grid.best_estimator_


# ──────────────────────────────────────────────────────────────
#  3. Random Forest with GridSearch
# ──────────────────────────────────────────────────────────────
def build_random_forest(X_train, y_train, cv_folds=5):
    """Train Random Forest with hyper-parameter tuning via GridSearchCV."""
    param_grid = {
        'max_depth': [5, 7, 9],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [15, 17, 19],
        'random_state': [3],
    }
    rf = RandomForestClassifier(n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy',
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"[model] RF best params: {grid.best_params_}  "
          f"CV acc={grid.best_score_:.4f}")
    return grid.best_estimator_


# ──────────────────────────────────────────────────────────────
#  4. Meta-learner SVC
# ──────────────────────────────────────────────────────────────
def build_meta_svc():
    """Create SVC meta-learner for stacking."""
    return SVC(kernel='rbf', probability=True, C=1.0,
               gamma='scale', random_state=42)


# ──────────────────────────────────────────────────────────────
#  5. Stacked Ensemble
# ──────────────────────────────────────────────────────────────
class StackedEnsemble:
    """
    Stacked Ensemble: XGBoost + MLP (DNN) + RandomForest  →  SVC meta-learner.

    Training flow:
    1. Build base estimators (XGBoost, MLP, RF).
    2. Use sklearn StackingClassifier with SVC final estimator.
    3. Evaluate on held-out test set.
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.ensemble = None
        self.individual_models = {}

    def build(self):
        """Construct the StackingClassifier."""
        # Base estimators
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8,
            eval_metric='mlogloss' if self.num_classes > 2 else 'logloss',
            random_state=42, n_jobs=-1,
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam',
            alpha=1e-4, learning_rate_init=0.001,
            max_iter=300, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
            batch_size=32, random_state=42,
        )
        rf = RandomForestClassifier(
            max_depth=9, min_samples_split=2,
            n_estimators=19, random_state=3, n_jobs=-1,
        )

        meta = build_meta_svc()

        estimators = [
            ('xgboost', xgb),
            ('dnn', mlp),
            ('random_forest', rf),
        ]

        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )
        return self.ensemble

    def train(self, X_train, y_train):
        """Build and fit the stacked ensemble."""
        print("\n[model] Building stacked ensemble (XGBoost + DNN + RF → SVC)…")
        self.build()
        self.ensemble.fit(X_train, y_train)

        # Store references to fitted base models for individual predictions
        for name, model in self.ensemble.named_estimators_.items():
            self.individual_models[name] = model

        print("[model] Stacked ensemble training complete ✓")

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble on test data."""
        y_pred = self.predict(X_test)
        avg = 'binary' if self.num_classes == 2 else 'weighted'
        results = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
            'recall':    recall_score(y_test, y_pred, average=avg, zero_division=0),
            'f1':        f1_score(y_test, y_pred, average=avg, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'report':    classification_report(y_test, y_pred, zero_division=0),
        }
        return results

    def individual_predictions(self, X):
        """Get predictions from each base model individually."""
        contributions = {}
        for name, model in self.individual_models.items():
            try:
                prob = model.predict_proba(X)[0]
                contributions[name] = {
                    'prediction': int(np.argmax(prob)),
                    'confidence': round(float(np.max(prob)) * 100, 2),
                    'probabilities': [round(float(p) * 100, 2) for p in prob],
                }
            except Exception:
                pass
        return contributions

    def feature_importance(self, feature_names):
        """Return RF-based feature importance sorted descending."""
        rf = self.individual_models.get('random_forest')
        if rf is None:
            return []
        imp = rf.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        return [{'feature': f, 'importance': round(float(v), 4)} for f, v in pairs]
