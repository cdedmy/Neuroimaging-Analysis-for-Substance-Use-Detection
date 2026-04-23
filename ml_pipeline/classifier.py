"""
ML classifier for substance use detection from FC matrices.

Trains multiple models and evaluates with k-fold cross-validation.
Appropriate for small datasets (n<50) with high-dimensional features (~20k).
Uses strong regularization and simple models to avoid overfitting.
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, balanced_accuracy_score
)


def build_models(n_features_select=500):
    """
    Build dictionary of models to compare.
    Each model is a pipeline: scaler -> feature selection -> classifier.

    Feature selection is critical: we have ~20k features but maybe 10-20 subjects.
    SelectKBest(f_classif) picks the most discriminative FC edges.
    """
    common_steps = [
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=n_features_select)),
    ]

    models = {
        'LogisticRegression_L2': Pipeline(common_steps + [
            ('clf', LogisticRegression(C=0.1, penalty='l2', max_iter=2000,
                                      solver='lbfgs', random_state=42))
        ]),
        'LogisticRegression_L1': Pipeline(common_steps + [
            ('clf', LogisticRegression(C=0.1, penalty='l1', max_iter=2000,
                                      solver='liblinear', random_state=42))
        ]),
        'SVM_Linear': Pipeline(common_steps + [
            ('clf', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        'SVM_RBF': Pipeline(common_steps + [
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ]),
        'RandomForest': Pipeline(common_steps + [
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=42, n_jobs=-1))
        ]),
    }
    return models


def evaluate_model(model, X, y, n_splits=5, random_state=42):
    """
    Run stratified k-fold cross-validation and return metrics.

    For small datasets, we use Leave-One-Out if possible, otherwise stratified k-fold.
    """
    n_classes = len(np.unique(y))
    min_class_size = min(np.bincount(pd.Categorical(y).codes))

    # Don't exceed min class size for splits
    n_splits = min(n_splits, min_class_size)
    if n_splits < 2:
        raise ValueError(f"Not enough samples: min class has {min_class_size} subjects")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
    }

    # AUC only for binary classification
    if n_classes == 2:
        try:
            y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
            # Use probabilities for positive class
            pos_class = np.unique(y)[1]
            metrics['roc_auc'] = roc_auc_score(y == pos_class, y_proba[:, 1])
        except Exception:
            metrics['roc_auc'] = None

    return metrics, y_pred


def train_all_models(X, y, n_features_select=500, n_splits=5):
    """Train and evaluate all models. Returns summary DataFrame."""
    models = build_models(n_features_select=n_features_select)
    results = []
    predictions = {}

    print(f"\n{'='*60}")
    print(f"Training {len(models)} models on {X.shape[0]} subjects, "
          f"{X.shape[1]} features")
    print(f"Classes: {np.unique(y, return_counts=True)}")
    print(f"Cross-validation: {n_splits}-fold stratified")
    print(f"{'='*60}\n")

    for name, model in models.items():
        print(f"→ {name}...")
        try:
            metrics, y_pred = evaluate_model(model, X, y, n_splits=n_splits)
            predictions[name] = y_pred
            result = {'model': name, **metrics}
            results.append(result)
            print(f"   acc={metrics['accuracy']:.3f}  "
                  f"bal_acc={metrics['balanced_accuracy']:.3f}  "
                  f"f1={metrics['f1_macro']:.3f}", end='')
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                print(f"  auc={metrics['roc_auc']:.3f}")
            else:
                print()
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({'model': name, 'error': str(e)})

    return pd.DataFrame(results), predictions


def get_best_model(X, y, n_features_select=500, n_splits=5):
    """Identify best model by balanced accuracy."""
    df, predictions = train_all_models(X, y, n_features_select, n_splits)
    if 'balanced_accuracy' not in df.columns:
        raise RuntimeError("No models succeeded")

    valid = df.dropna(subset=['balanced_accuracy'])
    best_row = valid.loc[valid['balanced_accuracy'].idxmax()]

    print(f"\n🏆 Best model: {best_row['model']} "
          f"(balanced_acc={best_row['balanced_accuracy']:.3f})")

    # Retrain best model on full dataset for feature importance
    models = build_models(n_features_select=n_features_select)
    best_model = models[best_row['model']]
    best_model.fit(X, y)

    return df, best_model, predictions[best_row['model']]


if __name__ == "__main__":
    # Quick test with synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=30, n_features=500,
                                n_informative=50, n_classes=2,
                                random_state=42)
    results_df, best, preds = get_best_model(X, y, n_features_select=100)
    print("\nResults:")
    print(results_df.to_string(index=False))
