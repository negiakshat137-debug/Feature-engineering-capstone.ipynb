"""
evaluation.py
Reusable model evaluation, feature importance, and selection utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, label: str = '') -> dict:
    """Return accuracy, ROC-AUC, and F1 for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        'label':    label,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'roc_auc':  round(roc_auc_score(y_test, y_prob), 4),
        'f1':       round(f1_score(y_test, y_pred), 4),
    }
    print(f'[{label}] Accuracy={results["accuracy"]} | ROC-AUC={results["roc_auc"]} | F1={results["f1"]}')
    return results


def cross_validate(pipeline, X, y, n_splits: int = 5, scoring: str = 'roc_auc') -> np.ndarray:
    """Run stratified k-fold CV and print mean ± std."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    print(f'CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


def plot_confusion_matrix(model, X_test, y_test, labels=None, save_path: str = None):
    """Plot and optionally save a confusion matrix."""
    labels = labels or ['Not Canceled', 'Canceled']
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=labels
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Feature Importance ────────────────────────────────────────────────────────

def get_rf_importance(X_train, y_train, feature_names, n_estimators: int = 150) -> pd.Series:
    """Fit RandomForest and return feature importances as Series."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)


def get_mutual_info(X_train, y_train, feature_names) -> pd.Series:
    """Return mutual information scores as Series."""
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    return pd.Series(mi, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(rf_imp: pd.Series, mi_imp: pd.Series,
                             top_n: int = 15, save_path: str = None):
    """Side-by-side bar charts for RF and MI feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rf_imp.head(top_n).sort_values().plot.barh(ax=axes[0], color='steelblue')
    axes[0].set_title(f'Random Forest Importance (Top {top_n})')
    mi_imp.head(top_n).sort_values().plot.barh(ax=axes[1], color='darkorange')
    axes[1].set_title(f'Mutual Information Score (Top {top_n})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Feature Selection ─────────────────────────────────────────────────────────

def remove_high_correlation(X: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Drop one of each pair of features with correlation above threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f'Dropping {len(to_drop)} high-correlation columns: {to_drop}')
    return X.drop(columns=to_drop)


def select_top_features(rf_imp, mi_imp, chi2_imp,
                         drop_cols=None, top_n: int = 20) -> list:
    """Union of top-N features from RF, MI, and Chi2 minus dropped columns."""
    drop_cols = set(drop_cols or [])
    candidates = (
        set(rf_imp.head(top_n).index) |
        set(mi_imp.head(top_n).index) |
        set(chi2_imp.head(top_n).index)
    ) - drop_cols
    return list(candidates)[:top_n]


# ── Comparison Table ──────────────────────────────────────────────────────────

def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Build and display a before/after comparison table.
    results: list of dicts with keys Version, Features, Preprocessing, Model, ROC-AUC, F1
    """
    df = pd.DataFrame(results)
    print('\n=== BEFORE vs AFTER FEATURE ENGINEERING ===')
    print(df.to_string(index=False))
    return df
