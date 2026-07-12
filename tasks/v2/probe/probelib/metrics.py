"""Evaluation metrics for the probe (binary classification on frozen embeddings).

Thin wrappers around scikit-learn so the rest of the code reads in plain terms
(auroc / accuracy / f1) and returns plain Python floats (JSON-serializable).
"""

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def auroc(y_true, proba):
    """Area under the ROC curve. Threshold-free and class-balance independent —
    the metric we use to RANK the pretraining runs. `proba` = P(class = 1)."""
    return float(roc_auc_score(y_true, proba))


def accuracy(y_true, pred):
    """Fraction correct at the 0.5 threshold. Class-balance dependent."""
    return float(accuracy_score(y_true, pred))


def f1(y_true, pred):
    """F1 of the positive class (harmonic mean of precision and recall)."""
    return float(f1_score(y_true, pred, zero_division=0))
