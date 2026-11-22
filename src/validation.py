"""Cross-validation, out-of-fold scores, F2 threshold and metrics. Used by 03a, 03b, 03c, 04.

Owner: Pranav
Provides: get_cv(), oof_scores(), save_oof(), best_f2_threshold(), classification_metrics().
See README for the agreed formats.
"""
import copy
import inspect
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import StratifiedKFold

from src.data import SEED, PARENT_DIR

SCORES_DIR = os.path.join(PARENT_DIR, "results", "scores")
N_SPLITS = 5


def get_cv(n_splits=N_SPLITS):
    """The one CV splitter everyone uses, so SVM, MLP and FIS see identical folds."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _score(model, X):
    """Continuous score where higher = more likely PD."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.score(X)  # FIS: score(X) -> values in [0, 1]


def oof_scores(model, X, y, cv=None):
    """Out-of-fold scores: each patient is scored by a model fitted on the other folds.

    `model` is any unfitted estimator (e.g. a Pipeline) with fit(X, y), or a FIS with
    fit(X) and score(X). A fresh copy is fitted in every fold, so preprocessing and
    FIS ranges are learned from the training folds only.
    Returns a Series indexed like X.
    """
    cv = cv or get_cv()
    scores = pd.Series(np.nan, index=X.index, dtype=float)
    for train_idx, val_idx in cv.split(X, y):
        m = clone(model) if hasattr(model, "get_params") else copy.deepcopy(model)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        if len(inspect.signature(m.fit).parameters) >= 2:
            m.fit(X_tr, y_tr)      # sklearn: fit(X, y)
        else:
            m.fit(X_tr)            # FIS: fit(X), no labels
        scores.iloc[val_idx] = _score(m, X.iloc[val_idx])
    return scores


def save_oof(name, y, scores):
    """Write results/scores/<name>_oof.csv as PatientID, y_true, score."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    out = pd.DataFrame({"y_true": y.astype(int), "score": scores}, index=y.index)
    out.index.name = "PatientID"
    path = os.path.join(SCORES_DIR, f"{name}_oof.csv")
    out.to_csv(path)
    return path


def best_f2_threshold(y_true, scores):
    """Threshold that maximises F2 on the given (out-of-fold) scores.

    Candidates are the unique score values; a patient is predicted PD when score >= t.
    Returns (threshold, best_f2).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    best_t, best_f2 = None, -1.0
    for t in np.unique(scores):
        f2 = fbeta_score(y_true, (scores >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = float(t), f2
    return best_t, best_f2


def classification_metrics(y_true, scores, threshold, model_name=""):
    """Confusion matrix + recall, precision, specificity, F2 at a fixed threshold.

    Returns a dict in the test_metrics.csv column order.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    return {"model": model_name, "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "recall": float(recall), "precision": float(precision),
            "specificity": float(specificity), "F2": float(f2)}