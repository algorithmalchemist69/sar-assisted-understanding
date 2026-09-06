"""Multi-label metrics.

BigEarthNet v2.0 is a multi-label benchmark (mean 3.5 labels per patch here), so
"accuracy" has to be stated carefully. We report:

* **macro F1**   -- the metric the assignment names; unweighted mean over classes,
                    so rare classes count as much as common ones. Primary.
* **micro F1**   -- pools all label decisions; dominated by frequent classes.
* **macro mAP**  -- mean average precision, threshold-free, so it separates
                    "the model ranks worse" from "the 0.5 threshold is misplaced".
* **Hamming accuracy** -- fraction of the N x C label decisions that are correct.
* **Exact-match accuracy** -- fraction of patches whose full label set is right.
                    Strict, and the closest analogue to single-label accuracy.

The decision threshold is fixed at 0.5 and is never tuned, on test or elsewhere.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(np.int8)
    y_true = y_true.astype(np.int8)

    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_accuracy": float((y_true == y_pred).mean()),
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()),
    }
    # AP is undefined for a class with no positives in this split.
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            aps.append(average_precision_score(y_true[:, c], probs[:, c]))
    out["macro_map"] = float(np.mean(aps)) if aps else float("nan")
    return out


def per_class_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str],
                      threshold: float = 0.5) -> list[dict]:
    y_pred = (probs >= threshold).astype(np.int8)
    y_true = y_true.astype(np.int8)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(classes))
    )
    rows = []
    for i, name in enumerate(classes):
        t, q = y_true[:, i], y_pred[:, i]
        ap = average_precision_score(t, probs[:, i]) if t.sum() > 0 else float("nan")
        rows.append({
            "class": name,
            "support": int(s[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "average_precision": float(ap),
            "tp": int(((t == 1) & (q == 1)).sum()),
            "fp": int(((t == 0) & (q == 1)).sum()),
            "fn": int(((t == 1) & (q == 0)).sum()),
            "tn": int(((t == 0) & (q == 0)).sum()),
        })
    return rows


def error_matrix(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """M[i, j] = P(model predicts class j | patch truly contains class i).

    The multi-label stand-in for a confusion matrix. The diagonal is per-class
    recall; large off-diagonal entries show which classes get dragged along, and
    comparing arm B with arm C shows which of those confusions SAR resolves.
    """
    y_pred = (probs >= threshold).astype(np.float64)
    n_classes = y_true.shape[1]
    m = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        rows = y_true[:, i] == 1
        m[i] = y_pred[rows].mean(0) if rows.sum() else np.nan
    return m
