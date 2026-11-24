"""Binary churn metrics and paired holdout bootstrap comparisons."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_predictions(y, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": len(y),
    }
    assert tn + fp + fn + tp == len(y)
    assert np.isclose(metrics["f1"], 2 * tp / (2 * tp + fp + fn)) if tp else metrics["f1"] == 0
    return metrics


def paired_bootstrap(y, baseline, hybrid, count, seed=42, threshold=0.5):
    rng = np.random.default_rng(seed + 1701)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    differences = {key: [] for key in ("accuracy", "f1", "roc_auc")}
    for _ in range(count):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        ys = y[sample]
        for key, fn in (("accuracy", accuracy_score), ("f1", f1_score), ("roc_auc", roc_auc_score)):
            first = baseline[sample] if key == "roc_auc" else baseline[sample] >= threshold
            second = hybrid[sample] if key == "roc_auc" else hybrid[sample] >= threshold
            differences[key].append(float(fn(ys, second) - fn(ys, first)))
    measured_baseline = score_predictions(y, baseline, threshold)
    measured_hybrid = score_predictions(y, hybrid, threshold)
    return {
        key: {
            "difference": measured_hybrid[key] - measured_baseline[key],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_replicates": count,
        }
        for key, values in differences.items()
    }
