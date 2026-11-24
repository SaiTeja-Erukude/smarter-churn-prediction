"""Independently reconcile a run with saved splits, predictions, and SHAP arrays."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .configuration import validate_protocol
from .constants import CONTROLS, CORE_MODELS, DATA_SHA256, METRICS
from .io import digest, write_json
from .paths import ROOT


def verify(results):
    results = Path(results)
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    assert manifest["dataset_sha256"] == DATA_SHA256, "Dataset identity differs"
    assert (
        digest(results / "protocol.json")
        == manifest["protocol_sha256"]
        == summary["protocol_sha256"]
    ), "Saved protocol changed"
    sources = {p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob("*.py")}
    sources.add("configs/encoder_manifest.json")
    assert sources == set(manifest["source_hashes"]), "Experiment source file set changed"
    for relative, expected in manifest["source_hashes"].items():
        assert digest(ROOT / relative) == expected, f"Experiment source changed: {relative}"
    models = CORE_MODELS + (CONTROLS if manifest["include_controls"] else [])
    assert manifest["models"] == models, "Model manifest differs"
    predictions = pd.read_csv(results / "predictions.csv")
    splits = pd.read_csv(results / "splits.csv").set_index("row_index")
    cv = pd.read_csv(results / "cv_scores.csv")
    assert splits.index.is_unique and set(splits.index) == set(range(7043))
    assert set(splits.partition) == {"test", "development"}
    assert set(splits.target) == {0, 1} and splits.target.sum() == 1869
    test = splits[splits.partition == "test"]
    development = splits[splits.partition == "development"]
    assert len(test) == summary["test_rows"] == int(np.ceil(7043 * protocol["test_fraction"]))
    assert len(development) == summary["development_rows"]
    assert test.target.sum() == summary["test_churn"]
    assert development.target.sum() == summary["development_churn"]
    assert set(test.index).isdisjoint(development.index)
    folds = protocol["development_cv_folds"]
    assert set(test.validation_fold) == {-1}
    assert set(development.validation_fold) == set(range(1, folds + 1))
    expected_groups = {
        (stage, model)
        for stage in ["test", *[f"cv-{n}" for n in range(1, folds + 1)]]
        for model in models
    }
    assert set(predictions.groupby(["stage", "model"]).groups) == expected_groups
    assert len(cv) == folds * len(models) and not cv.duplicated(["fold", "model"]).any()
    assert set(cv.model) == set(models) == set(summary["cv"])
    assert set(summary["test"]) == set(models) | {"Majority baseline"}
    checked = 0
    for (stage, model), group in predictions.groupby(["stage", "model"]):
        expected_rows = (
            test.index
            if stage == "test"
            else splits[splits.validation_fold == int(stage.split("-")[1])].index
        )
        assert group.row_index.is_unique and set(group.row_index) == set(expected_rows)
        assert np.array_equal(
            group.target.to_numpy(), splits.loc[group.row_index, "target"].to_numpy()
        )
        assert np.isfinite(group.probability).all() and group.probability.between(0, 1).all()
        assert np.array_equal(
            group.prediction, (group.probability >= protocol["decision_threshold"]).astype(int)
        )
        y, probability, predicted = group.target, group.probability, group.prediction
        measured = {
            "accuracy": accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability),
        }
        values = (
            summary["test"][model]
            if stage == "test"
            else cv[(cv.model == model) & (cv.fold == int(stage.split("-")[1]))].iloc[0]
        )
        for metric, actual in measured.items():
            assert np.isclose(actual, values[metric], rtol=0, atol=1e-12), (stage, model, metric)
        matrix = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        for metric, actual in zip(("tn", "fp", "fn", "tp"), matrix):
            assert actual == values[metric], (stage, model, metric)
        assert values["n"] == len(group)
        checked += 1
    for model, values in summary["cv"].items():
        group = cv[cv.model == model]
        assert len(group) == folds and set(values) == set(METRICS)
        for metric, aggregates in values.items():
            assert np.isclose(group[metric].mean(), aggregates["mean"], rtol=0, atol=1e-12)
            assert np.isclose(group[metric].std(ddof=1), aggregates["std"], rtol=0, atol=1e-12)
    for name in ("structured_shap.npz", "hybrid_shap.npz"):
        with np.load(results / name) as data:
            assert len(data["row_indices"]) == len(test) and set(data["row_indices"]) == set(
                test.index
            )
            assert np.isfinite(data["contributions"]).all()
            assert np.allclose(data["contributions"].sum(axis=1), data["raw_margins"], atol=1e-4)
    for metric, values in summary["paired_bootstrap_hybrid_minus_structured"].items():
        actual = (
            summary["test"]["Hybrid MiniLM + XGBoost"][metric] - summary["test"]["XGBoost"][metric]
        )
        assert np.isclose(actual, values["difference"], rtol=0, atol=1e-12)
        assert values["ci_low"] <= values["ci_high"]
        assert values["bootstrap_replicates"] == protocol["bootstrap_replicates"]
    report = {
        "passed": True,
        "model_stage_comparisons": checked,
        "development_rows": len(development),
        "holdout_rows": len(test),
        "holdout_churn": int(test.target.sum()),
        "verified": [
            "prediction completeness and metrics",
            "fold summaries",
            "partition assignments",
            "SHAP additivity",
            "source and protocol hashes",
            "paired-comparison point estimates",
        ],
    }
    write_json(results / "verification.json", report)
    print(json.dumps(report, indent=2))
    return report
