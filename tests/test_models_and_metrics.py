import numpy as np
import pytest

from churn_prediction.evaluation import paired_bootstrap, score_predictions
from churn_prediction.models import fit_model, make_model


def test_xgboost_weight_uses_training_class_counts(protocol):
    model = make_model("XGBoost", protocol, np.array([0, 0, 0, 1]))
    assert model.get_params()["scale_pos_weight"] == 3
    assert model.get_params()["random_state"] == protocol["seed"]


@pytest.mark.parametrize(
    "name,key", [("Feedforward NN", "feedforward"), ("Deep FNN", "deep_feedforward")]
)
def test_neural_training_obeys_configured_epochs(protocol, name, key):
    protocol[key]["epochs"] = 2
    protocol[key]["batch_size"] = 4
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    target = np.array([0, 0, 0, 1])
    model = make_model(name, protocol, target)
    fit_model(model, name, features, target, protocol)
    assert len(model.loss_curve_) == 2
    probabilities = model.predict_proba(features)[:, 1]
    assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()


def test_metric_threshold_and_confusion_counts():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.65, 0.9])
    metrics = score_predictions(target, probabilities, threshold=0.7)
    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert [metrics[key] for key in ("tn", "fp", "fn", "tp")] == [2, 0, 1, 1]
    assert metrics["roc_auc"] == 1


def test_identical_probabilities_have_zero_paired_difference():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])
    result = paired_bootstrap(target, probabilities, probabilities, count=10, threshold=0.6)
    for metric in ("accuracy", "f1", "roc_auc"):
        assert result[metric]["difference"] == 0
        assert result[metric]["ci_low"] == result[metric]["ci_high"] == 0
