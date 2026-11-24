"""Factories and fitting for the five model families."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from .constants import CORE_MODELS, CONTROLS


def make_model(name: str, protocol: dict, y_fit: np.ndarray):
    if name not in CORE_MODELS + CONTROLS:
        raise ValueError(f"Unknown model: {name}")
    if name == "Logistic Regression":
        return LogisticRegression(**protocol["logistic_regression"], random_state=protocol["seed"])
    if name in ("Feedforward NN", "Deep FNN"):
        spec = protocol["feedforward" if name == "Feedforward NN" else "deep_feedforward"].copy()
        spec.pop("epochs")
        spec["hidden_layer_sizes"] = tuple(spec["hidden_layer_sizes"])
        return MLPClassifier(
            **spec, random_state=protocol["seed"], shuffle=True, early_stopping=False
        )
    weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    return xgb.XGBClassifier(
        **protocol["xgboost"],
        random_state=protocol["seed"],
        scale_pos_weight=weight,
        objective="binary:logistic",
    )


def fit_model(model, name, X_fit, y_fit, protocol):
    if name in ("Feedforward NN", "Deep FNN"):
        weights = compute_sample_weight("balanced", y_fit)
        epochs = protocol["feedforward" if name == "Feedforward NN" else "deep_feedforward"][
            "epochs"
        ]
        for _ in range(epochs):
            model.partial_fit(X_fit, y_fit, classes=np.array([0, 1]), sample_weight=weights)
        assert model.n_iter_ == 1  # partial_fit performs one additional epoch per call
        assert len(model.loss_curve_) == epochs
    else:
        model.fit(X_fit, y_fit)
