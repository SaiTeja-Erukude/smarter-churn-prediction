import numpy as np
import pandas as pd

from churn_prediction.constants import DROP_COLUMNS
from churn_prediction.paths import DEFAULT_DATA
from churn_prediction.preprocessing import load_data, make_preprocessor


def test_source_dataset_and_predictor_exclusions(protocol):
    frame, features, target, categorical = load_data(DEFAULT_DATA, protocol["numeric_features"])
    assert frame.shape == (7043, 33)
    assert features.shape == (7043, 19)
    assert target.sum() == 1869
    assert features["Total Charges"].isna().sum() == 11
    assert set(DROP_COLUMNS).isdisjoint(features.columns)
    assert len(categorical) == 16


def test_holdout_cannot_change_scaling_or_category_vocabulary():
    training = pd.DataFrame({"amount": [10.0, 20.0, np.nan], "contract": ["a", "b", "a"]})
    holdout = pd.DataFrame({"amount": [1000.0, np.nan], "contract": ["unseen", "a"]})
    prep = make_preprocessor(["amount"], ["contract"])
    prep.fit(training)
    scaler = prep.named_transformers_["numeric"].named_steps["scale"]
    expected_mean = scaler.mean_.copy()
    transformed = prep.transform(holdout)
    assert expected_mean[0] == 15.0
    assert np.array_equal(scaler.mean_, expected_mean)
    assert transformed[0, 0] > 100
    assert transformed[1, 0] == 0  # Imputed from the training median.
    assert np.array_equal(transformed[0, 1:], [0, 0])
    assert list(prep.named_transformers_["categorical"].categories_[0]) == ["a", "b"]
