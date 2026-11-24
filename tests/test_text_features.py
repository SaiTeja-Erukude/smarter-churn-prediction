import pandas as pd
import pytest

from churn_prediction.configuration import validate_protocol
from churn_prediction.text_features import make_profile_text


def test_descriptions_use_only_allowed_account_fields(protocol):
    frame = pd.DataFrame(
        {
            "Contract": ["Month-to-month"],
            "Tenure Months": [12],
            "Monthly Charges": [35.0],
            "Internet Service": ["DSL"],
            "Tech Support": ["No"],
            "Churn Label": ["Yes"],
            "Churn Reason": ["SECRET"],
        }
    )
    original = make_profile_text(frame, protocol)
    frame["Churn Label"] = "No"
    frame["Churn Reason"] = "DIFFERENT"
    assert make_profile_text(frame, protocol) == original
    assert "SECRET" not in original[0]
    assert "up to 12 months" in original[0] and "up to 35 dollars" in original[0]


def test_text_cutoffs_follow_configuration(protocol):
    protocol["tenure_bin_edges_months"] = [6, 24]
    protocol["monthly_charge_bin_edges"] = [20, 60]
    frame = pd.DataFrame(
        {
            "Contract": ["One year"],
            "Tenure Months": [7],
            "Monthly Charges": [61.0],
            "Internet Service": ["DSL"],
            "Tech Support": ["Yes"],
        }
    )
    text = make_profile_text(frame, protocol)[0]
    assert "7 to 24 months" in text and "more than 60 dollars" in text


@pytest.mark.parametrize(
    "key,value",
    [
        ("decision_threshold", 1),
        ("development_cv_folds", 1),
        ("bootstrap_replicates", 0),
        ("text_source_features", ["Churn Reason"]),
    ],
)
def test_invalid_protocols_are_rejected(protocol, key, value):
    protocol[key] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)
