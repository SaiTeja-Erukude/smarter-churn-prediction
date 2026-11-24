"""Validate configurable split, training, and text-generation parameters."""

from .constants import DROP_COLUMNS


def validate_protocol(protocol):
    if not isinstance(protocol["seed"], int) or protocol["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < protocol["test_fraction"] < 1:
        raise ValueError("test_fraction must be between zero and one")
    if protocol["stratify_holdout"] is not True:
        raise ValueError("The holdout must be stratified")
    if (
        not isinstance(protocol["development_cv_folds"], int)
        or protocol["development_cv_folds"] < 2
    ):
        raise ValueError("development_cv_folds must be at least two")
    if not 0 < protocol["decision_threshold"] < 1:
        raise ValueError("decision_threshold must be between zero and one")
    if (
        not isinstance(protocol["bootstrap_replicates"], int)
        or protocol["bootstrap_replicates"] < 1
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if protocol["numeric_features"] != ["Tenure Months", "Monthly Charges", "Total Charges"]:
        raise ValueError("Use the documented three numeric predictors")
    if protocol["text_source_features"] != [
        "Contract",
        "Tenure Months",
        "Monthly Charges",
        "Internet Service",
        "Tech Support",
    ]:
        raise ValueError("Use the documented five account-description predictors")
    if set(protocol["text_source_features"]) & set(DROP_COLUMNS):
        raise ValueError("Text inputs must exclude outcomes and identifiers")
    for key in ("tenure_bin_edges_months", "monthly_charge_bin_edges"):
        edges = protocol[key]
        if len(edges) != 2 or not 0 <= edges[0] < edges[1]:
            raise ValueError(f"{key} must contain two increasing nonnegative cutoffs")
    if not 1 <= protocol["encoder"]["maximum_word_pieces"] <= 256:
        raise ValueError("maximum_word_pieces must be between 1 and 256")
    for key in ("feedforward", "deep_feedforward"):
        if not isinstance(protocol[key]["epochs"], int) or protocol[key]["epochs"] < 1:
            raise ValueError(f"{key}.epochs must be a positive integer")
