"""Load allowed predictors and fit transformations on training partitions."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .constants import DATA_SHA256, DROP_COLUMNS
from .io import digest


def load_data(path: Path, numeric: list[str]):
    if digest(path) != DATA_SHA256:
        raise ValueError("Dataset hash differs from the documented input")
    frame = pd.read_excel(path, sheet_name="Telco_Churn")
    assert frame.shape == (7043, 33)
    assert frame.CustomerID.is_unique
    y = frame["Churn Label"].map({"No": 0, "Yes": 1}).to_numpy(dtype=np.int64)
    X = frame.drop(columns=DROP_COLUMNS).copy()
    for name in numeric:
        X[name] = pd.to_numeric(X[name], errors="coerce")
    categorical = [name for name in X.columns if name not in numeric]
    # Pandas string extension types and object types receive the same treatment.
    for name in categorical:
        X[name] = X[name].astype(str)
    assert X.shape[1] == 19
    assert set(DROP_COLUMNS).isdisjoint(X.columns)
    return frame, X, y, categorical


def make_preprocessor(numeric, categorical):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
