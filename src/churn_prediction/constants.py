"""Dataset identity, predictor exclusions, and model names."""

DATA_SHA256 = "ce82e608d869df77107e3445ae5f868a950bedc57ff032f14567865aef9a6abb"
DATA_COMMIT = "8f2cd61b9cee501bf7011694b76644c20eded5a0"
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]
CORE_MODELS = [
    "Logistic Regression",
    "XGBoost",
    "Feedforward NN",
    "Deep FNN",
    "Hybrid MiniLM + XGBoost",
]
CONTROLS = [
    "Text-only XGBoost",
    "Structured + shuffled embeddings",
    "Structured + profile-ID one-hot",
]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
