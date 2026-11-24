"""Resolve repository assets independently of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.json"
ENCODER_MANIFEST = ROOT / "configs" / "encoder_manifest.json"
DEFAULT_DATA = ROOT / "data" / "raw" / "telco_customer_churn.xlsx"
DEFAULT_MODEL = ROOT / ".cache" / "models" / "all-MiniLM-L6-v2"
DEFAULT_OUTPUT = ROOT / "results" / "runs" / "latest"
