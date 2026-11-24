"""Deterministic account summaries and frozen MiniLM sentence embeddings."""

import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer
from .io import digest
from .paths import ENCODER_MANIFEST


def make_profile_text(X: pd.DataFrame, protocol: dict) -> list[str]:
    """Account descriptions, never simulated complaints or observed tickets.

    The allowlist excludes the target and every outcome/proxy field. All cutoffs
    are fixed in protocol.json, not estimated using training or test labels.
    """
    allowed = protocol["text_source_features"]
    assert set(allowed).issubset(X.columns)
    source = X[allowed]
    tenure_low, tenure_high = protocol["tenure_bin_edges_months"]
    charge_low, charge_high = protocol["monthly_charge_bin_edges"]
    notes = []
    for contract, tenure, charge, internet, support in source.itertuples(index=False, name=None):
        tenure_band = (
            f"up to {tenure_low} months"
            if tenure <= tenure_low
            else f"{tenure_low + 1} to {tenure_high} months"
            if tenure <= tenure_high
            else f"more than {tenure_high} months"
        )
        charge_band = (
            f"up to {charge_low} dollars"
            if charge <= charge_low
            else f"more than {charge_low} and up to {charge_high} dollars"
            if charge <= charge_high
            else f"more than {charge_high} dollars"
        )
        notes.append(
            f"Account description. Contract: {contract}. Tenure: {tenure_band}. "
            f"Monthly charge: {charge_band}. Internet service: {internet}. "
            f"Technical support subscription: {support}."
        )
    return notes


def encode_profiles(notes: list[str], model_dir: Path, out: Path, protocol: dict):
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = json.loads(ENCODER_MANIFEST.read_text(encoding="utf-8"))
    if manifest != expected:
        raise ValueError("Encoder manifest differs from the pinned checkpoint; run fetch-assets")
    assert manifest["revision"] == protocol["encoder"]["revision"]
    for entry in manifest["files"]:
        assert digest(model_dir / entry["file"]) == entry["sha256"]
    unique, inverse = np.unique(notes, return_inverse=True)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=protocol["encoder"]["maximum_word_pieces"])
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_dir / "onnx" / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    names = {item.name for item in session.get_inputs()}
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(unique), 32):
        batch = tokenizer.encode_batch(unique[start : start + 32].tolist())
        inputs = {
            "input_ids": np.array([item.ids for item in batch], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in batch], dtype=np.int64),
            "token_type_ids": np.array([item.type_ids for item in batch], dtype=np.int64),
        }
        tokens = session.run(
            None, {name: value for name, value in inputs.items() if name in names}
        )[0]
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (tokens * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        vectors.append(pooled)
    embeddings = np.concatenate(vectors, axis=0)[inverse].astype(np.float32)
    assert embeddings.shape == (len(notes), 384)
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    np.save(out / "profile_embeddings.npy", embeddings)
    pd.DataFrame(
        {"row_index": np.arange(len(notes)), "profile_id": inverse, "profile_text": notes}
    ).to_csv(out / "profile_text.csv", index=False)
    return (
        embeddings,
        inverse.reshape(-1, 1),
        {
            "unique_profiles": len(unique),
            "dimensions": 384,
            "encoding_seconds": time.perf_counter() - started,
            "checkpoint_manifest": manifest,
        },
    )
