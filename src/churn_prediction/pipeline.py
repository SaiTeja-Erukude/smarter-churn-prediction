"""Cross-validation, holdout evaluation, and auditable experiment outputs."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
import xgboost as xgb
from .constants import CORE_MODELS, CONTROLS, DATA_COMMIT, DATA_SHA256, METRICS
from .configuration import validate_protocol
from .evaluation import paired_bootstrap, score_predictions
from .io import digest, write_json
from .models import fit_model, make_model
from .paths import ROOT
from .plotting import save_figures
from .preprocessing import load_data, make_preprocessor
from .text_features import encode_profiles, make_profile_text


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py")) + [
        ROOT / "configs" / "encoder_manifest.json"
    ]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def run(args):
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new --output path.")
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.config
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    seed = protocol["seed"]
    threshold = protocol["decision_threshold"]
    # Save the fixed design before evaluating any outcomes.
    write_json(out / "protocol.json", protocol)
    frame, X, y, categorical = load_data(args.data, protocol["numeric_features"])
    development, test = train_test_split(
        np.arange(len(X)), test_size=protocol["test_fraction"], stratify=y, random_state=seed
    )
    assert set(development).isdisjoint(test)
    folds = StratifiedKFold(
        n_splits=protocol["development_cv_folds"], shuffle=True, random_state=seed
    )
    split_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(X)),
            "customer_id": frame.CustomerID,
            "target": y,
            "partition": "development",
            "validation_fold": -1,
        }
    )
    split_frame.loc[test, "partition"] = "test"
    cv_splits = []
    for fold, (train_local, valid_local) in enumerate(folds.split(development, y[development]), 1):
        train, valid = development[train_local], development[valid_local]
        split_frame.loc[valid, "validation_fold"] = fold
        cv_splits.append((fold, train, valid))
    split_frame.to_csv(out / "splits.csv", index=False)
    notes = make_profile_text(X, protocol)
    E, profile_ids, encoding_info = encode_profiles(notes, args.model_dir, out, protocol)
    print(
        f"Dataset: {len(X)} rows; dev={len(development)}, test={len(test)}; {encoding_info['unique_profiles']} unique text profiles",
        flush=True,
    )

    fold_scores, prediction_rows, runtimes, shap_rows = [], [], [], []
    test_scores, test_probabilities = {}, {}
    all_models = CORE_MODELS + (CONTROLS if args.include_controls else [])
    passes = cv_splits + [(0, development, test)]
    for fold, train, valid in passes:
        stage = "test" if fold == 0 else f"cv-{fold}"
        prep = make_preprocessor(protocol["numeric_features"], categorical)
        # Every learned transformation is fitted solely on this training slice.
        S_train = prep.fit_transform(X.iloc[train]).astype(np.float32)
        S_valid = prep.transform(X.iloc[valid]).astype(np.float32)
        ids_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        I_train = ids_encoder.fit_transform(profile_ids[train])
        I_valid = ids_encoder.transform(profile_ids[valid])
        shuffle_train = np.random.default_rng(seed + 100 + fold).permutation(E[train])
        shuffle_valid = np.random.default_rng(seed + 200 + fold).permutation(E[valid])
        feature_sets = {
            "structured": (S_train, S_valid),
            "hybrid": (np.column_stack([S_train, E[train]]), np.column_stack([S_valid, E[valid]])),
            "text": (E[train], E[valid]),
            "shuffled": (
                np.column_stack([S_train, shuffle_train]),
                np.column_stack([S_valid, shuffle_valid]),
            ),
            "ids": (np.column_stack([S_train, I_train]), np.column_stack([S_valid, I_valid])),
        }
        for name in all_models:
            feature_key = {
                "Hybrid MiniLM + XGBoost": "hybrid",
                "Text-only XGBoost": "text",
                "Structured + shuffled embeddings": "shuffled",
                "Structured + profile-ID one-hot": "ids",
            }.get(name, "structured")
            fit_features, valid_features = feature_sets[feature_key]
            model = make_model(name, protocol, y[train])
            started = time.perf_counter()
            with threadpool_limits(limits=4):
                fit_model(model, name, fit_features, y[train], protocol)
                probabilities = model.predict_proba(valid_features)[:, 1]
            elapsed = time.perf_counter() - started
            scores = score_predictions(y[valid], probabilities, threshold)
            runtimes.append({"stage": stage, "model": name, "fit_and_prediction_seconds": elapsed})
            if fold:
                fold_scores.append({"fold": fold, "model": name, **scores})
            else:
                test_scores[name] = scores
                test_probabilities[name] = probabilities
            for idx, probability in zip(valid, probabilities):
                prediction_rows.append(
                    {
                        "stage": stage,
                        "row_index": int(idx),
                        "model": name,
                        "target": int(y[idx]),
                        "probability": float(probability),
                        "prediction": int(probability >= threshold),
                    }
                )
            print(
                f"{stage:5} {name:35} AUC={scores['roc_auc']:.4f} F1={scores['f1']:.4f} ({elapsed:.1f}s)",
                flush=True,
            )
            if fold == 0 and name in ("XGBoost", "Hybrid MiniLM + XGBoost"):
                booster = model.get_booster()
                matrix = xgb.DMatrix(valid_features)
                contributions = booster.predict(matrix, pred_contribs=True)
                margins = booster.predict(matrix, output_margin=True)
                assert np.allclose(contributions.sum(axis=1), margins, atol=1e-4)
                group_names = protocol["numeric_features"].copy()
                for name_index, column in enumerate(categorical):
                    group_names.extend(
                        [column]
                        * len(prep.named_transformers_["categorical"].categories_[name_index])
                    )
                if feature_key == "hybrid":
                    group_names.extend(["Profile embeddings (384 dims)"] * 384)
                assert len(group_names) == contributions.shape[1] - 1
                for group in dict.fromkeys(group_names):
                    selection = np.array([item == group for item in group_names])
                    grouped = contributions[:, :-1][:, selection].sum(axis=1)
                    shap_rows.append(
                        {
                            "model": name,
                            "feature": group,
                            "mean_absolute_group_contribution": float(np.abs(grouped).mean()),
                        }
                    )
                np.savez_compressed(
                    out / ("hybrid_shap.npz" if feature_key == "hybrid" else "structured_shap.npz"),
                    row_indices=valid,
                    contributions=contributions,
                    raw_margins=margins,
                    feature_groups=np.array(group_names),
                )

        # Persist after every stage, including before any optional plotting.
        pd.DataFrame(fold_scores).to_csv(out / "cv_scores.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(runtimes).to_csv(out / "runtimes.csv", index=False)

    prior = float(y[development].mean())
    test_scores["Majority baseline"] = score_predictions(
        y[test], np.full(len(test), prior), threshold
    )
    differences = paired_bootstrap(
        y[test],
        test_probabilities["XGBoost"],
        test_probabilities["Hybrid MiniLM + XGBoost"],
        protocol["bootstrap_replicates"],
        seed=seed,
        threshold=threshold,
    )
    cv_frame = pd.DataFrame(fold_scores)
    cv_summary = {}
    for name in all_models:
        selected = cv_frame[cv_frame.model == name]
        cv_summary[name] = {
            metric: {
                "mean": float(selected[metric].mean()),
                "std": float(selected[metric].std(ddof=1)),
            }
            for metric in METRICS
        }
    shap_frame = pd.DataFrame(shap_rows)
    shap_frame.to_csv(out / "shap_groups.csv", index=False)
    summary = {
        "dataset_rows": len(X),
        "development_rows": len(development),
        "test_rows": len(test),
        "dataset_churn": int(y.sum()),
        "development_churn": int(y[development].sum()),
        "test_churn": int(y[test].sum()),
        "structured_features": list(X.columns),
        "blank_total_charges": int(X["Total Charges"].isna().sum()),
        "cv": cv_summary,
        "test": test_scores,
        "paired_bootstrap_hybrid_minus_structured": differences,
        "encoder": encoding_info,
        "protocol_sha256": digest(out / "protocol.json"),
    }
    write_json(out / "summary.json", summary)
    manifest = {
        "dataset_sha256": DATA_SHA256,
        "source_dataset_commit": DATA_COMMIT,
        "source_hashes": source_hashes(),
        "protocol_sha256": digest(out / "protocol.json"),
        "input_config_sha256": digest(protocol_path),
        "models": all_models,
        "include_controls": args.include_controls,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "onnxruntime",
                "tokenizers",
                "matplotlib",
                "openpyxl",
                "threadpoolctl",
            )
        },
    }
    write_json(out / "manifest.json", manifest)
    save_figures(out, y[test], test_scores, test_probabilities, shap_frame)
    print("Finished. Results and all prediction-level evidence:", out, flush=True)
