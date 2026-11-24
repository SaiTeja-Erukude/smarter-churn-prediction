# Smarter Churn Prediction Using LLM-Powered Hybrid Models

We compare logistic regression, XGBoost, two feedforward neural networks, and a
hybrid that combines structured account features with frozen MiniLM sentence
embeddings. The repository contains a runnable five-model experiment pipeline
for the IBM Telco Customer Churn sample.

## Setup

Use Python **3.13** and run the following commands from this repository's root.
Install the checkout in editable mode so the package can locate its configuration
and data files. The workflow runs locally on the CPU.

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Install the pinned experiment dependencies and development tools:

```sh
python -m pip install -e ".[dev]"
```

`requirements.txt` pins the experiment dependencies. The neural networks use
scikit-learn's `MLPClassifier`; the encoder uses ONNX Runtime on the CPU.

## Run the experiment

```sh
python -m churn_prediction fetch-assets
python -m churn_prediction run
python -m churn_prediction verify
```

The first command verifies the bundled spreadsheet and downloads approximately
91 MB of pinned encoder files into `.cache/models/all-MiniLM-L6-v2/`. It checks
SHA-256 hashes before accepting each file. After downloading the assets, training
and verification can run offline.

The default run fits all five models with five-fold stratified cross-validation
on the development partition, then evaluates them on an untouched 20% holdout.
It also evaluates a majority-class baseline. The configuration fixes the seed,
model parameters, text rules, threshold, and bootstrap settings before training.
Imputation, scaling, category vocabularies, and class weights are learned only
from the training partition in each evaluation stage.

The output directory defaults to `results/runs/latest/`. To preserve previous
runs, the command requires an empty output directory. Use a new name for later
runs:

```sh
python -m churn_prediction run --output results/runs/experiment_02
python -m churn_prediction verify --results results/runs/experiment_02
```

To include the three additional text-only, shuffled-embedding, and profile-ID
controls:

```sh
python -m churn_prediction run --include-controls --output results/runs/with_controls
python -m churn_prediction verify --results results/runs/with_controls
```

`--config`, `--data`, and `--model-dir` support explicit paths; see
`python -m churn_prediction run --help`. Defaults resolve from the installed
checkout regardless of the current directory. Explicit relative paths resolve
from the current directory. The `churn-prediction` console command and the three
wrappers in `src/scripts/` expose the same workflow after installation.

The wrapper commands can also be run directly from the repository root:

```sh
python src/scripts/fetch_assets.py
python src/scripts/run_experiments.py
python src/scripts/verify_results.py
```

## Models and text features

| Model | Implementation |
| --- | --- |
| Logistic regression | Balanced class weights, scaled numeric and one-hot categorical features |
| XGBoost | Structured features with training-partition class imbalance weighting |
| Feedforward NN | ReLU hidden layers with 64 and 32 units |
| Deep FNN | ReLU hidden layers with 128, 64, and 32 units |
| Hybrid MiniLM + XGBoost | Structured features concatenated with 384-dimensional sentence embeddings |

We generate deterministic account descriptions from contract, tenure, monthly
charges, internet service, and technical support subscription. These descriptions
provide a **synthetic text representation of account attributes**. The frozen
encoder is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Experiment outputs

Executions of `configs/experiment.json` save their measured results under
`results/runs/`. Each run directory contains its own predictions, configuration,
metrics, and ROC, confusion-matrix, and SHAP plots. The run command creates this
directory automatically; `results/` is ignored by Git.

`summary.json` contains cross-validation means and standard deviations, holdout
accuracy, precision, recall, churn F1, ROC-AUC, and average precision.

| Output | Contents |
| --- | --- |
| `summary.json`, `cv_scores.csv` | Holdout metrics, fold scores, and aggregate comparisons |
| `predictions.csv`, `splits.csv` | Targets, probabilities, predictions, and partition assignments |
| `profile_text.csv`, `profile_embeddings.npy` | Generated account descriptions and frozen text embeddings |
| `structured_shap.npz`, `hybrid_shap.npz`, `shap_groups.csv` | Tree-model feature contributions |
| `protocol.json`, `manifest.json`, `verification.json` | Configuration, source hashes, package versions, and verification |
| `runtimes.csv`, `figures/` | Model timings and vector plots |

The run exports probabilities, class predictions, split assignments, fold
scores, timings, configuration, package versions, and source hashes. Verification
checks prediction completeness, recalculates metrics and fold summaries, checks
SHAP additivity, and detects changes to the experiment source or saved protocol.
Grouped SHAP contributions come directly from XGBoost in raw log-odds units.

Cross-validation standard deviations describe variation among folds. Paired
bootstrap intervals summarize uncertainty conditional on the fitted models and
the fixed holdout. The evaluation focuses on the IBM sample and its synthetic
account descriptions.

## Repository layout

```text
github-repo/
├── configs/
│   ├── experiment.json          # Splits, models, encoder, and evaluation protocol
│   └── encoder_manifest.json    # Pinned encoder revision and asset hashes
├── data/
│   ├── README.md                # Dataset source and schema
│   └── raw/telco_customer_churn.xlsx
├── src/
│   ├── churn_prediction/        # Reusable experiment implementation
│   └── scripts/                 # Fetch, run, and verify command wrappers
├── tests/                       # Data, leakage, model, and metric checks
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Downloaded model files, generated run outputs, virtual environments, and Python
caches are excluded from Git. The repository includes the source spreadsheet,
configuration files, implementation, and tests.

## Checks

```sh
python -m pytest
python -m ruff check src tests
```

The tests use the bundled dataset and small synthetic inputs. Use the full run
and `verify` commands above to validate the complete five-model experiment.
