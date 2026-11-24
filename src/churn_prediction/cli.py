"""Command-line entry points for fetching, running, and verifying experiments."""

import argparse
from pathlib import Path

from .paths import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Five-model IBM Telco churn prediction experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch-assets", help="Download and verify the dataset and frozen encoder"
    )
    run = commands.add_parser(
        "run", help="Fit five models with development CV and a separate holdout"
    )
    verify = commands.add_parser(
        "verify", help="Reconcile a completed run against saved predictions"
    )
    for command in (fetch, run):
        command.add_argument(
            "--data", type=Path, default=DEFAULT_DATA, help="IBM Telco spreadsheet path"
        )
        command.add_argument(
            "--model-dir", type=Path, default=DEFAULT_MODEL, help="Verified MiniLM asset directory"
        )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment protocol JSON")
    run.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Empty directory for generated results"
    )
    run.add_argument(
        "--include-controls",
        action="store_true",
        help="Also fit text-only, shuffled-text, and profile-ID controls",
    )
    verify.add_argument(
        "--results", type=Path, default=DEFAULT_OUTPUT, help="Completed run directory"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-assets":
            from .assets import fetch

            fetch(args)
        elif args.command == "run":
            from .pipeline import run as run_experiment

            run_experiment(args)
        else:
            from .verification import verify as verify_results

            verify_results(args.results)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        parser.exit(1, f"Error: {error}\n")
