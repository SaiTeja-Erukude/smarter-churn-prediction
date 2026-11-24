"""Verify a completed run through the installed package."""

import sys
from churn_prediction.cli import main

if __name__ == "__main__":
    main(["verify", *sys.argv[1:]])
