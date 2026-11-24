"""Fetch verified public assets through the installed package."""

import sys
from churn_prediction.cli import main

if __name__ == "__main__":
    main(["fetch-assets", *sys.argv[1:]])
