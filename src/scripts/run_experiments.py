"""Run the installed package from any working directory."""

import sys
from churn_prediction.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
