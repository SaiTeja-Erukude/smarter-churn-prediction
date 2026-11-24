# Dataset

`raw/telco_customer_churn.xlsx` is the IBM Telco Customer Churn sample, sheet
`Telco_Churn`: 7,043 customers, 33 source columns, and 1,869 churn cases.
IBM describes this as a fictional-company sample.

We obtained the exact spreadsheet from the
[reference repository](https://github.com/Suhasnadh/smarter-churn-prediction/blob/8f2cd61b9cee501bf7011694b76644c20eded5a0/Data/Telco_customer_churn.xlsx).
The source revision is `8f2cd61b9cee501bf7011694b76644c20eded5a0`.

SHA-256:

```text
ce82e608d869df77107e3445ae5f868a950bedc57ff032f14567865aef9a6abb
```

The pipeline retains 19 predictors and maps `Churn Label` to `No = 0`, `Yes = 1`.
It excludes identifiers, geographic fields, `Count`, `Churn Value`, `Churn Score`,
`CLTV`, and `Churn Reason`. It parses tenure, monthly charges, and total charges
as numeric values. Eleven blank total-charge values become missing values;
median imputation is fitted within each training partition.

Synthetic account descriptions are generated from five retained predictors:
contract, tenure, monthly charges, internet service, and technical support
subscription. Generated descriptions and embeddings are written to the run
directory, preserving the source spreadsheet.

After completing the setup in the [main README](../README.md), run this command
from the repository root to verify or restore the spreadsheet and download the
encoder assets:

```sh
python -m churn_prediction fetch-assets
```
