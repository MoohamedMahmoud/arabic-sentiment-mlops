# Arabic Sentiment `src/` package

This package is the reusable implementation behind the Arabic sentiment project.

## Pipeline

```text
raw alldata.csv
    -> prepare.py
    -> data/processed/*.parquet
    -> DVC tracks the stage and data version
    -> train.py
    -> MLflow run + logged model
    -> Model Registry
    -> predict.py
    -> FastAPI
```

## Important boundaries

- `prepare.py` contains the actual raw-data cleaning and splitting logic.
- `data.py` only handles data I/O and schema checks.
- `preprocessing.py` is the shared AraBERTv0.2 tokenizer path for training and inference. The production wrapper loads the tokenizer saved with the trained model.
- `model.py` is the reusable model wrapper.
- `train.py` trains one configuration and logs the run/model.
- `tune.py` runs the small controlled sweep.
- `predict.py` loads the registered production alias.
- `api/` is the HTTP validation and API layer.
- `mlflow_utils.py` contains MLflow/Git/DVC helper functions.
- `mlflow_model.py` packages the model behind MLflow's generic PyFunc interface.

The raw dataset is never overwritten by `prepare.py`.

The current data-understanding notebook reports 88,738 cleaned rows. The previously observed DVC artifact had 88,737 rows (70,989/8,874/8,874). This package does not invent an extra deletion to force those numbers to match; run `prepare.py` once and reconcile that one-row difference before committing the new DVC lock.
