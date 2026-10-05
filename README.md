# MLOps Lab 2 – Automated Model Training and Versioning with GitHub Actions

![Model Retraining](https://github.com/piyush1420/mlops-lab2/actions/workflows/model_retraining_on_push.yml/badge.svg)
![Daily Retraining](https://github.com/piyush1420/mlops-lab2/actions/workflows/model_calibration.yml/badge.svg)

Lab 2 for IE-7374 MLOps, based on `Github_Labs/Lab2` from the course repo.
Every push to `main` makes GitHub Actions train a Random Forest model, score it,
save the model and its score with a timestamp as the version, and commit both
back to this repo. A second workflow does the same thing every day on a timer.

## Project structure

```
.github/workflows/
    model_retraining_on_push.yml   retrains on every push to main
    model_calibration.yml          retrains every day at midnight UTC
src/
    train_model.py                 loads data, trains the model, logs to MLflow
    evaluate_model.py              scores the model on held-out test data
models/                            versioned models, added by the workflows
metrics/                           versioned F1 scores, added by the workflows
requirements.txt
```

## What I changed from the original lab

### Dataset
- Replaced the random fake data (`make_classification`, 6 unnamed columns) with the **Iris flowers** dataset, which has 4 known columns: sepal length, sepal width, petal length, petal width (cm).
- The model predicts whether a flower is *Iris virginica* (1) or another species (0).

### Fair evaluation
- In the original, `train_model.py` and `evaluate_model.py` each generated new fake data with a random number of rows. A different row count produces data with different hidden patterns, so the model was tested on data unlike what it learned from, and scores were close to guessing.
- Now the data is split once (80% train, 20% test). The model learns from the 80% only, the 20% is saved to `data/`, and `evaluate_model.py` scores the model on those unseen flowers.

### Results

| Version | Data | F1 score |
|---|---|---|
| `20261005025744` | Original (fake data) | 0.34 |
| `20261005031022` | Original (fake data) | 0.53 |
| `20261005033323` | Iris with train/test split | 1.00 |

Iris is a small, easy dataset (30 test flowers), so a perfect score is expected.

### GitHub Actions
- Moved the workflows to `.github/workflows/`, the only folder GitHub runs workflows from, and changed the course-repo paths (`Labs/Github_Labs/Lab2/...`) to this repo's layout.
- Added `permissions: contents: write`, so the workflow is allowed to push the new model back to the repo.
- Set `MLFLOW_ALLOW_FILE_STORE=true`, because the newest MLflow refuses the `./mlruns` folder by default.
- Updated action versions and Python 3.9 to 3.12.
- Daily workflow: fixed a bug where file names made in earlier steps were empty in the commit step (each step is a new shell), and added a "Run workflow" button (`workflow_dispatch`).
- Both workflows now run `git pull --rebase` before pushing, so they don't fail when both run at the same time.

## Run locally (Mac)

```
python3 -m venv lab_02
source lab_02/bin/activate
pip install -r requirements.txt
timestamp=$(date '+%Y%m%d%H%M%S')
python src/train_model.py --timestamp "$timestamp"
python src/evaluate_model.py --timestamp "$timestamp"
cat ${timestamp}_metrics.json
```

To view the MLflow logbook: `mlflow ui --port 5001`, then open http://127.0.0.1:5001
