# Bitcoin Transaction Fraud Detection

This project trains a LightGBM classifier to estimate the probability that a Bitcoin transaction is illicit. It uses the provided `feat_1` through `feat_165` columns plus a directed incoming-degree graph feature. Labels `0` and `1` are supervised targets; `none` is treated as unlabeled and is excluded from supervised training.

## Setup

From the project root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The data files are expected in `data/`. The student notebook is `data/e1304439.ipynb`.

## Run

Run the complete temporal validation, final fit, and submission generation from the project root:

```bash
python -m src.train
```

The run writes `artifacts/validation_results.csv`, `models/final_model.joblib`, and `submissions/submission.csv`. Validation uses expanding time-step windows and reports ROC-AUC and PR-AUC. Only folds containing both classes in training and validation are scored.

The graph feature currently uses incoming degree only: the count of observed transactions whose outputs flow into the transaction. It intentionally omits outgoing degree because downstream transactions occur later and may not be available at prediction time. Graph visibility and same-time-step edges still require scrutiny against the competition's intended prediction-time information boundary.

Run the focused tests with:

```bash
python -m pytest
```

## Validation and limitations

The test set is later in time than the labeled training data, so random train/test splits are not representative. Temporal validation is the model-selection basis. The precomputed anonymized features may already contain neighbor aggregates; graph features should be retained only if future-period validation supports their value. This initial implementation is a reproducible baseline, not a claim that the model is leaderboard-optimal.

Before submission, verify that the expected sample submission and test rows have the same `index` sequence. The submission writer enforces this ordering and writes exactly `index,target`.

## Feature experiments

Run controlled engineered-feature ablations with:

```bash
python -m src.run_feature_experiments
```

The script compares the supplied features against adding `time_step`, raw or log incoming degree, row-wise summaries of the supplied features, and degree/summary combinations. It uses the same LightGBM settings for four expanding temporal folds and the longer steps 1–21 → 22–35 holdout. Results and normalized gain importances are written under `artifacts/`.

For a permutation-based ranking of the supplied predictors on the longer holdout, run:

```bash
python -m src.run_feature_importance
```

This tests the 30 highest-gain features from the steps 1–21 → 22–35 model by shuffling each across the validation rows and recording the ROC-AUC drop. Correlated features can mask one another's importance, so interpret the ranking as a diagnostic rather than a causal or universal feature ranking.

The downstream-degree variant is diagnostic only: it may use edges from later time steps, so it should not be treated as a deployable feature unless those edges are available at prediction time. The script also runs adversarial validation, training a separate classifier to distinguish labeled historical transactions from Kaggle test transactions while excluding `time_step`. Its AUC measures covariate separability, not fraud-classification performance; top domain features are saved separately.

The dataset CSVs and local notebook are intentionally excluded from source control. Place the competition files under `data/` locally before running the project.