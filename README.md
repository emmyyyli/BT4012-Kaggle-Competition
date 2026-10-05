# Bitcoin Transaction Fraud Detection

This project trains a LightGBM classifier to estimate the probability that a Bitcoin transaction is illicit. The default baseline uses the provided `feat_1` through `feat_165` columns; it excludes `time_step` and graph-degree features. Labels `0` and `1` are supervised targets; `none` is treated as unlabeled and is excluded from supervised classification training.

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

The feature ablation runner separately tests incoming degree. Outgoing degree is retained only as a diagnostic because downstream transactions may not be available at prediction time. Graph visibility and same-time-step edges still require scrutiny against the competition's intended prediction-time information boundary.

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

## GT-SSL-inspired embedding pilot

Run the unsupervised embedding pilot and matched 1–21 → 22–35 LightGBM holdout comparison with:

```bash
python -m src.gt_ssl_extractor
```

The default `--pretrain-scope causal` trains on steps 1–21 and evaluates the long holdout. To separately test transductive pretraining over all available graph nodes/features (without any labels), run `python -m src.gt_ssl_extractor --pretrain-scope full_transductive`. The latter includes validation-window attributes during SSL and is not a causal time-forward estimate; report it as a separate setting.

The pilot uses length-8 directed restarted walks, a feature-cosine similarity proxy, degree penalization, a structure-biased Transformer, masked continuous-feature reconstruction, and an edge-drop/feature-noise InfoNCE task. It has no label input and no time-step feature or time-aware walk transition. Because the anonymized feature table does not identify transaction amount, this is a documented GT-SSL-inspired approximation, not a faithful reproduction of the paper's amount weighting. The pilot defaults to 64-dimensional embeddings, two Transformer layers, and five SSL epochs for feasibility; these are experiment settings, not selected production settings.

For the long-holdout experiment, self-supervised training and scaler fitting use only graph nodes assigned to steps 1–21, including unlabeled nodes in that period, and training edges are restricted to those steps. Embeddings are then extracted for feature-table rows using their node features and the available directed graph. This is a transductive inference setup for the holdout and assumes validation-period graph structure is available when scoring those nodes; results must not be interpreted as a strict no-future-topology experiment. Nodes without feature rows use zero-valued attributes and the `is_missing_features` indicator and are excluded from the reconstruction targets.

The script reports matched baseline and augmented LightGBM ROC-AUC/PR-AUC for the full holdout and its two halves, plus a paired bootstrap interval for the full-holdout ROC-AUC difference. It saves pretraining losses, validation predictions, and an ID-keyed compressed embedding file under `artifacts/`. It does not change the regular training or submission pipeline. Treat an improvement on this single holdout as a pilot result requiring seed and time-window replication before adoption.

### Pilot result (64 dimensions, 5 epochs)

The supplied-feature LightGBM baseline scored **0.88374 ROC-AUC** on steps 22–35. Concatenating 64-dimensional embeddings decreased the score on both causal pretraining seeds: **0.84116** (seed 42) and **0.84306** (seed 7). Full transductive pretraining scored **0.86177** (seed 42), also below baseline. The paired row-bootstrap 95% interval for the ROC-AUC difference was [-0.04762, -0.03726] for causal seed 42, [-0.04524, -0.03626] for causal seed 7, and [-0.02496, -0.01901] for full transductive seed 42. These intervals condition on the fitted models and validation rows; they do not capture variation across model fits.

The augmented model was worse on both subwindows, with the largest deficit on steps 29–35. These pilot settings therefore do not justify adding GT-SSL embeddings to the LightGBM submission. The outcome does not rule out the paper's full method: this pilot omits amount and temporal walk weights, pseudo-labeling, Louvain consensus, and uses a smaller encoder than the paper. The issue is that the proposed self-supervised-only extractor, as tested with available inputs and leakage-conscious validation, did not improve the metric.

## LightGBM resilience experiments

Run matched temporal experiments for sample recency weighting and removing the distribution-shift-heavy features `feat_101` and `feat_103`:

```bash
python -m src.run_lgbm_robustness_experiments
```

The runner compares the unweighted baseline, exponential recency half-lives of 3, 7, and 14 time steps, dropping either feature individually, and dropping both. Sample weights are normalized to mean one within each training split and are passed to LightGBM alongside the existing class weighting. Every variant uses the same model settings and time-forward folds, including the steps 1–21 → 22–35 holdout; two seeds are run by default. Fold-level metrics, summaries, and validation predictions are saved under `artifacts/`. These remain experiments and do not alter the production submission pipeline.

### Initial results (seeds 7 and 42)

No candidate has yet shown a consistent improvement across seeds. On the long holdout, mean ROC-AUC across the two seeds was **0.86986** for baseline, **0.87478** for a 14-step recency half-life, **0.87499** after dropping `feat_103`, and **0.87367** after dropping `feat_101`. However, each of those three candidates improved one seed and reduced the other; paired AUC deltas ranged from -0.0080 to +0.0178 for half-life 14 and from -0.0074 to +0.0177 for dropping `feat_103`. Dropping both features was worse on average (0.86529). A 3-step half-life was clearly harmful on the long holdout (0.84298 mean AUC).

On the latest expanding fold (train through step 28, validate 29–35), all variants were close in ROC-AUC. Three-step weighting improved the mean by about 0.00075 across seeds, but that small gain did not transfer to the longer 1–21 → 22–35 holdout. These results are exploratory: the two seeds are too few to select a replacement model, and the historical/test gap remains unobserved in labels. Keep the unweighted 165-feature baseline until further chronological repeats support a stable gain.