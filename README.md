# Bitcoin Transaction Fraud Detection (Elliptic, BT4012 Kaggle)

Gradient-boosted tree models that estimate the probability that a Bitcoin transaction is
illicit. Only transactions labeled `0` (licit) or `1` (illicit) are used for supervised
training; `none` rows are treated as unlabeled. `time_step` is used only for chronological
splits, never as a model input.

The submission notebook is [`e1304439.ipynb`](e1304439.ipynb). The full experiment record,
with every result and the reasoning behind each decision, is in
[`report/findings.md`](report/findings.md).

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the competition files (`train.csv`, `test.csv`, `txs_edgelist.csv`,
`sample_submission.csv`) in `data/`. The data files are not under version control.

## Reproduce the final submission

```bash
python -m src.train                  # validate, refit with 10 seeds, write submissions/submission_final.csv
python -m src.build_notebook         # regenerate and execute e1304439.ipynb (same pipeline)
python -m pytest                     # unit tests
```

The final model is defined in [`src/final_config.py`](src/final_config.py): a tuned,
shallow LightGBM (9 leaves, 500 trees) on the 165 supplied features minus the five features
that drift most between the training and test periods (`feat_103`, `feat_136`, `feat_101`,
`feat_2`, `feat_139`). It is averaged over 10 seeds.

## Key findings

* **The test set is the labeled transactions of steps 36–49.** The 46,668 edge-list nodes
  without features are the unlabeled transactions of those steps. About 66% of test-node
  edges therefore lead to featureless nodes, against 0% in training. This mismatch explains
  why the earlier GraphSAGE blend dropped the Kaggle score to 0.91059.
* **Single-seed, single-split scores are too noisy to select models** (seed spread up to
  0.02 ROC-AUC). All comparisons use the multi-window, multi-seed harness in
  [`src/evaluation.py`](src/evaluation.py).
* **Short-horizon validation windows predict Kaggle; long-horizon windows do not.**
  Aggregated (neighbourhood) features looked best on long windows, but scored 0.880 on
  Kaggle against 0.943 for all features. The aggregated features drift about twice as much
  into the test period as within training.
* **Regularisation is the main lever.** Every hyperparameter search chose 4–9 leaves or depth
  3–4. XGBoost, CatBoost, cross-family ensembles, GraphSAGE and pseudo-labeling did not beat a
  tuned LightGBM.

| Kaggle submission | Public ROC-AUC |
|---|---|
| Original LightGBM (+ `time_step`, `in_degree`) | 0.94246 |
| 0.7 LightGBM + 0.3 GraphSAGE | 0.91059 |
| `shallow_all` (regularised LightGBM, 10 seeds) | 0.94267 |
| 50/50 rank blend all-features + aggregated-only | 0.92624 |
| `shallow_agg` (aggregated features only) | 0.88023 |

Round-2 scores for the tuned models are recorded in `report/findings.md`.

## Experiment scripts

| Script | Purpose |
|---|---|
| `src/run_step1_harness.py` | Re-score baseline vs regularised / feature-group variants on 5 windows × 5 seeds |
| `src/make_submissions.py` | Fit registered candidates or blends on all labeled rows and write `submissions/submission_<name>.csv` |
| `src/run_step3_features.py` | Correlated-cluster ablation, TreeSHAP stability, adversarial feature dropping |
| `src/run_step4_tuning.py` | Optuna tuning (LightGBM / XGBoost / CatBoost) on short-horizon windows; ensemble evaluation |
| `src/run_step5_stress.py` | Stress tests: extra drift, feature noise, withheld fraud-heavy steps, rare positives |
| `src/run_step6_extensions.py` | Directed GraphSAGE with test-consistent masking (`src/gnn_v2.py`); pseudo-labeling |
| `src/run_feature_experiments.py`, `src/run_feature_importance.py`, `src/run_lgbm_robustness_experiments.py`, `src/run_graph_benchmark.py`, `src/gt_ssl_extractor.py` | Earlier single-holdout experiments (degree features, permutation importance, recency weighting, first GraphSAGE, GT-SSL embedding pilot) |

Experiment outputs are written under `artifacts/`. Per-row prediction dumps (`*.csv.gz`) are
not tracked; summary tables are.

## Validation design

* Forward windows only: train on steps ≤ t, validate on a later block. Report windows are
  ≤14→15–35, ≤21→22–35, ≤21→22–28, ≤25→26–35 and ≤28→29–35.
* Five seeds per window; the headline metric is the seed-bagged ROC-AUC, with per-step AUC
  and a paired bootstrap of the difference against a baseline.
* Adoption rule: the mean AUC across windows must improve, and no window may lose more than
  0.01.
* Hyperparameters are tuned on short windows that validate no later than step 28. Steps
  29–35 are held out to confirm the choices.
* Limitation: no window inside steps 1–35 reproduces the train→test drift, so final choices
  between close candidates rely on the Kaggle public leaderboard. It covers about 4,600
  rows, so differences below about 0.005 are noise.
