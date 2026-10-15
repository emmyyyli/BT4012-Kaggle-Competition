# Bitcoin Transaction Fraud Detection (Elliptic, BT4012 Kaggle)

A two-stage graph-stacking model (LightGBM in both stages) that estimates the probability
that a Bitcoin transaction is illicit. Only transactions labeled `0` (licit) or `1` (illicit) are used for supervised
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

The final model is defined in [`src/final_config.py`](src/final_config.py) and implemented
in [`src/graph_stacking.py`](src/graph_stacking.py) (public ROC-AUC **0.96668**):

1. **Stage 1:** a tuned, shallow LightGBM (9 leaves, 500 trees) on the 165 supplied features
   scores labeled training rows out-of-fold (5 folds of consecutive time steps) and test rows
   with the mean of the fold models.
2. **Graph context:** over each transaction's *labeled* neighbours: stage-1 score
   mean/max/in-mean/out-mean, degrees, labeled-neighbour count and share, and neighbour means
   of five key features. Walk-weighted means over labeled nodes two hops away are added; walks
   may pass through featureless nodes.
3. **Stage 2:** the same LightGBM on supplied features plus graph context.
4. **Weighting and seeds:** both stages weight training rows toward recent steps (half-life
   14 steps); predictions are averaged over 10 seeds.

Context uses only labeled nodes because, in the test period, only labeled transactions have
features. Every graph feature is therefore computed the same way in training and test.

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
* **Regularisation is the main lever for the base model.** Every hyperparameter search chose
  4–9 leaves or depth 3–4. XGBoost, CatBoost, cross-family ensembles, GraphSAGE and
  pseudo-labeling did not beat a tuned LightGBM.
* **Graph context, computed the same way in training and test, drives the largest gains.**
  Labeled transactions cluster: 57% of an illicit transaction's labeled neighbours are illicit,
  against a 12% base rate. Labeled-neighbour stacking (+0.010), neighbour feature means (+0.005)
  and two-hop context (+0.003) took the public score from 0.944 to 0.967. Recency weighting adds
  about +0.005 independently.

| Kaggle submission | Public ROC-AUC |
|---|---|
| Original LightGBM (+ `time_step`, `in_degree`) | 0.94246 |
| 0.7 LightGBM + 0.3 GraphSAGE | 0.91059 |
| `shallow_all` (regularised LightGBM, 10 seeds) | 0.94267 |
| 50/50 rank blend all-features + aggregated-only | 0.92624 |
| `shallow_agg` (aggregated features only) | 0.88023 |
| `tuned_lgbm_all` (best non-graph model) | 0.94417 |
| `tuned_lgbm_all` minus five test-drifted features | 0.94365 |
| `tuned_lgbm_all_no_feat2` | 0.93673 |
| `halflife_14` (recency weights) | 0.95066 |
| `nbr_stack` (labeled-neighbour stacking) | 0.95389 |
| `nbr_stack_rich` (+ neighbour features, structure) | 0.95871 |
| `nbr_stack_rich_hl14` (+ recency) | 0.96342 |
| `nbr_stack_wide_hl14` (25-feature neighbour statistics) | 0.96506 |
| `nbr_stack_rich_2hop_hl14` (+ two-hop context, **final model**) | **0.96668** |

Full commentary for every round is in `report/findings.md`.

## Experiment scripts

| Script | Purpose |
|---|---|
| `src/run_step1_harness.py` | Re-score baseline vs regularised / feature-group variants on 5 windows × 5 seeds |
| `src/make_submissions.py` | Fit registered candidates or blends on all labeled rows and write `submissions/submission_<name>.csv` |
| `src/run_step3_features.py` | Correlated-cluster ablation, TreeSHAP stability, adversarial feature dropping |
| `src/run_step4_tuning.py` | Optuna tuning (LightGBM / XGBoost / CatBoost) on short-horizon windows; ensemble evaluation |
| `src/run_step5_stress.py` | Stress tests: extra drift, feature noise, withheld fraud-heavy steps, rare positives |
| `src/run_step8_variants.py` | Rounds 3–6: recency weighting, temporal bagging, graph-stacking variants (`evaluate`, `submit`, `blend`) |
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
  rows (about 270 illicit), so a single public score has a standard error of about 0.008.
  Differences between closely related models on the same rows are more precise (about
  0.001–0.002).
