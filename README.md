# Bitcoin Transaction Fraud Detection (Elliptic, BT4012 Kaggle)

A two-stage graph-stacking model (LightGBM in both stages) that estimates the probability
that a Bitcoin transaction is illicit. Only transactions labeled `0` (licit) or `1` (illicit) are used for supervised
training; `none` rows are treated as unlabeled. `time_step` is used only for chronological
splits, never as a model input.

The submission notebook is [`e1304439.ipynb`](e1304439.ipynb). The full experiment record,
with every result and the reasoning behind each decision, is in
[`docs/findings.md`](docs/findings.md).

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
in [`src/graph_stacking.py`](src/graph_stacking.py) (public ROC-AUC **0.97196**):

1. **Stage 1:** a tuned, shallow LightGBM (9 leaves, 500 trees) on the 165 supplied features
   scores labeled training rows out-of-fold (5 folds of consecutive time steps) and test rows
   with the mean of the fold models.
2. **Graph context:** over each transaction's *labeled* neighbours: stage-1 score
   mean/max/in-mean/out-mean, degrees, labeled-neighbour count and share, and
   mean/max/std/in-mean/out-mean of the 25 most influential supplied features. Walk-weighted means (stage-1 score and the same 25 features) over labeled nodes two hops away are added; walks
   may pass through featureless nodes. Label-free payment-chain shape from the full edge list
   (longest payment path into / out of the transaction, peel-chain runs) and contrast features
   (own value minus labeled-neighbour mean for the 25 key features) are added as well.
3. **Stage 2:** the same LightGBM on supplied features plus graph context, with its column
   share scaled by 165/330 so each tree sees about as many columns as a stage-1 tree.
4. **Weighting and seeds:** every fit weights training rows toward recent steps (half-life
   14 steps) and toward rows that look like the test period (covariate-shift importance
   weights from a train-vs-test classifier that excludes the 30 most time-encoding features);
   predictions are averaged over 10 seeds.

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
  and two-hop context (+0.003) took the public score from 0.944 to 0.968. Recency weighting adds
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
| `nbr_stack_rich_2hop_hl14` (+ two-hop context) | 0.96668 |
| `nbr_stack_rich_2hop_prop_hl14` (+ 3-hop score propagation) | 0.96108 |
| `blend_top_graph` (rank blend of three graph models) | 0.96587 |
| `nbr_stack_wide_2hop_hl14` (wide statistics + two-hop) | 0.96814 |
| `nbr_stack_wide_2hop_hl14`, 30 seeds / half-life 21 (stability checks) | 0.96785 / 0.96670 |
| `blend_top4` (rank blend of four graph models) | 0.96686 |
| `nbr_stack_wide_2hop_hl14_iter` (extra round of collective classification) | 0.96698 |
| `nbr_stack_wide_2hop_dir_hl14` (direction-split two-hop) | 0.96787 |
| `nbr_stack_wide_2hop_hl14_fwd` (forward-only stage-1 scores) | 0.96503 |
| `nbr_stack_wide_2hop_hub_hl14` / `_cent_hl14` (hub-weighted means / centrality) | 0.96766 / 0.96759 |
| `nbr_stack_wide_2hop_chain_hl14` (+ peel-chain shape; private pick 2) | 0.96858 |
| `nbr_stack_wide_2hop_chain_shift_hl14` (+ test-similarity importance weights) | 0.9702 |
| `nbr_stack_wide_2hopw_chain_shift_hl14` (+ two-hop means over all 25 features) | 0.97097 |
| `nbr_stack_wide_2hopw_max_chain_shift_hl14` (+ two-hop max score) | 0.97085 |
| `nbr_stack_wide_2hopw_chain_shift_s2reg_hl14` (+ stage-2 column matching) | 0.97190 |
| `…_s2c20_hl14` / `…_s2l8_hl14` (stage 2: fewer columns / stronger L2) | 0.97131 / 0.97136 |
| `…_s2reg_prune95_hl14` (stage 2 pruned to 151 features; private pick 2) | 0.97176 |
| `…_s2reg_contrast_hl14` (+ contrast features, **final model**, private pick 1) | **0.97196** |
| `…_s2reg_retuned_hl14` (settings re-tuned for more capacity) | 0.96207 |

Full commentary for every round is in `docs/findings.md`.

## Experiment scripts

| Script | Purpose |
|---|---|
| `src/run_step1_harness.py` | Re-score baseline vs regularised / feature-group variants on 5 windows × 5 seeds |
| `src/make_submissions.py` | Fit registered candidates or blends on all labeled rows and write `submissions/submission_<name>.csv` |
| `src/run_step3_features.py` | Correlated-cluster ablation, TreeSHAP stability, adversarial feature dropping |
| `src/run_step4_tuning.py` | Optuna tuning (LightGBM / XGBoost / CatBoost) on short-horizon windows; ensemble evaluation |
| `src/run_step5_stress.py` | Stress tests: extra drift, feature noise, withheld fraud-heavy steps, rare positives |
| `src/run_step8_variants.py` | Rounds 3–11: recency weighting, temporal bagging, graph-stacking variants (`evaluate`, `submit`, `blend`) |
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
