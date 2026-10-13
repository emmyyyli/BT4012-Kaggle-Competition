# Findings log — BT4012 Elliptic illicit-transaction competition

Running record of experiments and evidence, kept for writing the final report.
Each entry states what was run, the numbers, and what the numbers do and do not show.
All ROC-AUC values are on labeled validation rows from the training period (steps 1–35)
unless a Kaggle score is stated explicitly.

Started 2026-10-10. Competition closes 2026-10-24 23:55 SGT; report due 2026-10-25.

---

## 0. Starting point (before this round of work)

| Submission | Model | Kaggle public ROC-AUC |
|---|---|---|
| original | LightGBM, 165 features + `time_step` + `in_degree`, row subsampling inactive | **0.94246** |
| `submission_sage_blend.csv` | 0.7 × corrected LightGBM + 0.3 × GraphSAGE | 0.91059 |
| `submission.csv` | corrected LightGBM alone (no `time_step`/`in_degree`, `subsample_freq=1`) | not yet scored |

Earlier local results (`README.md`, `artifacts/*.csv`): supplied features alone scored 0.88374
on the steps 1–21 → 22–35 holdout (seed 42). Adding `time_step`, `in_degree`, row summaries
or GT-SSL embeddings did not beat that number, and neither did recency weighting nor
dropping `feat_101`/`feat_103`.

---

## 1. Data facts established 2026-10-10

**1.1 The test set is exactly the labeled transactions of steps 36–49.**
31,235 labeled training rows + 15,329 test rows = 46,564, which is the documented labeled
total of the original Elliptic dataset (4,545 illicit + 42,019 licit). The test file holds no
unlabeled transactions.

**1.2 The 46,668 edge-list nodes missing from the feature tables are the unlabeled
transactions of steps 36–49.** Every one is assigned (via its connected component) to a
step between 36 and 49 (per step: 36: 4,685 · 37: 2,808 · 38: 2,135 · 39: 1,577 · 40: 3,270 ·
41: 4,210 · 42: 4,986 · 43: 3,693 · 44: 3,384 · 45: 4,377 · 46: 2,807 · 47: 4,275 · 48: 2,483 ·
49: 1,978). 157,101 + 46,668 = 203,769, the original node count.

**1.3 Train and test graph neighbourhoods differ systematically.** Of the 36,969 edges that
touch a test node, 65.6% connect to a node with no features. For edges touching a training
node the share is 0%. Any model that reads neighbour features (GraphSAGE, GT-SSL embeddings)
therefore sees blank neighbours only at test time. This mechanism concretely explains the
GraphSAGE blend's drop (0.94246 → 0.91059), and it was already in place before the step-43
market shutdown. The comparison is still confounded (see §0), so it is a hypothesis to test.

**1.4 Illicit share varies strongly by step** (0.4%–36% of labeled rows; e.g. step 9: 31.9%,
step 13: 36.0%, step 5: 0.4%). Pooled AUC across steps therefore mixes within-step ranking
with between-step score shifts.

**1.5 Local vs aggregated features.** The Elliptic documentation lists 94 local features (the
first is the time step) and 72 one-hop aggregates. With `time_step` split out, `feat_1`–`feat_93`
are the local features and `feat_94`–`feat_165` the aggregates. I tried to verify this by
rebuilding aggregates from neighbour max/min/mean/std of local features over steps 1–35 (where
neighbourhoods are complete). The result was inconclusive: median best |corr| was 0.24 for
aggregated columns and 0.47 for local columns, and only 4 of the 72 exceeded 0.9. The mapping
therefore rests on the documentation, not on reconstruction.

**1.6 Seed noise is as large as most effects tested so far.** On the 1–21 → 22–35 holdout,
the baseline LightGBM over seeds 0–4 gave mean 0.8753, sd 0.0103 (seeds 42 / 7 gave 0.884 /
0.856). Most earlier ablation deltas (±0.005) are inside that noise.

## 2. First horizon probe (scratch script, 3-seed bag, 2026-10-10)

| Features / model | ≤14→15–35 | ≤21→22–35 | ≤21→22–28 | ≤25→26–35 | ≤28→29–35 |
|---|---|---|---|---|---|
| all 165, baseline LGBM | 0.897 | 0.874 | 0.940 | 0.996 | 0.997 |
| all 165, shallow LGBM | 0.888 | 0.905 | 0.956 | 0.995 | 0.996 |
| aggregated 72, baseline LGBM | 0.942 | 0.901 | 0.941 | 0.953 | 0.951 |
| aggregated 72, shallow LGBM | 0.953 | 0.922 | 0.949 | 0.954 | 0.953 |
| local 93, shallow LGBM | 0.875 | 0.879 | 0.945 | 0.994 | 0.995 |
| Random Forest, all / aggregated | 0.907 / 0.907 | 0.825 / 0.838 | 0.892 / 0.890 | 0.991 / 0.928 | 0.993 / 0.931 |

Shallow LGBM = `num_leaves=8, min_child_samples=100, colsample_bytree=0.5, reg_lambda=5`.
Per-time-step percentile ranking of features (to remove level drift) **hurt**: 0.845 vs 0.877
on 1–21 → 22–35. Random Forest underperformed LightGBM on every window and was dropped.

Reading: windows that validate soon after training favour all features (local features carry
short-range signal). Windows that span or follow steps 22–25 favour aggregated-only and
shallow trees. The Kaggle test spans 1–14 steps after the training cut, so it falls between
these regimes. That motivates a fixed rank blend of the two rather than choosing one.

---

## 3. Step 1 — evaluation harness (2026-10-10)

**Code:** `src/evaluation.py` (harness), `src/model_zoo.py` (model factories),
`src/run_step1_harness.py` (this experiment), tests in `tests/test_evaluation.py`.

**Protocol.** Labeled rows only; `time_step` is used for splitting and never as a feature.
- Five forward windows: train ≤14 → validate 15–35, ≤21 → 22–35, ≤21 → 22–28, ≤25 → 26–35,
  ≤28 → 29–35.
- Five seeds (0–4) per window. The headline metric is the seed-bagged ROC-AUC (mean of the
  five seeds' probabilities), which matches how the submission averages seeds.
- AUC per validation time step, plus a paired row bootstrap (300 draws) of the bagged AUC
  difference against the baseline.
- **Adoption rule (fixed before running):** mean bagged AUC across windows must improve, and
  no window may lose more than 0.01.
- For tuning, `TUNING_WINDOWS` (validation ≤ step 28) and `CONFIRM_WINDOWS` (validation 29–35)
  are kept apart so tuned choices are checked on steps that took no part in selecting them.

**Results** (`artifacts/harness/step1_*.csv`), seed-bagged ROC-AUC:

| Candidate | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | Mean | Adopt? |
|---|---|---|---|---|---|---|---|
| baseline_all (current production) | 0.8981 | 0.9413 | 0.8770 | **0.9965** | **0.9968** | 0.9419 | — |
| shallow_all | 0.8890 | 0.9537 | 0.8998 | 0.9949 | 0.9960 | 0.9467 | **yes** (+0.0047; worst −0.0090) |
| shallow_agg | **0.9529** | 0.9482 | 0.9213 | 0.9547 | 0.9536 | 0.9462 | no (worst −0.0432) |
| baseline_agg | 0.9420 | 0.9397 | 0.8994 | 0.9533 | 0.9523 | 0.9374 | no |
| shallow_local | 0.8760 | 0.9433 | 0.8736 | 0.9944 | 0.9950 | 0.9364 | no |
| **blend_shallow_all_agg** (50/50 rank) | 0.9412 | **0.9625** | **0.9258** | 0.9861 | 0.9876 | **0.9606** | no — worst window −0.0103, 0.0003 past the limit |

Mean seed-to-seed sd of single-seed AUC: 0.0039 (baseline) vs 0.0032 (shallow_all), 0.0016
(shallow_agg) and 0.0020 (blend), so the regularised and aggregated models are also more stable.

Paired bootstrap 95% CI for the blend's ΔAUC vs baseline: ≤14→15–35 [+0.039, +0.049];
≤21→22–28 [+0.016, +0.027]; ≤21→22–35 [+0.044, +0.054]; ≤25→26–35 [−0.013, −0.008];
≤28→29–35 [−0.011, −0.007]. The trade-off is clear-cut in both directions.

**Per-step view (≤21 → 22–35).** Most steps are near-perfect for every model. The pooled
deficit comes from four fraud-heavy steps: 24 (137 illicit), 29 (329), 31 (106) and 32 (342).
On those, aggregated features help a lot. Step 29: baseline 0.677 → shallow_agg 0.815, blend
0.810. Step 32: 0.782 → 0.921 / 0.876. These look like new illicit patterns that local
transaction attributes from earlier steps do not describe. The one-hop aggregates still
transfer, presumably because they describe the payment neighbourhood rather than the
transaction itself.

**Interpretation.** Models trained close to the validation period do best with all features.
Models extrapolating 7+ steps across a pattern change do best with aggregates. The Kaggle test
covers 1–14 steps beyond step 35, so it includes both short-range and long-range rows. The
blend gives up about 0.01 at short range to gain 0.02–0.05 at long range. Its expected effect
on Kaggle is positive but unproven; that is the main question for the next submissions.

## 4. Step 2 — submissions generated for diagnosis (2026-10-10)

`src/make_submissions.py` refits each candidate on all 31,235 labeled rows with 10 seeds and
writes `submissions/submission_<name>.csv`. Blends average per-model percentile ranks
of the test predictions (the output is a valid score in [0,1]; ROC-AUC depends only on order).

Spearman rank correlation between test-set submissions:

| | submission (corrected LGBM) | shallow_all | shallow_agg | blend | sage_blend (scored 0.91059) |
|---|---|---|---|---|---|
| submission (corrected LGBM) | 1 | 0.922 | 0.710 | 0.889 | 0.645 |
| shallow_agg | 0.710 | 0.682 | 1 | 0.916 | 0.484 |

The GraphSAGE blend reorders the test set far more than any tabular change (ρ = 0.645 with its
own LightGBM component weighted 70%). That fits §1.3: GraphSAGE met blank neighbours only at
test time.

**Kaggle submission queue** (scores to be filled in when submitted; public LB = ~30% of test,
~4,600 rows, so treat differences < ~0.005 as noise):

| # | File | Purpose | Public AUC |
|---|---|---|---|
| 1 | `submission.csv` | corrected pure LightGBM: isolates the GraphSAGE effect vs 0.91059 | not submitted (held back, see §6) |
| 2 | `submission_shallow_all.csv` | does regularisation alone transfer? | **0.94267** (2026-10-10) |
| 3 | `submission_shallow_agg.csv` | does the long-horizon aggregated effect transfer? | **0.88023** (2026-10-10) |
| 4 | `submission_blend_shallow_all_agg.csv` | short + long horizon blend (leading candidate) | **0.92624** (2026-10-10) |

## 5. Step 3 — feature ablation and stability (2026-10-10)

**Code:** `src/run_step3_features.py`; outputs in `artifacts/features/`. Reference model is
`shallow_all`; report windows; 3 seeds (0–2).

**5.1 Grouped (correlated-cluster) ablation.** The 165 features form 65 clusters
(Spearman |ρ|, average linkage, distance threshold 0.3; median size 2, largest 15). Each
cluster was dropped in turn.
- Two clusters matter far more than any other:
  - cluster 2, 15 local features (feat_59–61, 65–67, 77–79, 83–85, 89–91): dropping it costs
    −0.0218 mean AUC (worst window −0.060);
  - cluster 4, 10 local features (feat_23–55, odd numbers): dropping it costs −0.0056
    (worst −0.026).
  The earlier single-column permutation leader `feat_65` sits in cluster 2. Its correlated
  partners hid most of the group's importance, which is why grouped ablation was needed.
- Dropping **`feat_2` alone** gave the largest gain: +0.0046 mean, improving 4 of 5 windows,
  worst −0.0019. `feat_3` alone: +0.0011. Both pass the adoption rule individually.
  Caution: 65 drops were tested, so the best of them is selected from noise to some degree.
  `feat_2` is still a strong lead because it is also the most important feature by SHAP (see
  5.2): the model leans on it heavily, yet removing it helps later windows. That pattern fits
  a relationship that changes over time.
- Most other clusters change the mean by less than ±0.002, which is within noise.

**5.2 Stability of TreeSHAP importance** (tuning windows only: validation ≤ step 28).
Most consistently important features (minimum share across windows): feat_2 (5.2–7.7%),
feat_47, feat_41, feat_58, feat_59, feat_90, feat_53, feat_132. Feature subsets selected from
this ranking were scored on windows including the untouched confirmation window ≤21 → 29–35:

| Subset | mean (6 windows) | ≤21→29–35 (confirm) | ≤28→29–35 (confirm) |
|---|---|---|---|
| shallow_all (165) | 0.9329 | 0.8577 | 0.9961 |
| stable_top20 | **0.9386** | **0.8779** | 0.9930 |
| stable_top60 / top120 | 0.9322 / 0.9331 | 0.8612 / 0.8588 | 0.9957 / 0.9960 |
| drop top-20% most variable (CV) | 0.9250 | 0.8323 | 0.9960 |

A compact 20-feature model generalises somewhat better at long range (+0.020 on the
confirmation window) and slightly worse at short range (−0.003), the same trade-off as the
aggregated-only model but smaller. Dropping the most variable features hurt. The gains are
modest relative to the aggregated-feature effect in §3.

**5.3 Adversarial (time-separating) features.** In every window the features that best
separate training-period rows from validation-period rows are feat_103, feat_101, feat_136,
then feat_53/feat_52/feat_88. Dropping the top-k separating features per window (k chosen from
the window's own train-vs-validation separation; no fraud labels used):

| k dropped | 0 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| mean AUC (5 windows) | **0.9480** | 0.9472 | 0.9446 | 0.9436 | 0.9456 | 0.9425 |

No depth of adversarial dropping helps; the idea is rejected. The drifting features still carry
fraud signal, and trees that split on them do not appear to be the failure mode. Training rows
vs Kaggle test rows are best separated by feat_103, feat_136, feat_101, **feat_2**, feat_139,
feat_100, feat_52, feat_60, feat_66, feat_89. `feat_2` being 4th here is independent support for
the drop-`feat_2` lead in 5.1: the model's most-used feature is also one of the most drifted.

**Carried into Step 4:** variants that drop `feat_2`, and the stable top-20 subset, as extra
feature sets alongside all 165 and aggregated-only.

## 6. Kaggle results, round 1 (2026-10-10) — the long-horizon hypothesis fails on test

| Submission | Public ROC-AUC | Mean of short windows (≤25→26–35, ≤28→29–35) | Mean of long windows (≤14→15–35, ≤21→22–35) |
|---|---|---|---|
| original (165 + `time_step` + `in_degree`) | 0.94246 | — | — |
| shallow_all | **0.94267** | 0.9955 | 0.8944 |
| blend_shallow_all_agg (50/50 rank) | 0.92624 | 0.9869 | 0.9335 |
| shallow_agg | 0.88023 | 0.9542 | 0.9371 |
| sage_blend (earlier) | 0.91059 | — | — |

**What it shows.**
- On the test period the ordering is all-features > blend > aggregated-only, with
  large gaps (0.943 / 0.926 / 0.880). Only the *short* validation windows rank the three
  models this way. The long windows ranked them in reverse. The Step 1 blend decision should
  have followed the adoption rule, which narrowly rejected the blend; the long-window gains did
  not carry over.
- shallow_all (no `time_step`, no `in_degree`, regularised, 10-seed bag) ties the original
  0.94246 within public-LB noise (~±0.005). Removing `time_step`/`in_degree` therefore did not
  cost test AUC, and the earlier drop to 0.91059 is attributable to the GraphSAGE component.
  For the same reason `submission.csv` was not spent: the question it answers is now settled.

**Why aggregated features fail on test: drift beyond step 35 is concentrated in them.**
Median shift per feature in units of the reference period's IQR (labeled rows):

| Comparison | Local: share of features shifting > 1 IQR | Aggregated: share > 1 IQR | Aggregated: mean shift |
|---|---|---|---|
| ≤21 → 22–35 (long validation) | 0.0% | 11.1% | 0.32 |
| ≤28 → 29–35 (short validation) | 0.0% | 9.7% | 0.28 |
| ≤35 → test 36–49 | 1.1% | **19.4%** | **0.44** |

The aggregated features drift about twice as much into the test period as within any
validation window. Local features hardly move. Five of the six strongest train-vs-test
separators are aggregated features (feat_103, feat_136, feat_101, feat_139, feat_100; §5.3).
Validation inside steps 1–35 cannot reproduce test-period drift of this size, which is why
the long-window validation misled. This is the main limitation of local validation for this
competition, and an important point for the report.

**Changes made in response.**
- `TUNING_WINDOWS` became short-horizon (≤14→15–21, ≤18→19–25, ≤21→22–28), and
  `CONFIRM_WINDOWS` became the two short windows that matched Kaggle's ranking (≤25→26–35,
  ≤28→29–35). Steps 26–28 overlap the last tuning window; 29–35 are fully held out.
- Step 4 now tunes all-feature, all-without-`feat_2` and local-only models. Local features
  are the most stable into the test period. The ensemble draws diversity from model families
  (LightGBM/XGBoost/CatBoost) instead of feature groups.
- A new variant drops the five most test-drifted features (`all_no_test_drift5`). Window
  validation cannot judge it (the drift is not present in steps 1–35); only Kaggle can.

## 7. Step 6 — graph model and pseudo-labeling (2026-10-10)

**Code:** `src/gnn_v2.py`, `src/run_step6_extensions.py`; outputs in `artifacts/extensions/`.
Report windows, 3 seeds; reference `shallow_all`.

**7.1 Directed GraphSAGE under test-consistent masking.** Separate in/out-neighbour
aggregation, LayerNorm instead of BatchNorm, scaler fitted on training-period nodes, fixed
200 epochs. Two regimes:
- *masked*: unlabeled nodes blank in every step, the same rule as the test period;
- *old_regime*: unlabeled nodes keep their features in training steps and are blank only in
  validation steps, which is what the first GraphSAGE effectively faced at test time.

| Candidate | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | mean |
|---|---|---|---|---|---|---|
| shallow_all | 0.8884 | **0.9557** | **0.9048** | 0.9948 | 0.9961 | **0.9480** |
| sage_masked | 0.8236 | 0.9028 | 0.8226 | 0.9741 | 0.9823 | 0.9011 |
| sage_old_regime | 0.8508 | 0.9147 | 0.8440 | 0.9638 | 0.9725 | 0.9091 |
| shallow_all + 0.2 sage_masked | 0.8859 | 0.9497 | 0.8946 | 0.9962 | 0.9978 | 0.9449 |
| shallow_all + 0.2 sage_old_regime | 0.8898 | 0.9522 | 0.8985 | **0.9966** | **0.9979** | 0.9470 |

- GraphSAGE alone is 0.04–0.05 below LightGBM on average, consistent with Weber et al. (2019),
  where tree models on the supplied features beat GCNs.
- Masking lowers GraphSAGE on long windows (its neighbours carry less information) but raises
  it on the short windows (0.974/0.982 vs 0.964/0.973). This confirms that the blank-neighbour
  mismatch costs the old-style model accuracy when it is scored under test conditions.
- A 0.2-weight blend gains about +0.0015 on the two short windows and loses up to 0.010 on
  longer ones. It fails the adoption rule and is not pursued. The earlier 0.91059 result is a
  further reason not to spend submissions on it.

**7.2 Pseudo-labeling** (unlabeled rows of the training steps only; confident predictions
added at weight 0.5). Pools averaged 76.8k unlabeled rows: the strict thresholds (≥0.95 / ≤0.02)
added about 1.9k positives and 40.7k negatives; the loose ones (≥0.80 / ≤0.10) about 4.4k and
56.4k.

| Candidate | mean | worst window | Δ vs shallow_all |
|---|---|---|---|
| shallow_all | 0.9480 | 0.8884 | — |
| pseudo_strict | 0.9176 | 0.8272 | −0.0303 |
| pseudo_loose | 0.9193 | 0.8363 | −0.0286 |

Pseudo-labeling hurt every window; rejected. A plausible reason: Elliptic's unlabeled
transactions are not a random subset of labeled ones (labels exist mainly where an entity could
be attributed), so confident pseudo-labels feed the model a shifted population. The large
pseudo-negative volume also swamps the 3.6k true positives. This matches Lorenz et al. (2020),
who found that unlabeled-data methods underperform supervised trees on Elliptic.

## 8. Step 4 — hyperparameter search and ensembles (2026-10-10)

**Code:** `src/run_step4_tuning.py`; outputs in `artifacts/tuning/` (`best_params.json`,
`*_trials.csv`, `step4_*.csv`, `blends.json`).

**Search.** Optuna TPE sampler, seed 0. Objective: seed-bagged ROC-AUC (2 seeds) averaged over
the short-horizon tuning windows ≤14→15–21, ≤18→19–25, ≤21→22–28. Trials: LightGBM 40,
XGBoost 25, CatBoost 15 per feature set. The LightGBM search was started from the hand-picked
shallow setting.

| Study | Best tuning AUC | Selected settings (abridged) |
|---|---|---|
| lgbm / all | **0.9724** | 9 leaves, 500 trees, lr 0.025, min_child 49, colsample 0.64, λ 2.0 |
| lgbm / all minus feat_2 | 0.9713 | 4 leaves, 1200 trees, lr 0.015, min_child 74, colsample 0.50 |
| lgbm / local | 0.9666 | 9 leaves, 700 trees, lr 0.061, min_child 201 |
| xgb / all | 0.9698 | depth 4, 500 trees, lr 0.020, colsample 0.57 |
| xgb / all minus feat_2 | 0.9690 | depth 3, 800 trees, lr 0.012 |
| xgb / local | 0.9615 | depth 4, 1200 trees, lr 0.065 |
| catboost / all | 0.9672 | depth 3, 800 iterations, lr 0.028, rsm 0.43 |

The run hit the 2-hour limit for background jobs before the CatBoost studies on all-minus-feat_2
and local features. They were not rerun: CatBoost was the weakest family on all features, and
local-only models trailed in both other families. Every search preferred shallow trees (4–9
leaves, depth 3–4), which confirms that regularisation is the main lever.

**Evaluation** (5 seeds; report windows; the two short windows ≤25→26–35 and ≤28→29–35 are
the ones whose ranking matched Kaggle, §6):

| Candidate | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | mean | Δ vs shallow_all (min window) |
|---|---|---|---|---|---|---|---|
| tuned_lgbm_all_no_feat2 | 0.9180 | 0.9582 | 0.9149 | 0.9929 | 0.9943 | **0.9557** | +0.0090 (−0.0020) |
| ensemble lgbm+xgb, no feat_2 | 0.9134 | 0.9580 | 0.9153 | 0.9907 | 0.9930 | 0.9541 | +0.0074 (−0.0042) |
| **tuned_lgbm_all_no_test_drift5** | 0.9129 | 0.9546 | 0.9087 | **0.9967** | **0.9972** | 0.9540 | +0.0073 (**+0.0009**) |
| tuned_lgbm_all | 0.9003 | **0.9590** | 0.9119 | 0.9953 | 0.9961 | 0.9525 | +0.0058 (+0.0001) |
| 0.9 lgbm_all + 0.1 lgbm_local | 0.8965 | 0.9590 | 0.9114 | 0.9956 | 0.9964 | 0.9518 | +0.0051 (+0.0003) |
| ensemble lgbm+xgb+cat, all | 0.9002 | 0.9551 | 0.9047 | 0.9946 | 0.9956 | 0.9501 | +0.0034 (−0.0004) |
| tuned_catboost_all | 0.9200 | 0.9472 | 0.8912 | 0.9943 | 0.9950 | 0.9495 | +0.0029 (−0.0085) |
| shallow_all (Kaggle 0.94267) | 0.8890 | 0.9537 | 0.8998 | 0.9949 | 0.9960 | 0.9467 | — |
| tuned_xgb_all | 0.8757 | 0.9559 | 0.9060 | 0.9920 | 0.9942 | 0.9448 | −0.0019 (−0.0133) |
| tuned_lgbm_local / tuned_xgb_local | 0.860 / 0.852 | 0.952 / 0.952 | 0.897 / 0.882 | 0.994 / 0.993 | 0.995 / 0.994 | 0.940 / 0.935 | rejected |

`all_no_test_drift5` drops the five strongest train-vs-test separators: feat_103, feat_136,
feat_101, feat_2, feat_139. Four of the five are aggregated features, matching §6.

**Reading.**
- Tuned LightGBM beats every other family. Cross-family ensembles do *not* beat the best
  single LightGBM, so model-family diversity adds no value here.
- The tuned blend weight on local-only models came out at 0.1 for both families. Tuning scores
  are flat from 0 to 0.2, so the local blend adds essentially nothing.
- Removing the five test-drifted features is the only change that improves **every** window
  (min +0.0009) and is best on both Kaggle-aligned short windows (+0.0018 / +0.0011 vs
  shallow_all). That is consistent with the drift mechanism in §6, although validation cannot
  reproduce the test-period drift, so its true benefit can only show on Kaggle.
- Dropping only feat_2 gives the best long-window and mean scores but is *below* shallow_all
  on the short windows (−0.0020 / −0.0017). Given §6, expect it to do no better on Kaggle.

**Finalists for Kaggle round 2:** tuned_lgbm_all_no_test_drift5, tuned_lgbm_all,
tuned_lgbm_all_no_feat2 (10-seed bags; files `submissions/submission_<name>.csv`).

## 9. Step 5 — stress tests (2026-10-10)

**Code:** `src/run_step5_stress.py`; outputs `artifacts/stress/stress_summary.csv`,
`stress_window_results.csv`. Each model is fitted once per report window and seed (3 seeds) and
scored on perturbed validation rows:
- `drift_0.5`, `drift_1.0`: every feature pushed further along the observed train→test median
  shift;
- `noise_local`, `noise_aggregated`: Gaussian noise with sd = 0.25 × the training sd, applied
  to one feature group;
- `withhold_high_illicit`: refit without the fraud-heavy training steps 9, 13, 20, 28, 29, 32
  (illicit share above 25%);
- `rare_positives`: illicit cases subsampled to 2% prevalence, 200 draws; mean and 5th
  percentile reported.

Mean ROC-AUC over all 5 report windows:

| Model | clean | drift 0.5 / 1.0 | noise local | noise aggregated | withhold high-illicit | rare-positive mean / p5 | worst scenario |
|---|---|---|---|---|---|---|---|
| shallow_all | 0.9480 | 0.9349 / 0.9364 | **0.8741** | 0.9396 | 0.9379 | 0.9483 / 0.9371 | 0.8741 |
| tuned_lgbm_all | 0.9523 | 0.9423 / 0.9421 | 0.8644 | 0.9412 | 0.9432 | 0.9523 / 0.9415 | 0.8644 |
| tuned_lgbm_all_no_test_drift5 | 0.9542 | 0.9463 / 0.9455 | 0.8713 | 0.9391 | 0.9491 | 0.9538 / 0.9421 | 0.8713 |
| tuned_lgbm_all_no_feat2 | **0.9559** | **0.9495 / 0.9489** | 0.8652 | **0.9430** | **0.9533** | **0.9562 / 0.9458** | 0.8652 |

Same, restricted to the two Kaggle-aligned short windows (≤25→26–35, ≤28→29–35):

| Model | clean | drift 0.5 / 1.0 | noise local | noise aggregated | withhold | rare p5 |
|---|---|---|---|---|---|---|
| shallow_all | 0.9955 | 0.9892 / 0.9857 | **0.8968** | 0.9911 | 0.9947 | 0.9924 |
| tuned_lgbm_all | 0.9957 | 0.9919 / 0.9889 | 0.8771 | 0.9907 | 0.9946 | 0.9929 |
| **tuned_lgbm_all_no_test_drift5** | **0.9969** | 0.9917 / **0.9895** | 0.8908 | **0.9915** | **0.9963** | **0.9940** |
| tuned_lgbm_all_no_feat2 | 0.9935 | 0.9871 / 0.9859 | 0.8659 | 0.9865 | 0.9925 | 0.9900 |

**Reading.**
- Every model's weak point is noise in the *local* features (−0.07 to −0.09). The models
  depend on local transaction attributes, and the real test drift is not there (§6: local
  features barely move). The more heavily tuned models are a little more fragile to that noise
  than shallow_all.
- Under simulated additional drift, withheld fraud-heavy steps, and rare positives,
  no_test_drift5 is best on the Kaggle-aligned windows and no_feat2 is best on the all-window
  average. Withholding the fraud-heavy steps costs little (≤0.01), so no model hinges on a
  few fraud spikes.
- Rare-positive regime (2% illicit, like the post-step-43 period): AUC is unchanged on average
  and the 5th-percentile draw loses only about 0.01, so the ranking holds even when illicit
  cases are scarce.
- By the worst-case rule, shallow_all (0.874) narrowly leads no_test_drift5 (0.871). The gap
  comes entirely from the synthetic local-noise scenario, the least realistic shock for this
  test period. The overall evidence favours **tuned_lgbm_all_no_test_drift5** as the primary
  candidate and **shallow_all** (already 0.94267 on Kaggle) as the safe second final pick.

Test-set rank correlation: no_test_drift5 vs tuned_lgbm_all 0.994 (near-duplicates; expect
Kaggle scores within noise), vs shallow_all 0.974, vs no_feat2 0.959.

## 10. Step 7 — final pipeline and deliverables (2026-10-10)

- `src/final_config.py`: version-controlled final configuration. **Provisional** final model:
  `tuned_lgbm_all_no_test_drift5` (tuned LightGBM, 165 features minus feat_103, feat_136,
  feat_101, feat_2, feat_139; 10 seeds). Safe alternative recorded: `shallow_all`.
- `src/train.py`: validates the configured model against the original LightGBM on the 5
  report windows (3 seeds), refits on all labeled rows and writes
  `submissions/submission_final.csv` plus `models/final_model.joblib`. Notebook validation
  output: final 0.9542 mean vs original 0.9408 (+0.0134), better in all 5 windows (min
  +0.0003), seed sd 0.0017 vs 0.0048.
- `e1304439.ipynb` (repo root, tracked) is generated and executed by `src/build_notebook.py`.
  It covers the data audit, graph/drift diagnostics, configuration, validation, final fit,
  submission and the saved experiment tables. Its `submission_final.csv` is identical to
  `submissions/submission_tuned_lgbm_all_no_test_drift5.csv` (max abs difference 0.0).
- `requirements.txt` pinned to the versions used (numpy 1.26.4 because torch 2.2.2 needs
  NumPy 1.x). README rewritten for the new pipeline. 32 unit tests pass.

**Kaggle round 2 queue (2026-10-11):**

| File | Purpose | Public AUC |
|---|---|---|
| `submission_tuned_lgbm_all_no_test_drift5.csv` | provisional final model | **0.94365** (2026-10-11) |
| `submission_tuned_lgbm_all_no_feat2.csv` | best all-window mean; tests the feat_2-only drop | **0.93673** (2026-10-11) |
| `submission_tuned_lgbm_all.csv` | tuning alone vs shallow_all (0.94267); rank ρ 0.994 with drift5, so low information | **0.94417** (2026-10-11) |

**Final private-leaderboard picks (2 allowed):** the best round-2 model if it beats 0.94267
by more than the ~0.005 noise level; otherwise drift5 plus `shallow_all` (0.94267).

## 11. Kaggle round 2 (2026-10-11) — a plateau at about 0.944

Leaderboard leader at this point: **0.96849** public.

| Submission | Public ROC-AUC | Short windows ≤25 / ≤28 (local) | Mean of 5 windows (local) |
|---|---|---|---|
| **tuned_lgbm_all** | **0.94417** | 0.9953 / 0.9961 | 0.9525 |
| tuned_lgbm_all_no_test_drift5 | 0.94365 | 0.9967 / 0.9972 | 0.9540 |
| shallow_all (round 1) | 0.94267 | 0.9949 / 0.9960 | 0.9467 |
| original (time_step + in_degree) | 0.94246 | — | — |
| tuned_lgbm_all_no_feat2 | 0.93673 | 0.9929 / 0.9943 | 0.9557 |

**Implications.**
1. **Tuning helps only marginally on test.** tuned_lgbm_all beats shallow_all by +0.0015
   and the original model by +0.0017, both within public-LB noise (~±0.005). Five tree
   variants now span 0.9367–0.9442. The supplied-feature gradient-boosting approach has hit a
   plateau about 0.025 below the leader.
2. **Dropping the test-drifted features does not help on test** (−0.0005 vs tuned_lgbm_all,
   noise). The drift diagnosis (§6) explains why aggregated-only failed, but removing the
   drifted columns from an all-feature model is neutral. The trees evidently do not rely on
   them in a harmful way.
3. **Dropping feat_2 alone hurts (−0.0074).** That matches its position as the weakest
   finalist on the short windows: short-window validation predicted the bottom of the ranking
   again (3 of 3 rounds directionally right for differences larger than ~0.005). It is
   confounded with that model's different hyperparameters (4 leaves, 1200 trees), so it does
   not show that feat_2 itself matters.
4. **The local proxy is saturated.** The Kaggle-aligned short windows score 0.995–0.997,
   while the test scores 0.944. They can rank large differences but cannot resolve the
   +0.01–0.02 improvements now needed. The long windows (0.90–0.92) match the test's
   difficulty level but ranked feature groups wrongly. No single local window both matches the
   test difficulty and ranks correctly. From here the public leaderboard has to serve as the
   validation set for close calls. With about 13 days × 3 submissions left there is budget for
   that, but only differences above ~0.005 are informative, and the private LB (70%) decides
   the grade.
5. **Grade arithmetic** (score = 10 × (AUC / best)^5, using the current public leader as
   proxy): 0.94417 → 8.81 / 10; 0.955 → 9.32; 0.960 → 9.57. Each +0.005 AUC is worth about
   0.25 points.

### 11.1 Diagnostics run after round 2 (2026-10-11)

**Homophily is strong, and labeled-neighbour structure is consistent between train and test.**
- On training edges whose endpoints are both labeled (23,900), a neighbour of an illicit
  transaction is illicit 57.4% of the time, against a base rate of 11.7% (4.9× lift). Label
  agreement across an edge is 94.7%.
- Restricted to *labeled* neighbours (the only neighbours with features in the test period),
  training and test look alike: 78.2% vs 74.6% of nodes have at least one labeled neighbour,
  and labeled neighbours make up 63.1% vs 60.9% of all neighbours. Neighbour features computed
  over labeled neighbours only therefore avoid the mismatch that broke GraphSAGE (§1.3).
- Univariate AUCs on labeled training rows: number of labeled neighbours 0.324 (i.e. 0.676 when
  inverted: illicit transactions have fewer labeled neighbours), in-degree 0.458, out-degree
  0.417. Oracle mean neighbour label: 0.777 on the 78% of rows with labeled neighbours. That
  oracle uses true labels, so it is only an upper bound.

**Pilot: two-stage labeled-neighbour stacking** (code: scratchpad pilot, results
`artifacts/extensions/stacking_pilot_window_table.csv`).
- Stage 1 is tuned_lgbm_all. Training-window rows are scored out-of-fold (5 folds of time
  steps); validation rows by the window model.
- Stage 2 adds the mean/max/in-mean/out-mean of stage-1 scores over labeled neighbours, plus
  degree and labeled-neighbour counts. 2 seeds.

| Variant | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | mean |
|---|---|---|---|---|---|---|
| base (tuned_lgbm_all) | 0.8991 | 0.9587 | 0.9117 | 0.9953 | 0.9960 | 0.9522 |
| + structure | 0.9009 | 0.9563 | 0.9037 | 0.9954 | 0.9964 | 0.9505 |
| + neighbour scores | 0.8948 | 0.9554 | 0.9010 | 0.9961 | 0.9975 | 0.9490 |
| + both | 0.8955 | 0.9524 | 0.8928 | 0.9960 | 0.9974 | 0.9468 |

Neighbour scores add +0.0008 / +0.0015 on the short windows and lose 0.004–0.01 on the long
ones. Under the adoption rule this is not adopted on local evidence. Because the short windows
are saturated, a single Kaggle submission is the only way to judge it properly.

**Leakage check (negative).** Transaction ID vs label: AUC 0.560 pooled, 0.545 within step.
File row order: 0.656 pooled but 0.566 within step, so the pooled figure reflects how fraud
prevalence varies by step. Neither offers a usable shortcut, and none was used.

**Per-step score calibration is not the bottleneck** (`artifacts/extensions/prior_shift_pilot.csv`).
On long windows, pooled AUC is 0.02–0.03 below the mean within-step AUC, which suggested that
scores were mis-levelled between steps. Adjusting each step's scores to a new illicit prior,
mean over 5 windows (tuned_lgbm_all = 0.9525):
- EM prior estimate (Saerens et al., 2002): 0.9440;
- mean-probability prior: 0.9511;
- oracle true per-step prevalence: 0.9571;
- per-step percentile ranking: worse on every window (e.g. 0.8710 vs 0.9119).

Even perfect prevalence knowledge adds only +0.0046, and the practical estimators lose. The gap
comes from a few steps where within-step ranking itself fails (§3: steps 24, 29, 31, 32), not
from miscalibration.

### 11.2 Final-pick status
- Best public score: tuned_lgbm_all (0.94417). It is simpler than drift5 (no arbitrary drops)
  and within noise of it.
- Recommended config change: `src/final_config.py` → tuned_lgbm_all.
- Recommended private-LB picks today: **tuned_lgbm_all** + **shallow_all** (ρ 0.973 between
  them, both ≥ 0.9427). drift5 is a near-duplicate of tuned_lgbm_all (ρ 0.994), so picking
  both would waste the second pick's diversity.

## 12. Proposed next steps (2026-10-11)

The supplied features plus gradient boosting appear to be at a plateau of about 0.944. Closing
part of the 0.025 gap needs **new information or a different training distribution**, not more
tuning. Local windows are saturated, so each idea gets a quick local sanity check (it must not
lose more than 0.01 on any window) and is then judged on Kaggle. Ordered by expected value per
effort:

| # | Idea | Why it might help | Local evidence so far | Cost |
|---|---|---|---|---|
| 1 | **Recency-focused training**: train only on steps ≥ 15 / ≥ 22, or exponential recency weights (half-life 7–14) on the tuned model | Test starts right after step 35, and the hard steps (29, 32) carry recent fraud patterns. The earlier recency test used the old deep model on a single long holdout. | Half-life 3 improved the ≤28 window slightly (README); never tested with tuned params on short windows | low |
| 2 | **Temporal bagging**: average tuned LightGBMs trained on different start windows (1–35, 10–35, 20–35) and seeds | Combines long-memory and recent-pattern models without choosing between them; variance reduction | — | low |
| 3 | **Labeled-neighbour stacking** (pilot above) | Strong homophily (4.9× lift) and a train/test-consistent neighbourhood rule; the only new information source left in the data | +0.001 short / −0.004 to −0.01 long | medium |
| 4 | **Covariate-shift importance weighting**: weight training rows by p(test \| x) / p(train \| x) from an adversarial classifier that excludes the 5 time-proxy features (Sugiyama et al., 2007) | Focuses the model on training transactions resembling the test period | not testable locally (the shift exists only train→test) | medium |
| 5 | **De-trend only the drifted aggregated features**: per-step robust standardisation of the top ~10 train-vs-test separators, leaving the rest raw | Keeps their within-step signal while removing the level drift | Ranking *all* features per step hurt (0.845); targeted version untested | low |
| 6 | **Per-time-step error analysis on the hard validation steps** (24, 29, 31, 32): which features flip, and whether illicit transactions there form new clusters | Tells us whether the test's difficulty is new fraud typologies (favours ideas 1, 3) or feature drift (favours 4, 5) | — | low |

**Suggested Kaggle schedule** (3 per day; days 1–2 shown):
- **Day 3:** recency variant (#1), temporal bag (#2), neighbour stacking (#3), all on the
  tuned_lgbm_all base.
- **Day 4:** the best of day 3 combined with #4 or #5, plus one seed/variance check of the best.
- **Afterwards:** keep two slots a day for combinations and one for re-checking.

**Not recommended:** further hyperparameter search, cross-family ensembles, GraphSAGE variants,
pseudo-labeling and aggregated-only models (all tested; §5–§9, §11). External data such as the
Elliptic++ wallet features is out unless the competition rules allow it.

**Deliverable housekeeping** (independent of model work): switch `final_config.py` to
tuned_lgbm_all and rebuild the notebook; start the 4-page report from §1, §3, §6, §8 and §11,
which carry the main narrative; the final two picks are listed in §11.2.

## 13. Round 3 variants (2026-10-12)

**Code:** `src/run_step8_variants.py` (tests: `tests/test_round3_variants.py`); outputs in
`artifacts/round3/`. All variants are built on tuned_lgbm_all's settings and use all 165
features. Local sanity check: report windows, 2 seeds.

| Variant | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | mean | Adoption rule |
|---|---|---|---|---|---|---|---|
| temporal_bag (all / last 25 / last 15 steps, rank-averaged) | 0.8984 | 0.9581 | **0.9145** | 0.9963 | 0.9972 | **0.9529** | pass (worst −0.0007) |
| halflife_14 (recency weights) | 0.9061 | 0.9560 | 0.9078 | 0.9965 | 0.9973 | 0.9527 | pass (worst −0.0040) |
| recent_21 (last 21 steps only) | 0.8991 | 0.9587 | 0.9117 | 0.9964 | 0.9973 | 0.9526 | pass (0.0000) |
| tuned_lgbm_all (reference; Kaggle 0.94417) | 0.8991 | 0.9587 | 0.9117 | 0.9953 | 0.9960 | 0.9522 | — |
| halflife_7 | **0.9125** | 0.9519 | 0.9018 | **0.9968** | **0.9975** | 0.9521 | fail (−0.0100) |
| nbr_stack (labeled-neighbour stacking) | 0.8948 | 0.9554 | 0.9010 | 0.9961 | **0.9975** | 0.9490 | fail (−0.0107) |
| recent_14 | 0.8991 | 0.9384 | 0.8844 | 0.9968 | 0.9974 | 0.9432 | fail (−0.0274) |

(For windows that start at step ≤ 14, "last 21 steps" covers all training steps, which is why
recent_21 equals the reference there.)

Every recency-type variant lifts the two Kaggle-aligned short windows by +0.001 to +0.0015.
That is consistent, but small and close to the windows' ceiling. nbr_stack is submitted despite
failing the rule because it is the only variant that adds new information (§11.1), and only
Kaggle can measure it.

**Kaggle round 3 queue** (10-seed bags; test rank correlation with tuned_lgbm_all: halflife_14
0.992, temporal_bag 0.985, nbr_stack 0.977):

| File | Purpose | Public AUC |
|---|---|---|
| `submission_halflife_14.csv` | recency weighting (idea #1) | **0.95066** (2026-10-12) |
| `submission_temporal_bag.csv` | temporal bagging (idea #2) | **0.94672** (2026-10-12) |
| `submission_nbr_stack.csv` | labeled-neighbour stacking (idea #3) | **0.95389** (2026-10-12) |

**Final config switched to tuned_lgbm_all** (best public score so far, 0.94417);
`src/final_config.py` keeps the drift5 drop list documented but unused.

## 14. Kaggle round 3 (2026-10-12) — two real gains: neighbours and recency

| Submission | Public ROC-AUC | Δ vs tuned_lgbm_all (0.94417) | Local mean (5 windows) | Local adoption rule |
|---|---|---|---|---|
| **nbr_stack** | **0.95389** | **+0.0097** | 0.9490 | fail (−0.0107) |
| halflife_14 | 0.95066 | +0.0065 | 0.9527 | pass |
| temporal_bag | 0.94672 | +0.0026 (noise) | 0.9529 | pass |
| tuned_lgbm_all | 0.94417 | — | 0.9522 | — |

Leader 0.96849; gap now 0.0146 (was 0.0243). Grade proxy 10 × (0.95389/0.96849)^5 = **9.27**
(was 8.81).

**Main drivers.**
1. **Labeled-neighbour information (+0.0097).** This is the first gain from information beyond
   a transaction's own supplied features. It works because (a) illicit transactions cluster:
   57% of an illicit transaction's labeled neighbours are illicit, against 11.7% base (§11.1);
   and (b) the neighbourhood is defined so that it means the same thing in training and test:
   only *labeled* neighbours (the only ones with features in steps 36–49), scored by a model
   that never sees their labels. GraphSAGE failed on exactly the second point (§1.3, §7).
2. **Recency weighting (+0.0065).** Weighting training rows by 2^((t − 35)/14) (step 35 = 1.0,
   step 21 = 0.5, step 7 = 0.25) shifts the model toward the most recent fraud patterns. That
   suits a test period that starts right after step 35.
3. Temporal bagging is directionally positive but within noise; it averages full-history and
   recent models, a weaker form of driver 2.

**What the stacked model relies on** (1-seed diagnostic refit on all labeled rows):
- the four neighbour-score features carry 22.5% of stage-2 training gain but only 7.0% of mean
  |SHAP| on test rows. Ranked by test SHAP: in-neighbour mean 10th, overall mean 12th,
  out-neighbour mean 19th, max 40th;
- 74.6% of test rows have at least one labeled neighbour (78.2% of training rows);
- stage 2 moves test ranks more for rows *with* labeled neighbours (mean |Δ percentile| 0.054)
  than without (0.045). Rows without neighbours also move, so part of the gain may come from
  stage 2 re-learning the base features with a different tree structure, not only from
  neighbour information. A dedicated ablation would be needed to separate the two;
- rank moves are similar before and after the step-43 shutdown (0.051 vs 0.053).
- The smaller SHAP share on test than on training suggests that test neighbour scores (from the
  full stage-1 model) are distributed differently from training neighbour scores (out-of-fold).
  Aligning the two is a possible further gain (§15, idea C).

**Validation lesson.** Local windows predicted neither gain. nbr_stack *failed* the local rule
and halflife_14 showed only +0.0005. Local windows fall short in both directions: they reward
the wrong feature groups (§6) and miss the right ideas (here). For this competition the public
LB is the only reliable selector for changes of ±0.005–0.01. The private LB (70%) may still
differ, which argues for picking two *different* strong models at the end.

### 14.1 Current best model — `nbr_stack` (exact specification)

| Component | Setting |
|---|---|
| Training rows | 31,235 labeled transactions, steps 1–35 (`none` rows excluded); `time_step` never a feature |
| Base features | all 165 supplied features (`feat_1`–`feat_165`) |
| Model (both stages) | LightGBM, binary, `class_weight="balanced"`, 500 trees, learning rate 0.0253, 9 leaves, min_child_samples 49, colsample_bytree 0.644, subsample 0.983 (subsample_freq 1), reg_lambda 2.04 — Optuna-tuned on short-horizon windows (§8) |
| Stage 1 | Training rows: out-of-fold scores, 5 folds of consecutive time steps (each fold's steps scored by a model trained on the other steps). Test rows: a model trained on all labeled rows |
| Neighbour features | Edge list in both directions; for each transaction, over neighbours that are labeled training rows or test rows: mean, max, mean over incoming (predecessor) neighbours, mean over outgoing (successor) neighbours of stage-1 scores. NaN when there are no labeled neighbours (LightGBM's missing-value branch) |
| Stage 2 | Same LightGBM settings on 165 + 4 = 169 features |
| Seeds | 10 (0–9); each seed repeats both stages; test probabilities averaged |
| Code | `src/run_step8_variants.py` (`neighbour_stack`), `python -m src.run_step8_variants submit nbr_stack` |

`halflife_14` uses the same LightGBM settings and 165 features, with sample weights
2^((t − t_max)/14) normalised to mean 1, multiplied into the class-balancing weights; 10 seeds.

Test rank correlation nbr_stack vs halflife_14: 0.968, different enough to make a good pair of
final picks.

**Final picks as of today:** nbr_stack (0.95389) + halflife_14 (0.95066).

## 15. Proposed next steps after round 3 (2026-10-12)

Ordered by expected value; each is one Kaggle submission (the local check only guards against
breakage).

| # | Idea | Rationale |
|---|---|---|
| A | **Combine the two drivers**: nbr_stack with half-life-14 recency weights in both stages | The gains come from independent mechanisms (graph context vs temporal drift), so they may add up |
| B | **Rank blend** nbr_stack + halflife_14 (50/50) | Cheap; ρ = 0.968, so the blend has diversity to exploit |
| C | **Richer, better-aligned neighbour features**: counts of labeled neighbours, in/out degree, and neighbour aggregates of the top local features (feat_53, feat_59, feat_61, feat_5, feat_3) over labeled neighbours; also score test neighbours out-of-fold, using an ensemble of the 5 fold models instead of the full model, so training and test neighbour scores come from the same kind of model | Targets the train/test SHAP-share mismatch above and adds non-score neighbour context |
| D | **Iterative (collective) classification**: repeat stage 2 using stage-2 neighbour scores (2–3 rounds; Lu & Getoor 2003; Neville & Jensen 2000) | Propagates evidence along fraud chains beyond one hop |
| E | **Two-hop neighbour scores** (mean score of labeled neighbours-of-neighbours) | Laundering chains are often longer than one hop |
| F | **Half-life sweep** (7, 10, 21) on the best stacked model | 14 was a guess; 7 was best on the Kaggle-aligned windows locally |

Suggested schedule: day 4 → A, B, C; day 5 → D or E on the day-4 winner, plus F.

**Deliverables:** once the final approach is settled, move the stacking pipeline into
`src/train.py` / `final_config.py` (currently tuned_lgbm_all) and rebuild `e1304439.ipynb`, so
the notebook reproduces the best submission.

## 16. Round 4 variants (2026-10-13)

**Code:** `src/run_step8_variants.py` (`make_neighbour_stack`, `neighbour_context_features`,
`blend` command); tests in `tests/test_round3_variants.py`; outputs `artifacts/round4.log`,
`artifacts/round3/round4_*.csv`.

- **A, `nbr_stack_hl14`:** nbr_stack with recency weights 2^((t − t_last)/14) (mean one) in
  stage 1, its out-of-fold fits and stage 2. The reference step is the last training step for
  every fit, so the out-of-fold models weight steps the same way as the full model.
- **B, `blend_nbr_hl14`:** equal-weight rank average of the scored files
  `submission_nbr_stack.csv` (0.95389) and `submission_halflife_14.csv` (0.95066).
- **C, `nbr_stack_rich`:** nbr_stack plus
  - in/out degree over the full edge list;
  - the number and share of labeled neighbours;
  - neighbour means of feat_53, feat_59, feat_61, feat_5, feat_3 (the top test-SHAP local
    features, §14) over labeled neighbours;
  - target (test) neighbour scores taken as the mean of the 5 out-of-fold stage-1 models
    instead of a separate full-data model, so training and test neighbour scores come from the
    same kind of model.

  No recency weights, so A and C each test one change against nbr_stack.

Local sanity check (2 seeds):

| Variant | ≤14→15–35 | ≤21→22–28 | ≤21→22–35 | ≤25→26–35 | ≤28→29–35 | mean |
|---|---|---|---|---|---|---|
| tuned_lgbm_all (Kaggle 0.94417) | 0.8991 | 0.9587 | 0.9117 | 0.9953 | 0.9960 | 0.9522 |
| nbr_stack (Kaggle 0.95389) | 0.8948 | 0.9554 | 0.9010 | 0.9961 | 0.9975 | 0.9490 |
| **nbr_stack_hl14** | 0.8994 | 0.9525 | 0.8971 | **0.9972** | **0.9981** | 0.9489 |
| nbr_stack_rich | 0.8940 | 0.9486 | 0.8878 | 0.9961 | 0.9978 | 0.9449 |

As with nbr_stack, every stacked variant loses on the long windows and gains on the
Kaggle-aligned short windows. nbr_stack_hl14 posts the highest short-window scores of any model
so far. nbr_stack_rich matches nbr_stack on short windows and loses more on long ones. Given §14
(local windows missed nbr_stack's +0.0097), these numbers serve only as a breakage check.

Test rank correlations: A vs nbr_stack 0.991, C vs nbr_stack 0.992, B vs nbr_stack 0.992,
A vs C 0.983.

**Kaggle round 4 queue:**

| File | Purpose | Public AUC |
|---|---|---|
| `submission_nbr_stack_hl14.csv` | A — do the two drivers add up? | **0.95858** (2026-10-13) |
| `submission_blend_nbr_hl14.csv` | B — blend of the two best files | **0.95328** (2026-10-13) |
| `submission_nbr_stack_rich.csv` | C — richer, aligned neighbour context | **0.95871** (2026-10-13) |

## 17. Kaggle round 4 (2026-10-13) — graph context keeps paying; blending with a non-graph model does not

| Submission | Public ROC-AUC | Δ vs nbr_stack (0.95389) | What changed |
|---|---|---|---|
| **nbr_stack_rich** (C) | **0.95871** | +0.0048 | + neighbour feature means, structure counts, fold-averaged test scores |
| nbr_stack_hl14 (A) | 0.95858 | +0.0047 | + recency weights (half-life 14) in both stages |
| nbr_stack | 0.95389 | — | round 3 |
| blend_nbr_hl14 (B) | 0.95328 | −0.0006 | 50/50 rank average with halflife_14 (no graph) |
| halflife_14 | 0.95066 | −0.0032 | round 3 |
| tuned_lgbm_all | 0.94417 | −0.0097 | round 2 |

Leader 0.96849; gap now 0.0098 (from 0.0243 before round 3). Grade proxy
10 × (0.95871/0.96849)^5 = **9.50** (8.81 → 9.27 → 9.50).

**Implications.**
1. **Both extensions improve on nbr_stack by about the same amount (+0.0047 / +0.0048).** Each is
   at the edge of the ~0.005 noise band on its own. Two independent changes both landing about
   +0.005 is stronger evidence than either alone. Because each was tested separately against the
   same reference, their combination (recency + rich context) is the natural next candidate.
2. **Recency weighting has now helped twice:** +0.0065 on the plain model (round 3) and +0.0047
   on the stacked model. It is a robust driver, not a one-off.
3. **Blending the graph model with a non-graph model hurts slightly** (B 0.95328 < nbr_stack
   0.95389). The non-graph model adds no information the stack lacks and dilutes the neighbour
   signal. Model diversity without new information does not help here; the same lesson as the
   cross-family ensembles (§8) and temporal bagging (§14).
4. **Local validation still cannot rank these.** Every stacked variant failed the local rule,
   and B, C and A scored nearly equal on the short windows. The public LB remains the arbiter;
   the 70% private LB is the residual risk.

### 17.1 What makes the stacked model stronger (1-seed diagnostic of nbr_stack_rich)

Mean |SHAP| share by feature group, training rows vs test rows:

| Group | Train | Test |
|---|---|---|
| neighbour stage-1 scores (mean, max, in-mean, out-mean) | 0.059 | 0.046 |
| neighbour feature means (feat_53/59/61/5/3 over labeled neighbours) | 0.065 | **0.071** |
| structure (in/out degree, labeled-neighbour count and share) | 0.009 | 0.009 |
| **graph-derived total** | 0.133 | **0.126** |

- In nbr_stack (round 3), the neighbour scores took 22.5% of training gain but only 7.0% of
  test SHAP. In the rich model, the graph-derived features carry the **same share on train and
  test** (13.3% vs 12.6%), so the model uses graph context on the test period as it learned to.
- The **neighbour feature means hold up best on test**: `nbr_mean_feat_59` (the average of
  feat_59 over a transaction's labeled neighbours) is the **3rd most important feature on test
  rows**, behind only feat_53 and feat_125. Unlike neighbour *scores*, these need no stage-1
  model, so the stage-1 model cannot carry its train/test shift into them.
- Score alignment barely changed levels: the mean test stage-1 score is 0.0884 (fold-average)
  vs 0.0877 (full model). Training out-of-fold scores average 0.142, consistent with a lower
  illicit share in the test period. The gain in C therefore comes from the new neighbour
  context, not from the alignment.
- Structure counts contribute little (0.9%).

**Interpretation for the report.** The supplied aggregated features summarise each
transaction's *whole* neighbourhood, and they drift strongly into the test period (§6). The
self-built aggregates summarise only *labeled* neighbours, a neighbourhood defined identically
in training and test, and they carry their weight into the test period. The design principle
that made graph information usable: **define every graph feature so that it is computed the
same way at training and prediction time.**

### 17.2 Current best model — `nbr_stack_rich` (specification)

As nbr_stack (§14.1), with these changes:
- **Test-row stage-1 scores:** mean of the five out-of-fold stage-1 models (instead of one model
  fitted on all rows).
- **Extra stage-2 features (8 more, 177 total):**
  - in_degree and out_degree (full edge list);
  - number of labeled neighbours, and that number divided by total degree;
  - neighbour means of feat_53, feat_59, feat_61, feat_5, feat_3 over labeled neighbours.
- **Settings:** LightGBM settings, seeds (10) and training rows unchanged. No recency weights.
- **Command:** `python -m src.run_step8_variants submit nbr_stack_rich`.

**Final picks as of today:** nbr_stack_rich (0.95871) + nbr_stack_hl14 (0.95858). Their test
rank correlation is 0.983, the most different pair among the top models.

## 18. Proposed next steps toward 0.96+ (2026-10-13)

| # | Idea | Rationale | Cost |
|---|---|---|---|
| 1 | **Combine A + C: `nbr_stack_rich_hl14`** (rich context + half-life-14 weights in both stages) | Two independent ~+0.005 gains, each tested separately against the same reference | low |
| 2 | **Wider neighbour feature context**: neighbour mean/max/std of the top 20–30 features by test SHAP (not just 5), split into incoming vs outgoing neighbours | Neighbour feature means are the best-transferring graph signal (17.1); only 5 features were used | low |
| 3 | **Two-hop and iterated context**: labeled neighbours-of-neighbours (scores and feature means), then a second stacking round using stage-2 scores (collective classification; Lu & Getoor 2003) | Laundering moves funds through chains; one hop only sees the immediate step | medium |
| 4 | **Half-life sweep (7, 10, 21) on the best stack** | 14 was never tuned; 7 was best on the Kaggle-aligned windows locally | low |
| 5 | **Own stage-1 score as a stage-2 feature, and 20 seeds** | Standard stacking addition; more seeds reduce variance, which matters at ±0.005 margins | low |

Not recommended: rank blends with non-graph models (B, §17), aggregated-only models, further
hyperparameter search, GraphSAGE and pseudo-labeling.

**Suggested schedule:** day 5 → #1, #2 (on top of #1), #3-lite (two-hop means on top of #1); day 6
→ #4 and #5 on the day-5 winner, plus one slot held in reserve.

**Deliverables (needed regardless of further gains):** port the winning stack into
`src/final_config.py` / `src/train.py` and rebuild `e1304439.ipynb`, so the notebook reproduces
the best submission. The notebook currently reproduces tuned_lgbm_all (0.94417).
