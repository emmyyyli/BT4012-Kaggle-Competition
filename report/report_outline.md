# Report outline: e1304439.pdf (max 4 pages)

Working outline for the written report. Each section gives a **page budget**, the **one message**
it must land, the **content with its evidence**, and the **source** in `report/findings.md` (§).
Numbers are copied from the findings log; recheck any marked _[update]_ before submission.

Deadline: 25 Oct 2026 23:59 SGT. The report must include the GitHub link:
https://github.com/emmyyyli/BT4012-Kaggle-Competition (make sure the assessors can access it).

**Page budget (4 pages, ~11pt, two figures + four compact tables):**

| Section | Pages |
|---|---|
| 1 Introduction | 0.4 |
| 2.1 Data preprocessing & feature engineering | 0.9 |
| 2.2 Model development & validation | 0.6 |
| 2.3 Model exploration & final strategy | 0.7 |
| 3 Results & discussion | 1.0 |
| 4 Conclusion & reflection | 0.3 |
| References + repo link | 0.1 |

**The storyline in one sentence:** a well-regularised tree model on the supplied features
plateaued at about 0.944. The decisive gains (to 0.968) came from graph context built so that
every feature is computed *the same way at training and prediction time*, plus recency
weighting. Validation had to be redesigned because time-forward drift made standard
cross-validation misleading.

---

## 1. Introduction (≈0.4 page)

**Message:** Detecting illicit Bitcoin transactions is an imbalanced, temporally drifting,
graph-structured problem. The approach is designed around those three properties.

- **Task:** classify transactions in the Elliptic Bitcoin graph as licit or illicit; the
  metric is ROC-AUC on future time steps (train steps 1–35, test steps 36–49).
- **Why it matters:** anti-money-laundering investigators need to catch illicit flows early
  without being flooded by false alarms. A ranking metric (AUC) mirrors triage by risk score.
- **Three challenges, each with the strategy that answers it:**
  1. *Few labels, heavy imbalance:* 31,235 labeled rows, 11.7% illicit; 110,537 unlabeled.
     → supervised learning on labeled rows only, with balanced class weights.
  2. *Temporal drift:* the test period is later, and a classifier separates train from test
     rows with AUC ≈ 1.0. → time-forward validation, recency weighting, `time_step` never
     used as a feature.
  3. *Graph structure with missing context in the test period.* → label-free neighbourhood
     features computed the same way in training and test.
- **Headline result:** public ROC-AUC 0.94246 (first submission) → **0.96858** (leaderboard
  top 0.97012) _[add the private score once released]_.

## 2. Methodology

### 2.1 Data preprocessing and feature engineering (≈0.9 page)

**Message:** The data audit, not the model, revealed the key constraint. In the test period,
only labeled transactions have features, so graph features must be built around that.

**Table 1: data audit** (§1)

| Item | Value |
|---|---|
| Train / test rows | 141,772 (steps 1–35) / 15,329 (steps 36–49) |
| Labels (train) | 27,591 licit, 3,644 illicit (11.7%), 110,537 unlabeled |
| Features | 165 anonymised: `feat_1`–`feat_93` local, `feat_94`–`feat_165` 1-hop aggregates (per Elliptic documentation) |
| Missing values | none in supplied features |
| Edges / nodes | 234,355 edges; 203,769 nodes, of which 46,668 have no feature row |
| Cross-time-step edges | 0 |

**Preprocessing decisions, each with its reason:**
- **`none` rows are treated as unlabeled, not as licit**, and excluded from supervised
  training. Pseudo-labeling them was tested and *hurt* local AUC by 0.03 (§7.2): unlabeled
  rows are not a random sample of labeled ones.
- **`time_step` is used only for splitting.** As a feature it lowered long-horizon AUC
  (0.8782 vs 0.8837, §0), and trees cannot extrapolate a counter to unseen steps 36–49.
- **No imputation was needed.** For the 46,668 featureless nodes, see the key finding below.

**Key finding: the test set's structure** (§1.1–1.3)
- The test file is exactly the labeled transactions of steps 36–49: 31,235 + 15,329 = 46,564,
  the documented Elliptic labeled total.
- The 46,668 featureless nodes are the *unlabeled* transactions of steps 36–49.
- So **65.6% of test-node edges lead to featureless nodes, against 0% in training.** Any model
  that reads all neighbours sees different inputs at test time. That explains why GraphSAGE
  lowered the score (0.94246 → 0.91059).

**Feature drift** (§6). Share of features whose median shifts by more than 1 IQR:

| | Local | Aggregated |
|---|---|---|
| within training (≤28 → 29–35) | 0.0% | 9.7% |
| training → test | 1.1% | **19.4%** |

The supplied aggregates drift about twice as much into the test period, which explains why an
aggregated-only model scored 0.880 on Kaggle. Dropping the five most-drifted features was
neutral (0.94365 vs 0.94417), so all 165 features are kept.

**Graph feature engineering: labeled-neighbour stacking** (§11.1, §14, §17, §20; implemented in
`src/graph_stacking.py`)
- **Evidence it is worth doing:** labeled transactions are strongly homophilous. An illicit
  transaction's labeled neighbour is illicit 57.4% of the time, against a 11.7% base rate.
- **Design principle:** use only *labeled* neighbours, the only kind with features in both
  periods. Coverage then matches: 78.2% (train) vs 74.6% (test) of rows have one.
- **Features, in order of introduction and Kaggle gain:**
  1. Stage-1 *out-of-fold* scores of labeled neighbours (mean, max, in-mean, out-mean).
     No target labels are used (+0.0097).
  2. Neighbour means of key features, degree and labeled-neighbour counts (+0.0048).
     Neighbour feature means were the best-transferring graph signal: `nbr_mean_feat_59` was
     the 3rd most important feature on test rows.
  3. **Two-hop context:** walk-weighted means over labeled nodes two steps away, with walks
     passing *through* featureless nodes (+0.0033). This gives context to 21.8% of test rows
     that had none at one hop; rows with no context fall from 25% to 3.6%.
  4. Mean/max/std of 25 key features over labeled neighbours (+0.0015, within noise).

**Table 2 (or a small figure): context coverage** (§20.1)

| Rows | 1-hop labeled | 2-hop labeled | 2-hop only | none |
|---|---|---|---|---|
| train (labeled) | 78.2% | 87.8% | 14.3% | 7.5% |
| test | 74.6% | 93.8% | 21.8% | 3.6% |

**Feature selection** (§5, keep brief):
- Correlated-cluster ablation: only two clusters of local features matter; dropping the main
  one costs −0.022.
- SHAP-stability subsets and adversarial feature dropping did not help.
- Conclusion: no feature selection; the regularised trees cope.

### 2.2 Model development and validation (≈0.6 page)

**Message:** Validation was the hardest part. Standard CV overstated performance, and the
harness was rebuilt around time-forward windows and seed noise.

- **Model:** LightGBM. Tuned with Optuna (40 trials) on short time-forward windows:
  - 9 leaves, 500 trees, learning rate 0.025;
  - min_child_samples 49, colsample 0.64, subsample 0.98, λ = 2.0.
  - Every search (LightGBM/XGBoost/CatBoost) chose shallow trees (4–9 leaves or depth 3–4),
    so regularisation is the main defence against drift (§8).
- **Class imbalance:** `class_weight="balanced"`; AUC is threshold-free, so no resampling.
- **Recency:** sample weights 2^((t − 35)/14) (step 35 = 1, step 21 = 0.5). This gave +0.0065
  on the plain model and +0.0047 on the stack (§14, §17).
- **Stacking without leakage:** stage-1 scores for training rows are out-of-fold, with folds as
  blocks of consecutive time steps. Test rows get the mean of the fold models, so training and
  test neighbour scores come from the same kind of model.
- **Seeds:** 10 seeds averaged; 30 seeds change the test ranking negligibly (ρ = 0.9997, §25).
- **Validation harness** (`src/evaluation.py`, §3):
  - five forward windows (train ≤ t → validate later steps) × five seeds;
  - seed-bagged ROC-AUC, per-step AUC and a paired bootstrap;
  - adoption rule: the mean must improve and no window may lose more than 0.01.
  - Why it was needed: the original 4-fold CV averaged about 0.98 while Kaggle gave 0.942, and
    seed-to-seed spread reached 0.02 on a single holdout.
- **The limitation, stated honestly** (§6, §11, §14):
  - Long windows (≈0.90 AUC) match the test's difficulty but ranked models wrongly
    (aggregated-only looked best locally and scored 0.880 on Kaggle).
  - Short windows (≈0.996) ranked models correctly but are saturated, so they cannot resolve
    +0.01 gains.
  - Hence local validation served as a correctness and breakage check, and the public LB
    (30%) as the final arbiter. Each submission tested one stated hypothesis against a fixed
    reference, to limit public-LB overfitting.

### 2.3 Model exploration and final strategy (≈0.7 page)

**Message:** Complexity without new information never helped; new, consistently computed
information always did.

**Table 3: models explored** (choose about 10 rows; local = mean of 5 windows unless noted)

| Approach | Evidence | Public AUC | Decision |
|---|---|---|---|
| LightGBM + `time_step` + `in_degree` (first submission) | — | 0.94246 | baseline |
| Random Forest | worse than LightGBM on every window (§2) | — | rejected |
| Regularised / tuned LightGBM (165 features) | +0.010 local mean (§8) | 0.94417 | base model |
| XGBoost, CatBoost, cross-family ensembles | no better than LightGBM (§8) | — | rejected |
| Aggregated-only LightGBM | best on long windows | 0.88023 | rejected (drift) |
| GraphSAGE blend (all neighbours) | — | 0.91059 | rejected (test mismatch) |
| GraphSAGE with test-consistent masking | −0.047 local mean (§7) | — | rejected |
| GT-SSL self-supervised embeddings | −0.04 long holdout (README) | — | rejected |
| Pseudo-labeling | −0.03 local mean (§7) | — | rejected |
| Recency weighting (half-life 14) | short windows +0.001 | 0.95066 | adopted |
| Labeled-neighbour stacking | failed local rule | 0.95389 | adopted |
| + rich context / + recency / + two-hop | | 0.95871 / 0.96342 / 0.96668 | adopted |
| + wide 25-feature statistics | | 0.96814 | adopted |
| + peel-chain shape (final; rounds 9–11 in findings §31–35) | | **0.96858** | final model |
| 3-hop score propagation | best long-window graph variant | 0.96108 | rejected |
| Final model, 30 seeds / half-life 21 | ρ ≥ 0.998 with final | 0.96785 / 0.96670 | stability check; within noise |
| 4-model graph blend | | 0.96686 | private pick 2 |
| Blend with non-graph model | | 0.95328 | rejected |

**Why the final model and the two private picks** (§20.2, §23, §25):
- Public SE ≈ 0.0077 (≈270 illicit rows), and public and private are random parts of the
  same rows. The top three models are within 1–2 *paired* SE (≈0.0015) of each other.
- **Pick 1 (performance):** `nbr_stack_wide_2hop_chain_hl14`, 0.96858 (findings §35).
- **Pick 2 (steady):** `blend_top4`, a rank blend of the four best graph models (0.96686,
  within 0.0013 of pick 1). It averages over feature configurations.
- Kaggle scores the better of the two picks on the private 70%, so pick 2 is insurance.

## 3. Results and discussion (≈1 page)

**Figure 1 (main figure): `report/figures/fig1_public_auc.png`.** Public AUC of all 23
scored submissions (R0–R7) with the best-so-far step line, annotated with what changed at each step.
The leader's 0.97012 is shown dashed, and the final score carries a ±1 SE bar (0.0077),
showing that the gap to the leader is within noise. Regenerate with
`python -m src.report_metrics` _[update `SUBMISSIONS` with the round-7 scores first]_.

**Figure 2 (Section 2.1, optional if space is short):
`report/figures/fig2_context_coverage.png`.** Rows with labeled graph context: one hop 78.2%
train vs 74.6% test; one or two hops 92.5% vs 96.4%. It shows that the restricted
neighbourhood is comparable in both periods, and that two-hop reach helps the test period most.

**Table 4: final model vs non-graph reference across validation windows** (3 seeds;
`artifacts/final/final_validation_window_metrics.csv`)

| Window | Reference ROC-AUC / PR-AUC | Final ROC-AUC / PR-AUC |
|---|---|---|
| ≤25 → 26–35 (Kaggle-aligned) | 0.9954 / 0.9820 | **0.9971 / 0.9895** |
| ≤28 → 29–35 (Kaggle-aligned) | 0.9960 / 0.9809 | **0.9982 / 0.9933** |
| ≤21 → 22–28 | 0.9583 / 0.8818 | 0.9375 / 0.8714 |
| ≤21 → 22–35 | 0.9109 / 0.8136 | 0.8654 / 0.7980 |
| ≤14 → 15–35 | 0.9010 / 0.8427 | 0.8821 / 0.8459 |

**Table 5: triage operating points** (precision/recall/F1; 3 seeds;
`artifacts/report/precision_recall.csv`). The validation windows are ~18% illicit, so the
operating points adapt to each window's prevalence rather than using a fixed review budget.

| Window | Model | Precision = recall at k = #illicit | Precision at 90% recall | Precision at 95% recall | Best F1 (P / R) |
|---|---|---|---|---|---|
| ≤25 → 26–35 (1,307 illicit / 7,035) | non-graph LightGBM | 0.941 | 0.968 | 0.917 | 0.942 (0.967 / 0.918) |
| | **final graph stack** | **0.953** | **0.985** | **0.962** | **0.957** (0.961 / 0.953) |
| ≤28 → 29–35 (1,102 illicit / 6,028) | non-graph LightGBM | 0.946 | 0.972 | 0.940 | 0.947 (0.955 / 0.940) |
| | **final graph stack** | **0.966** | **0.993** | **0.979** | **0.966** (0.966 / 0.966) |

**Reading for the report:** at a 95% catch rate, the share of false alarms among flagged
transactions drops from 8.3% to 3.8% (≤25 window) and from 6.0% to 2.1% (≤28 window). The
graph context roughly **halves investigator workload wasted on false alarms** at the same
recall. This is the operational meaning of the AUC gain, and it ties back to the brief's
"catch bad actors early without drowning investigators in false alarms".

**Discussion points: strengths.**
- Large, mechanism-backed gains: +0.024 public AUC overall, of which +0.0225 came from graph
  context, each step a separate test.
- Stable: seed count and the recency setting barely change predictions (ρ ≥ 0.998).
- Leak-free: no test labels or `time_step` features; out-of-fold stacking.

**Discussion points: limitations and causes.**
- Local validation cannot reproduce train→test drift. Graph variants *lose* on long windows
  locally but win on test, because the test period's sparse labeled neighbourhoods make
  two-hop reach more valuable there (coverage 21.8% vs 14.3%).
- Model selection relied on the public LB (30%, SE ≈ 0.008), so the top public score is
  slightly optimistic (selection bias of ≈0.001). The gap to the leader (0.0020) is within
  noise.
- Anonymised features prevent domain-level interpretation (no amounts or fees).
- The step-43 dark-market shutdown in the test period (Weber et al., 2019) cannot be analysed
  without test labels.

## 4. Conclusion and reflection (≈0.3 page)

**Message:** In fraud detection, *how information is made available* matters more than model
complexity.

- **What worked:** a regularised tree base model; graph features computed the same way in
  training and test; recency weighting; testing one hypothesis per submission.
- **What did not:** GNNs and embeddings that read the full neighbourhood (train/test mismatch),
  pseudo-labels from unlabeled transactions, and ensembles without new information.
- **What could be improved:** a validation scheme that simulates test-period label sparsity
  (for example, masking labels inside validation windows), and directional two-hop paths
  (forward vs backward fund flow).
- **Takeaway for AI-driven fraud detection:** context from related transactions is powerful,
  but only if the features are built from information that will actually be available when the
  model is used. The same discipline prevents leakage in a production AML system.

## References (≈0.1 page; verify details before citing)

- Weber, M. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph
  Convolutional Networks for Financial Forensics. KDD Workshop on Anomaly Detection in Finance.
- Pareja, A. et al. (2020). EvolveGCN: Evolving Graph Convolutional Networks for Dynamic
  Graphs. AAAI.
- Lorenz, J. et al. (2020). Machine learning methods to detect money laundering in the Bitcoin
  blockchain in the presence of label scarcity. ICAIF.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
- Lu, Q. & Getoor, L. (2003). Link-based Classification. ICML.
- Hanley, J. A. & McNeil, B. J. (1982). The meaning and use of the area under a ROC curve.
  Radiology.

---

### Checklist before submitting
- [x] Round-7 scores and final picks filled in (findings §27): pick 1 `submission_nbr_stack_wide_2hop_hl14.csv`, pick 2 `submission_blend_top4.csv`. Select both on Kaggle before 24 Oct 23:55 SGT.
- [ ] Repo link works for the assessors; `e1304439.ipynb` reproduces pick 1 (verified §25).
- [x] Figures 1–2 regenerated with round 7 and leader 0.97012.
- [ ] Tables fit within 4 pages; file named `e1304439.pdf`.
- [ ] Each claim in the report traces to a number in `findings.md`.
