"""Generate and execute ``e1304439.ipynb``, the submission notebook required by the brief.

The notebook is generated from this script so that its code always calls the same
``src`` functions as ``python -m src.train``. Run from the project root:

    python -m src.build_notebook            # build and execute (writes outputs into the file)
    python -m src.build_notebook --no-run   # build only
"""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("e1304439.ipynb")


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Illicit Bitcoin transaction detection (Elliptic) — final model

This notebook reproduces the final Kaggle submission end to end: data checks, feature
preparation, time-forward validation, final training and the submission file. The
experiments that led to these choices are run by the scripts in `src/` and recorded in
`report/findings.md`; their saved result tables are summarised at the end.

**Approach in brief**
* Supervised training uses only transactions with a confirmed label (`0` licit / `1` illicit).
  `none` rows are unlabeled and excluded from the supervised fit.
* `time_step` is used only to split data chronologically, never as a model input; trees cannot
  extrapolate a raw time counter into the unseen test steps 36-49.
* Validation always trains on earlier steps and scores later steps, over several windows and
  random seeds, because single-seed, single-split scores varied by up to 0.02 ROC-AUC.
* Labeled transactions cluster: 57% of an illicit transaction's labeled neighbours are illicit
  (base rate 12%). The final model adds graph context with a two-stage stack: stage-1 scores
  and features of *labeled* neighbours (one and two hops away) become features for stage 2.
  Context is restricted to labeled nodes because those are the only transactions with features
  in the test period, so every graph feature is computed the same way in training and test.
* Label-free payment-chain shape from the full edge list (longest payment path into / out of
  each transaction, and the length of the unbroken peeling chain around it) is added too.
* Training rows are weighted toward recent steps (half-life 14 steps) and toward rows that
  look like the test period (covariate-shift importance weights from a train-vs-test
  classifier, computed after removing the 30 features that most strongly encode time).
* Both stages use a tuned, shallow LightGBM; the configuration is in `src/final_config.py`
  (Kaggle public ROC-AUC 0.97196).
"""),
    code("""
from pathlib import Path
import sys

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "train.py").exists())
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import load_data
from src.evaluation import REPORT_WINDOWS, labeled_frame, window_table
from src.features import get_feature_groups
from src.final_config import FINAL_NAME, FINAL_SEEDS, FINAL_STACK
from src.graph import audit_graph_coverage
from src.graph_stacking import build_graph, two_hop_features
from src import train as final_pipeline

pd.set_option("display.width", 160)
"""),
    md("""
## 1. Data audit

`train.csv` holds steps 1-35 (labeled and unlabeled rows), `test.csv` steps 36-49, and
`txs_edgelist.csv` the payment-flow edges (an output of `txId1` is spent by `txId2`).
`load_data` validates the schema, label values, unique IDs and numeric features.
"""),
    code("""
train, test, edges, sample_submission = load_data(PROJECT_ROOT / "data")
labeled = labeled_frame(train)
groups = get_feature_groups(train.columns)

print(f"train rows: {len(train):,} | test rows: {len(test):,} | edges: {len(edges):,}")
print("labels:", train["label"].value_counts(dropna=False).rename({pd.NA: "none"}).to_dict())
print("missing feature values:", int(train[groups['all']].isna().sum().sum() + test[groups['all']].isna().sum().sum()))
print(f"labeled train + test = {len(labeled) + len(test):,} (the documented Elliptic labeled total is 46,564)")
"""),
    code("""
per_step = labeled.groupby("time_step")["label"].agg(labeled_rows="size", illicit="sum")
per_step["illicit_share"] = (per_step["illicit"] / per_step["labeled_rows"]).round(3)
per_step.T
"""),
    md("""
The test file contains exactly the labeled transactions of steps 36-49. The edge list also
references 46,668 transactions with no feature row: these are the unlabeled transactions of
steps 36-49. No edge connects two different time steps.
"""),
    code("""
audit = audit_graph_coverage(train, test, edges)
time_of = pd.concat([train.set_index("txId")["time_step"], test.set_index("txId")["time_step"]])
dedup = edges.drop_duplicates()
src_t, dst_t = dedup["txId1"].map(time_of), dedup["txId2"].map(time_of)
known = src_t.notna() & dst_t.notna()
audit["cross_time_step_edges"] = int((src_t[known] != dst_t[known]).sum())
test_ids = set(test["txId"]); feature_ids = set(train["txId"]) | test_ids
touch_test = dedup[dedup["txId1"].isin(test_ids) | dedup["txId2"].isin(test_ids)]
other = np.where(touch_test["txId1"].isin(test_ids), touch_test["txId2"], touch_test["txId1"])
audit["share_of_test_edges_to_featureless_nodes"] = round(float((~pd.Series(other).isin(feature_ids)).mean()), 3)
pd.Series(audit)
"""),
    md("""
Two-thirds of a test transaction's neighbours have no features, while every training
transaction's neighbours do. A graph neural network that reads all neighbour features
therefore meets a different input distribution at test time (GraphSAGE lowered the Kaggle
score; `report/findings.md` sections 1.3 and 7). The final model avoids this by using only
*labeled* neighbours, which have features in both periods, and by letting two-hop walks pass
through featureless nodes. Coverage of that context is similar in training and test:
"""),
    code("""
graph = build_graph(edges)
context_rows = pd.concat([labeled[["txId"]], test[["txId"]]], ignore_index=True)
scored = set(context_rows["txId"])
one_hop = graph[graph["u"].isin(scored) & graph["v"].isin(scored)].groupby("u").size()
dummy = context_rows.assign(**{c: 0.0 for c in ["feat_3", "feat_5", "feat_53", "feat_59", "feat_61"]})
two_hop = two_hop_features(graph, dummy, pd.Series(0.0, index=context_rows["txId"]))["hop2_labeled_walks"]
coverage = {}
for name, frame in [("labeled training rows", labeled), ("test rows", test)]:
    has_one = one_hop.reindex(frame["txId"]).fillna(0).to_numpy() > 0
    has_two = two_hop.reindex(frame["txId"]).fillna(0).to_numpy() > 0
    coverage[name] = {"1-hop labeled neighbour": has_one.mean(), "2-hop labeled node": has_two.mean(),
                      "2-hop only": (has_two & ~has_one).mean(), "no graph context": (~has_one & ~has_two).mean()}
pd.DataFrame(coverage).T.round(3)
"""),
    md("""
## 2. Features and distribution shift

Per the Elliptic documentation, `feat_1`-`feat_93` describe the transaction itself (local) and
`feat_94`-`feat_165` aggregate its one-hop neighbourhood. The table measures how far each group's
medians move, in units of the earlier period's interquartile range.
"""),
    code("""
def median_shift(reference, target, columns):
    iqr = (reference[columns].quantile(0.75) - reference[columns].quantile(0.25)).replace(0, np.nan)
    return ((target[columns].median() - reference[columns].median()).abs() / iqr).fillna(0)

rows = []
comparisons = {
    "steps<=28 -> 29-35 (validation)": (labeled[labeled.time_step <= 28], labeled[labeled.time_step > 28]),
    "steps<=35 -> test 36-49": (labeled, test),
}
for name, (reference, target) in comparisons.items():
    for group in ("local", "aggregated"):
        shift = median_shift(reference, target, groups[group])
        rows.append({"comparison": name, "group": group, "features": len(groups[group]),
                     "mean_shift_IQR": round(shift.mean(), 3), "share_shift_over_1_IQR": round((shift > 1).mean(), 3)})
pd.DataFrame(rows)
"""),
    md("""
The aggregated features drift about twice as much into the test period as within the training
period. A model built only on aggregated features scored 0.880 on Kaggle versus 0.943 for all
features, so the final model keeps all supplied features and relies on regularisation rather
than on the aggregated group.

## 3. Final model configuration

`graph_stacking.make_neighbour_stack(**FINAL_STACK)`:
1. **Stage 1** — tuned LightGBM on the 165 supplied features scores labeled training rows
   out-of-fold (5 folds of consecutive time steps) and test rows with the mean of the 5 fold
   models.
2. **Graph context** — for every transaction: mean / max / incoming-mean / outgoing-mean of
   stage-1 scores over labeled neighbours; in/out degree; number and share of labeled
   neighbours; mean / max / std / incoming-mean / outgoing-mean over labeled neighbours of the
   25 most influential supplied features; and walk-weighted two-hop means of the stage-1 score
   and of the same 25 features; and label-free chain shape from the full
   edge list: longest payment path into / out of the transaction, a peel-like flag (one funder,
   at most two spenders) and the length of the unbroken peel run before / after it.
   Contrast features: for each of the 25 key features, the transaction's own value minus the
   mean over its labeled neighbours.
3. **Stage 2** — the same LightGBM on supplied features + graph context, except that each tree
   samples a share of columns scaled by 165 / (number of stage-2 columns), so it sees about as
   many columns (about 106) as a stage-1 tree.
4. Both stages weight training rows by 2^((t - last step)/14) times a covariate-shift
   importance weight: a LightGBM separates labeled training rows from test rows (5-fold
   out-of-fold, no labels) after three rounds of dropping the 10 features with the largest
   separating gain; weight = sqrt(p / (1 - p)), mean one, clipped to [0.2, 5]. Classes are
   balanced; seeds 0-9 are averaged.
"""),
    code("""
from src.model_zoo import TUNED_LGBM_PARAMS
print("final model:", FINAL_NAME, "| stack options:", FINAL_STACK, "| seeds:", len(FINAL_SEEDS))
pd.Series(TUNED_LGBM_PARAMS, name="LightGBM settings (both stages)")
"""),
    md("""
## 4. Time-forward validation

Each window trains on labeled rows up to a step and scores a later block. Three seeds per window
are averaged (seed bagging, as in the submission). The reference is the tuned LightGBM without
graph context (Kaggle 0.94417). The short windows (train <= 25 or <= 28) are the ones whose
model ranking matched the Kaggle leaderboard; the graph stack wins there but loses on the long
windows, and the Kaggle leaderboard (+0.0225 over the reference) sided with the short windows.
Local windows cannot reproduce the test period's drift, so they serve here as a reproducibility
check rather than as the selection criterion (`report/findings.md` sections 11, 14, 20).
Runtime: several minutes.
"""),
    code("""
tables = final_pipeline.validate(labeled, graph, seeds=(0, 1, 2), output_dir=PROJECT_ROOT / "artifacts" / "final")
window_table(tables["window_metrics"]).round(4)
"""),
    code("""
tables["summary"].round(4)
"""),
    md("""
## 5. Final training and submission

Both stages are refitted on all 31,235 labeled rows from steps 1-35 for each of the ten seeds,
and the seed predictions for the test rows are averaged. The writer enforces the `index,target`
format and the sample-submission row order.
"""),
    code("""
target = final_pipeline.stack_predict(labeled, test, graph, FINAL_SEEDS)
submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": target})
path = final_pipeline.write_submission(submission, PROJECT_ROOT / "submissions" / "submission_final.csv",
                                       sample_submission=sample_submission)
print(path.relative_to(PROJECT_ROOT), submission.shape)
submission.describe().T
"""),
    md("""
## 6. Experiment record

Summary tables saved by the experiment scripts (if present locally). Full commentary, including
the Kaggle scores of each submitted variant, is in `report/findings.md`.
"""),
    code("""
for label, relative in [
    ("Step 1 - harness comparison", "artifacts/harness/step1_summary.csv"),
    ("Step 4 - tuned models and ensembles", "artifacts/tuning/step4_summary.csv"),
    ("Step 5 - stress tests", "artifacts/stress/stress_summary.csv"),
    ("Step 6 - GraphSAGE", "artifacts/extensions/gnn_summary.csv"),
    ("Step 6 - pseudo-labeling", "artifacts/extensions/pseudo_summary.csv"),
    ("Rounds 3-5 - graph stacking variants (local check)", "artifacts/round3/round5_window_table.csv"),
]:
    path = PROJECT_ROOT / relative
    if path.exists():
        print(f"\\n=== {label} ({relative}) ===")
        print(pd.read_csv(path).round(4).head(15).to_string(index=False))
"""),
]


def build(run=True):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = CELLS
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if run:
        NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    print(f"Wrote {build(run=not args.no_run)}")


if __name__ == "__main__":
    main()
