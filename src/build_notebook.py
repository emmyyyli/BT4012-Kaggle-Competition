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
* The final model is a regularised gradient-boosted tree model on the supplied features; the
  configuration lives in `src/final_config.py`.
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
from src.final_config import FINAL_MEMBERS, FINAL_NAME, FINAL_SEEDS
from src.graph import audit_graph_coverage
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
transaction's neighbours do. A graph neural network that reads neighbour features therefore
meets a different input distribution at test time; this is why the graph models tested were
not used in the final submission (`report/findings.md`, sections 1.3 and 7).

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
"""),
    code("""
print("final model:", FINAL_NAME, "| seeds per member:", len(FINAL_SEEDS))
pd.DataFrame([{k: v for k, v in m.items() if k != "params"} | {"params": m["params"]} for m in FINAL_MEMBERS])
"""),
    md("""
## 4. Time-forward validation

Each window trains on labeled rows up to a step and scores a later block. Three seeds per window
are averaged (seed bagging, as in the submission). The reference is the original production
LightGBM (31 leaves, 800 trees, 90% column sampling). The short windows (train <= 25 or <= 28)
are the ones whose model ranking matched the Kaggle leaderboard.
"""),
    code("""
candidates, weights = final_pipeline.build_members(train.columns)
tables = final_pipeline.validate(labeled, candidates, weights, seeds=(0, 1, 2), output_dir=PROJECT_ROOT / "artifacts" / "final")
window_table(tables["window_metrics"]).round(4)
"""),
    code("""
tables["summary"].round(4)
"""),
    md("""
## 5. Final training and submission

Every member is refitted on all 31,235 labeled rows from steps 1-35 with each seed; seed
predictions are averaged, and members (if several) are rank-averaged with fixed weights. The
writer enforces the `index,target` format and the sample-submission row order.
"""),
    code("""
target, fitted = final_pipeline.fit_and_predict(labeled, test, candidates, weights, FINAL_SEEDS)
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
