"""Step 6: graph model under test-consistent masking, and pseudo-labeling.

``gnn``     Directed GraphSAGE (``gnn_v2``) on every report window, three seeds, in two
            regimes. ``masked``: unlabeled nodes are blank in every step, matching the test
            period. ``old_regime``: unlabeled nodes keep features during training steps but
            are blank in validation steps, reproducing what the first GraphSAGE faced at
            test time. Each is scored alone and in fixed 0.2 / 0.3 rank blends with
            ``shallow_all``.

``pseudo``  Self-training for ``shallow_all``: score the unlabeled rows of the training
            steps (never validation or test steps), add confident ones as pseudo-labels with
            weight 0.5, and refit. Two pre-declared confidence thresholds are tested.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import (
    REPORT_WINDOWS,
    Candidate,
    add_rank_blend,
    evaluate_predictions,
    labeled_frame,
    run_candidates,
    window_positions,
    window_table,
)
from src.features import get_feature_groups
from src.gnn_v2 import train_and_predict
from src.graph import build_union_graph_data
from src.model_zoo import lgbm_factory

OUTPUT_DIR = Path("artifacts/extensions")
SEEDS = (0, 1, 2)
PSEUDO_THRESHOLDS = {"pseudo_strict": (0.95, 0.02), "pseudo_loose": (0.80, 0.10)}
PSEUDO_WEIGHT = 0.5


def _old_regime_graph(graph_data, train_end):
    """Blank unlabeled nodes only after ``train_end`` (the first GraphSAGE's situation)."""
    node_df = graph_data["node_df"].copy()
    times = node_df["time_step"].astype("float").to_numpy()
    # Pretend training-period unlabeled nodes are labeled so masking keeps their features.
    keep = (times <= train_end) & node_df["label"].isna().to_numpy()
    node_df["is_test"] = node_df["is_test"] | keep
    return {**graph_data, "node_df": node_df}


def run_gnn_experiment(train, test, edges, labeled, groups):
    graph_data = build_union_graph_data(train, test, edges)
    node_index = pd.Series(graph_data["node_df"].index.to_numpy(), index=graph_data["node_df"]["txId"])
    labeled_nodes = node_index.loc[labeled["txId"]].to_numpy()
    times = labeled["time_step"].to_numpy()
    frames = []
    for window_name, window in REPORT_WINDOWS.items():
        _, valid_pos = window_positions(times, window)
        for regime in ("masked", "old_regime"):
            data = graph_data if regime == "masked" else _old_regime_graph(graph_data, window[0])
            for seed in SEEDS:
                probabilities = train_and_predict(data, train_end=window[0], seed=seed)
                frames.append(pd.DataFrame({
                    "candidate": f"sage_{regime}", "window": window_name, "seed": seed,
                    "position": valid_pos, "time_step": times[valid_pos],
                    "label": labeled["label"].to_numpy()[valid_pos],
                    "probability": probabilities[labeled_nodes[valid_pos]],
                }))
            print(f"  {window_name} sage_{regime}", flush=True)
    reference = run_candidates(labeled, [Candidate("shallow_all", tuple(groups["all"]), lgbm_factory())], REPORT_WINDOWS, SEEDS, verbose=False)
    predictions = pd.concat([reference, *frames], ignore_index=True)
    for regime in ("masked", "old_regime"):
        for weight in (0.2, 0.3):
            predictions = add_rank_blend(
                predictions, {"shallow_all": 1 - weight, f"sage_{regime}": weight}, f"blend_sage_{regime}_{weight}"
            )
    tables = evaluate_predictions(predictions, "shallow_all", OUTPUT_DIR, "gnn", bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    table.to_csv(OUTPUT_DIR / "gnn_window_table.csv")
    print(table.round(4).to_string())
    print(tables["summary"].round(4).to_string(index=False))


def run_pseudo(train, labeled, groups):
    columns = list(groups["all"])
    candidate = Candidate("shallow_all", tuple(columns), lgbm_factory())
    make = candidate.make_model
    unlabeled = train.loc[train["label"].isna()]
    times = labeled["time_step"].to_numpy()
    labels = labeled["label"].to_numpy()
    frames, counts = [], []
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        pool = unlabeled.loc[unlabeled["time_step"] <= window[0], columns]
        X_train, y_train = labeled.iloc[train_pos][columns], labels[train_pos]
        X_valid = labeled.iloc[valid_pos][columns]
        for seed in SEEDS:
            base = make(seed).fit(X_train, y_train)
            outputs = {"shallow_all": base.predict_proba(X_valid)[:, 1]}
            pool_scores = base.predict_proba(pool)[:, 1]
            for name, (high, low) in PSEUDO_THRESHOLDS.items():
                pos, neg = pool_scores >= high, pool_scores <= low
                X_aug = pd.concat([X_train, pool.loc[pos], pool.loc[neg]], ignore_index=True)
                y_aug = np.r_[y_train, np.ones(pos.sum(), int), np.zeros(neg.sum(), int)]
                weight = np.r_[np.ones(len(y_train)), np.full(pos.sum() + neg.sum(), PSEUDO_WEIGHT)]
                outputs[name] = make(seed).fit(X_aug, y_aug, sample_weight=weight).predict_proba(X_valid)[:, 1]
                counts.append({"window": window_name, "seed": seed, "variant": name, "pseudo_pos": int(pos.sum()), "pseudo_neg": int(neg.sum()), "pool": len(pool)})
            for name, probability in outputs.items():
                frames.append(pd.DataFrame({
                    "candidate": name, "window": window_name, "seed": seed, "position": valid_pos,
                    "time_step": times[valid_pos], "label": labels[valid_pos], "probability": probability,
                }))
        print(f"  {window_name}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    pd.DataFrame(counts).to_csv(OUTPUT_DIR / "pseudo_label_counts.csv", index=False)
    tables = evaluate_predictions(predictions, "shallow_all", OUTPUT_DIR, "pseudo", bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    table.to_csv(OUTPUT_DIR / "pseudo_window_table.csv")
    print(pd.DataFrame(counts).groupby("variant")[["pseudo_pos", "pseudo_neg", "pool"]].mean().round(0).to_string())
    print(table.round(4).to_string())
    print(tables["summary"].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", choices=["gnn", "pseudo"])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, edges, _ = load_data()
    labeled = labeled_frame(train)
    groups = get_feature_groups(train.columns)
    for experiment in args.experiments:
        print(f"\n===== {experiment} =====", flush=True)
        if experiment == "gnn":
            run_gnn_experiment(train, test, edges, labeled, groups)
        else:
            run_pseudo(train, labeled, groups)


if __name__ == "__main__":
    main()
