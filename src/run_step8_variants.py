"""Step 8: round-3 variants built on the tuned all-feature LightGBM (tuned_lgbm_all).

Local windows are saturated (short windows score ~0.996 while Kaggle scores ~0.944), so each
variant gets a local sanity check here (it must not lose more than 0.01 on any report window)
and is then judged on the Kaggle public leaderboard. Variants:

``recent_K``      train only on the last K labeled time steps before the cut-off
``halflife_H``    exponential recency weights with half-life H steps (mean-one weights)
``temporal_bag``  rank-average of models trained on all steps, the last 25 and the last 15
``nbr_stack``     two-stage stacking over *labeled* neighbours: stage-1 scores (out-of-fold for
                  training rows, grouped by time step) are aggregated over each transaction's
                  labeled neighbours (mean, max, incoming mean, outgoing mean) and added as
                  features for stage 2. In the test period only labeled transactions have
                  features, so restricting to labeled neighbours keeps train and test consistent
                  (report/findings.md section 11.1).

Usage:
    python -m src.run_step8_variants evaluate            # local sanity check, 2 seeds
    python -m src.run_step8_variants submit NAME [NAME]  # 10-seed Kaggle submission files
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import REPORT_WINDOWS, evaluate_predictions, labeled_frame, window_positions, window_table
from src.features import get_feature_groups
from src.final_config import TUNED_LGBM_PARAMS
from src.model_zoo import lgbm_factory
from src.run_lgbm_robustness_experiments import exponential_recency_weights
from src.submission import write_submission

OUTPUT_DIR = Path("artifacts/round3")
EVAL_SEEDS = (0, 1)
SUBMIT_SEEDS = tuple(range(10))
STACK_FOLDS = 5


def _fit(frame, columns, seed, weights=None):
    model = lgbm_factory(TUNED_LGBM_PARAMS)(seed)
    kwargs = {} if weights is None else {"sample_weight": weights}
    return model.fit(frame[columns], frame["label"].to_numpy(), **kwargs)


def recent_steps(last_k):
    def fit_predict(train, target, columns, seed, graph=None):
        cutoff = train["time_step"].max() - last_k
        subset = train.loc[train["time_step"] > cutoff]
        return _fit(subset, columns, seed).predict_proba(target[columns])[:, 1]
    return fit_predict


def half_life(steps):
    def fit_predict(train, target, columns, seed, graph=None):
        weights = exponential_recency_weights(train["time_step"].to_numpy(), steps)
        return _fit(train, columns, seed, weights).predict_proba(target[columns])[:, 1]
    return fit_predict


def temporal_bag(lengths=(None, 25, 15)):
    """Rank-average models trained on all steps and on the last ``k`` steps."""
    members = [recent_steps(k) if k else recent_steps(10_000) for k in lengths]

    def fit_predict(train, target, columns, seed, graph=None):
        ranks = [pd.Series(m(train, target, columns, seed)).rank(pct=True).to_numpy() for m in members]
        return np.mean(ranks, axis=0)
    return fit_predict


def build_graph(edges):
    """Directed edge table in both orientations, tagged with the direction seen from ``u``."""
    edges = edges[["txId1", "txId2"]].drop_duplicates()
    outgoing = edges.rename(columns={"txId1": "u", "txId2": "v"}).assign(direction="out")
    incoming = edges.rename(columns={"txId2": "u", "txId1": "v"}).assign(direction="in")
    return pd.concat([outgoing, incoming], ignore_index=True)


def neighbour_score_features(graph, scores):
    """Aggregate ``scores`` (Series indexed by txId) over each node's scored neighbours."""
    linked = graph.loc[graph["u"].isin(scores.index) & graph["v"].isin(scores.index)].copy()
    linked["score"] = linked["v"].map(scores)
    grouped = linked.groupby("u")["score"]
    features = pd.DataFrame({"nbr_score_mean": grouped.mean(), "nbr_score_max": grouped.max()})
    for direction in ("in", "out"):
        part = linked.loc[linked["direction"] == direction].groupby("u")["score"].mean()
        features[f"nbr_score_mean_{direction}"] = part
    return features.reindex(scores.index)  # NaN where a node has no scored neighbour


def neighbour_stack(train, target, columns, seed, graph):
    """Two-stage stacking; neighbours are restricted to rows in ``train`` or ``target``."""
    train = train.reset_index(drop=True)
    target = target.reset_index(drop=True)
    oof = np.full(len(train), np.nan)
    steps = np.sort(train["time_step"].unique())
    for fold_steps in np.array_split(steps, STACK_FOLDS):
        held = train["time_step"].isin(fold_steps).to_numpy()
        oof[held] = _fit(train.loc[~held], columns, seed).predict_proba(train.loc[held, columns])[:, 1]
    target_scores = _fit(train, columns, seed).predict_proba(target[columns])[:, 1]
    scores = pd.concat([pd.Series(oof, index=train["txId"]), pd.Series(target_scores, index=target["txId"])])
    neighbour = neighbour_score_features(graph, scores)
    stacked_columns = [*columns, *neighbour.columns]
    train_x = pd.concat([train[columns], neighbour.loc[train["txId"]].reset_index(drop=True)], axis=1)
    target_x = pd.concat([target[columns], neighbour.loc[target["txId"]].reset_index(drop=True)], axis=1)
    model = lgbm_factory(TUNED_LGBM_PARAMS)(seed).fit(train_x[stacked_columns], train["label"].to_numpy())
    return model.predict_proba(target_x[stacked_columns])[:, 1]


def variants():
    registry = {"tuned_lgbm_all": recent_steps(10_000)}
    for k in (21, 14):
        registry[f"recent_{k}"] = recent_steps(k)
    for h in (7, 14):
        registry[f"halflife_{h}"] = half_life(h)
    registry["temporal_bag"] = temporal_bag()
    registry["nbr_stack"] = neighbour_stack
    return registry


def evaluate(labeled, columns, graph):
    times = labeled["time_step"].to_numpy()
    frames = []
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        train, target = labeled.iloc[train_pos], labeled.iloc[valid_pos]
        for name, fit_predict in variants().items():
            for seed in EVAL_SEEDS:
                frames.append(pd.DataFrame({
                    "candidate": name, "window": window_name, "seed": seed, "position": valid_pos,
                    "time_step": times[valid_pos], "label": target["label"].to_numpy(),
                    "probability": fit_predict(train, target, columns, seed, graph),
                }))
        print(f"  {window_name}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    tables = evaluate_predictions(predictions, "tuned_lgbm_all", OUTPUT_DIR, "round3", bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    table.to_csv(OUTPUT_DIR / "round3_window_table.csv")
    print(table.round(4).to_string())
    print(tables["summary"].round(4).to_string(index=False))


def submit(labeled, test, columns, graph, names, sample_submission):
    registry = variants()
    for name in names:
        scores = np.mean([registry[name](labeled, test, columns, seed, graph) for seed in SUBMIT_SEEDS], axis=0)
        submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": scores})
        path = write_submission(submission, Path("submissions") / f"submission_{name}.csv", sample_submission=sample_submission)
        print(f"Wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "submit"])
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, edges, sample_submission = load_data()
    labeled = labeled_frame(train)
    columns = get_feature_groups(train.columns)["all"]
    graph = build_graph(edges)
    if args.command == "evaluate":
        evaluate(labeled, columns, graph)
    else:
        submit(labeled, test, columns, graph, args.names, sample_submission)


if __name__ == "__main__":
    main()
