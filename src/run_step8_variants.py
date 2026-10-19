"""Step 8: round-3 variants built on the tuned all-feature LightGBM (tuned_lgbm_all).

Local windows are saturated (short windows score ~0.996 while Kaggle scores ~0.944), so each
variant gets a local sanity check here (it must not lose more than 0.01 on any report window)
and is then judged on the Kaggle public leaderboard. Variants:

``recent_K``      train only on the last K labeled time steps before the cut-off
``halflife_H``    exponential recency weights with half-life H steps (mean-one weights)
``temporal_bag``  rank-average of models trained on all steps, the last 25 and the last 15
``nbr_stack_wide_2hop_chain_shift_hl14``
                  round-12: final (chain) model + covariate-shift importance weights
``nbr_stack_wide_2hop_hub_hl14`` / ``_chain_`` / ``_cent_``
                  round-11: pick 1 + degree-down-weighted neighbour means; + peel-chain
                  shape; + PageRank / reach / neighbour-degree centrality
``nbr_stack_wide_2hop_hl14_fwd``
                  round-10: stage-1 neighbour scores for training rows come from models fitted
                  on earlier steps only, matching how target rows are scored
``nbr_stack_wide_2hop_dir_hl14``
                  round-9: + two-hop score means split by payment direction
``nbr_stack_wide_2hop_hl14_iter``
                  round-8: one extra round of collective classification (context rebuilt
                  from out-of-fold stage-2 scores, then a third stage)
``nbr_stack_wide_2hop_hl21``
                  round-7: recency sensitivity of the best model (half-life 21)
``nbr_stack_wide_2hop_hl14`` / ``nbr_stack_rich_2hop_prop_hl14``
                  round-6: wide + two-hop; two-hop + 3-hop score propagation
``nbr_stack_rich_hl14`` / ``nbr_stack_wide_hl14`` / ``nbr_stack_rich_2hop_hl14``
                  round-5: rich + recency; wide 25-feature neighbour context; two-hop context
``nbr_stack_hl14`` / ``nbr_stack_rich``
                  round-4 extensions of nbr_stack: recency weights (half-life 14) in both
                  stages; richer neighbour context with fold-averaged target scores
``nbr_stack``     two-stage stacking over *labeled* neighbours: stage-1 scores (out-of-fold for
                  training rows, grouped by time step) are aggregated over each transaction's
                  labeled neighbours (mean, max, incoming mean, outgoing mean) and added as
                  features for stage 2. In the test period only labeled transactions have
                  features, so restricting to labeled neighbours keeps train and test consistent
                  (report/findings.md section 11.1).

The stacking implementation lives in ``src/graph_stacking.py``.

Usage:
    python -m src.run_step8_variants evaluate            # local sanity check, 2 seeds
    python -m src.run_step8_variants submit NAME [NAME]  # 10-seed Kaggle submission files
    python -m src.run_step8_variants blend OUT NAME NAME # rank-average existing submission files
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
from src.graph_stacking import (  # noqa: F401  (re-exported for tests and experiments)
    RICH_NEIGHBOUR_FEATURES,
    STACK_FOLDS,
    WIDE_NEIGHBOUR_FEATURES,
    _fit,
    build_graph,
    make_neighbour_stack,
    neighbour_context_features,
    neighbour_score_features,
    neighbour_stack,
    neighbour_wide_features,
    propagation_features,
    two_hop_features,
)
from src.run_lgbm_robustness_experiments import exponential_recency_weights
from src.submission import write_submission

OUTPUT_DIR = Path("artifacts/round3")
EVAL_SEEDS = (0, 1)
SUBMIT_SEEDS = tuple(range(10))


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


def variants():
    registry = {"tuned_lgbm_all": recent_steps(10_000)}
    for k in (21, 14):
        registry[f"recent_{k}"] = recent_steps(k)
    for h in (7, 14):
        registry[f"halflife_{h}"] = half_life(h)
    registry["temporal_bag"] = temporal_bag()
    registry["nbr_stack"] = neighbour_stack
    registry["nbr_stack_hl14"] = make_neighbour_stack(half_life_steps=14)
    registry["nbr_stack_rich"] = make_neighbour_stack(rich=True)
    registry["nbr_stack_rich_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True)
    registry["nbr_stack_wide_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True, wide=True)
    registry["nbr_stack_rich_2hop_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True, two_hop=True)
    registry["nbr_stack_wide_2hop_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True, wide=True, two_hop=True)
    registry["nbr_stack_wide_2hop_hl21"] = make_neighbour_stack(half_life_steps=21, rich=True, wide=True, two_hop=True)
    registry["nbr_stack_wide_2hop_dir_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True, wide=True, two_hop=True, directional=True)
    registry["nbr_stack_wide_2hop_hl14_iter"] = make_neighbour_stack(half_life_steps=14, rich=True, wide=True, two_hop=True, extra_rounds=1)
    registry["nbr_stack_wide_2hop_hl14_fwd"] = make_neighbour_stack(half_life_steps=14, rich=True, wide=True, two_hop=True, forward_oof=True)
    pick1 = dict(half_life_steps=14, rich=True, wide=True, two_hop=True)
    registry["nbr_stack_wide_2hop_hub_hl14"] = make_neighbour_stack(**pick1, hub=True)
    registry["nbr_stack_wide_2hop_chain_hl14"] = make_neighbour_stack(**pick1, chain=True)
    registry["nbr_stack_wide_2hop_cent_hl14"] = make_neighbour_stack(**pick1, centrality=True)
    registry["nbr_stack_wide_2hop_chain_shift_hl14"] = make_neighbour_stack(**pick1, chain=True, shift_weights=True)
    registry["nbr_stack_rich_2hop_prop_hl14"] = make_neighbour_stack(half_life_steps=14, rich=True, two_hop=True, propagate=True)
    return registry


def evaluate(labeled, columns, graph, names=None, prefix="round3"):
    times = labeled["time_step"].to_numpy()
    registry = variants()
    if names:
        registry = {name: registry[name] for name in ["tuned_lgbm_all", *names]}
    frames = []
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        train, target = labeled.iloc[train_pos], labeled.iloc[valid_pos]
        for name, fit_predict in registry.items():
            for seed in EVAL_SEEDS:
                frames.append(pd.DataFrame({
                    "candidate": name, "window": window_name, "seed": seed, "position": valid_pos,
                    "time_step": times[valid_pos], "label": target["label"].to_numpy(),
                    "probability": fit_predict(train, target, columns, seed, graph),
                }))
        print(f"  {window_name}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    tables = evaluate_predictions(predictions, "tuned_lgbm_all", OUTPUT_DIR, prefix, bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    table.to_csv(OUTPUT_DIR / f"{prefix}_window_table.csv")
    print(table.round(4).to_string())
    print(tables["summary"].round(4).to_string(index=False))


def submit(labeled, test, columns, graph, names, sample_submission, seeds=SUBMIT_SEEDS):
    registry = variants()
    suffix = "" if tuple(seeds) == SUBMIT_SEEDS else f"_s{len(seeds)}"
    for name in names:
        scores = np.mean([registry[name](labeled, test, columns, seed, graph) for seed in seeds], axis=0)
        submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": scores})
        path = write_submission(submission, Path("submissions") / f"submission_{name}{suffix}.csv", sample_submission=sample_submission)
        print(f"Wrote {path}", flush=True)


def blend_files(output_name, names, sample_submission):
    """Equal-weight rank average of existing ``submissions/submission_<name>.csv`` files."""
    frames = [pd.read_csv(Path("submissions") / f"submission_{name}.csv") for name in names]
    for frame in frames:
        if not frame["index"].equals(frames[0]["index"]):
            raise ValueError("Submission files to blend must share the same index order")
    ranks = np.mean([frame["target"].rank(pct=True).to_numpy() for frame in frames], axis=0)
    submission = pd.DataFrame({"index": frames[0]["index"].to_numpy(), "target": ranks})
    path = write_submission(submission, Path("submissions") / f"submission_{output_name}.csv", sample_submission=sample_submission)
    print(f"Wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "submit", "blend"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--prefix", default=None, help="output table prefix for evaluate (default round3/round4)")
    parser.add_argument("--seeds", type=int, default=len(SUBMIT_SEEDS), help="submit: number of seeds (0..N-1); files get an _sN suffix unless N=10")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, edges, sample_submission = load_data()
    labeled = labeled_frame(train)
    columns = get_feature_groups(train.columns)["all"]
    graph = build_graph(edges)
    if args.command == "blend":  # blend OUTPUT_NAME NAME1 NAME2 ...
        blend_files(args.names[0], args.names[1:], sample_submission)
    elif args.command == "evaluate":
        evaluate(labeled, columns, graph, args.names, prefix=args.prefix or ("round4" if args.names else "round3"))
    else:
        submit(labeled, test, columns, graph, args.names, sample_submission, seeds=tuple(range(args.seeds)))


if __name__ == "__main__":
    main()
