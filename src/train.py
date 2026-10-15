"""End-to-end final pipeline: validate the configured model, refit, write the submission.

1. Load and check the competition files (``src.data.load_data``).
2. Build the directed transaction graph from the edge list (``src.graph_stacking``).
3. Validate with the temporal harness (``src.evaluation``): five forward windows, several seeds,
   seed-bagged ROC-AUC, alongside the non-graph tuned LightGBM reference.
4. Run the final graph-stacking model (``src.final_config``) on all labeled rows (steps 1-35)
   for every seed in ``FINAL_SEEDS`` and average the test predictions.
5. Write ``submissions/submission_final.csv`` in the sample-submission order.

Run from the project root: ``python -m src.train`` (``--validation-seeds`` with no values
skips step 3).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_DIR, PROJECT_ROOT, load_data
from src.evaluation import (
    REPORT_WINDOWS,
    Candidate,
    evaluate_predictions,
    fit_model,
    labeled_frame,
    run_candidates,
    window_positions,
    window_table,
)
from src.features import get_feature_groups
from src.final_config import FINAL_NAME, FINAL_SEEDS, FINAL_STACK, REFERENCE_MEMBERS
from src.graph_stacking import build_graph, make_neighbour_stack
from src.model_zoo import catboost_factory, lgbm_factory, xgb_factory
from src.submission import write_submission

FACTORIES = {"lgbm": lgbm_factory, "xgb": xgb_factory, "catboost": catboost_factory}


def build_members(columns, members=REFERENCE_MEMBERS):
    """Turn single-model config dictionaries into harness ``Candidate`` objects and weights."""
    groups = get_feature_groups(columns)
    candidates, weights = [], {}
    for spec in members:
        dropped = set(spec.get("drop", []))
        feature_columns = tuple(c for c in groups[spec["feature_set"]] if c not in dropped)
        factory = FACTORIES[spec["family"]](spec["params"])
        candidates.append(Candidate(spec["name"], feature_columns, factory))
        weights[spec["name"]] = spec["weight"]
    return candidates, weights


def fit_and_predict(labeled, test, candidates, weights, seeds):
    """Refit single-model members on all labeled rows per seed; rank-average if several."""
    member_scores, fitted = {}, {}
    for candidate in candidates:
        columns = list(candidate.columns)
        models = [
            fit_model(candidate, labeled[columns], labeled["label"].to_numpy(), seed, labeled["time_step"].to_numpy())
            for seed in seeds
        ]
        fitted[candidate.name] = {"columns": columns, "models": models}
        member_scores[candidate.name] = np.mean([m.predict_proba(test[columns])[:, 1] for m in models], axis=0)
    if len(candidates) == 1:
        return member_scores[candidates[0].name], fitted
    total = sum(weights.values())
    ranks = {name: pd.Series(score).rank(pct=True).to_numpy() for name, score in member_scores.items()}
    return sum(weights[name] * ranks[name] for name in weights) / total, fitted


def stack_predict(train, target, graph, seeds, stack=FINAL_STACK):
    """Seed-averaged graph-stacking predictions for ``target`` rows."""
    columns = get_feature_groups(train.columns)["all"]
    model = make_neighbour_stack(**stack)
    return np.mean([model(train, target, columns, seed, graph) for seed in seeds], axis=0)


def validate(labeled, graph, seeds, output_dir):
    """Score the final stack and the non-graph reference on the report windows."""
    candidates, _ = build_members(labeled.columns)
    reference = run_candidates(labeled, candidates, REPORT_WINDOWS, seeds, verbose=False)
    times = labeled["time_step"].to_numpy()
    columns = get_feature_groups(labeled.columns)["all"]
    model = make_neighbour_stack(**FINAL_STACK)
    frames = [reference]
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        train, target = labeled.iloc[train_pos], labeled.iloc[valid_pos]
        for seed in seeds:
            frames.append(pd.DataFrame({
                "candidate": FINAL_NAME, "window": window_name, "seed": seed, "position": valid_pos,
                "time_step": times[valid_pos], "label": target["label"].to_numpy(),
                "probability": model(train, target, columns, seed, graph),
            }))
    predictions = pd.concat(frames, ignore_index=True)
    return evaluate_predictions(predictions, candidates[0].name, output_dir, "final_validation", bootstrap_iterations=200)


def run_pipeline(data_dir=DATA_DIR, output_dir=None, validation_seeds=(0, 1, 2), seeds=FINAL_SEEDS):
    output_dir = PROJECT_ROOT if output_dir is None else Path(output_dir)
    train, test, edges, sample_submission = load_data(data_dir)
    labeled = labeled_frame(train)
    graph = build_graph(edges)

    tables = None
    if validation_seeds:
        tables = validate(labeled, graph, tuple(validation_seeds), output_dir / "artifacts" / "final")

    target = stack_predict(labeled, test, graph, seeds)
    submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": target})
    submission_path = write_submission(
        submission, output_dir / "submissions" / "submission_final.csv", sample_submission=sample_submission
    )
    config_path = output_dir / "models" / "final_model_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"name": FINAL_NAME, "stack": FINAL_STACK, "seeds": list(seeds)}, indent=2))
    return tables, submission, submission_path, config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()
    tables, submission, submission_path, config_path = run_pipeline(validation_seeds=tuple(args.validation_seeds))
    if tables is not None:
        print("=== Seed-bagged ROC-AUC by forward window ===")
        print(window_table(tables["window_metrics"]).round(4).to_string())
        print("\n=== Summary vs non-graph tuned LightGBM ===")
        print(tables["summary"].round(4).to_string(index=False))
    print(f"\nFinal model: {FINAL_NAME}; configuration {config_path}")
    print(f"Saved submission: {submission_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
