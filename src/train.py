"""End-to-end final pipeline: validate the configured final model, refit, write the submission.

1. Load and check the competition files (``src.data.load_data``).
2. Build each configured member from ``src.final_config`` (feature set + model settings).
3. Validate with the temporal harness (``src.evaluation``): five forward windows, several
   seeds, seed-bagged ROC-AUC, compared with the original production LightGBM.
4. Refit every member on all labeled rows (steps 1-35) with ``FINAL_SEEDS`` and average seeds.
5. Rank-average members (if more than one) and write ``submissions/submission_final.csv``.

Run from the project root: ``python -m src.train`` (``--validation-seeds 0`` skips step 3).
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_DIR, PROJECT_ROOT, load_data
from src.evaluation import (
    REPORT_WINDOWS,
    Candidate,
    add_rank_blend,
    evaluate_predictions,
    fit_model,
    labeled_frame,
    run_candidates,
    window_table,
)
from src.features import get_feature_groups
from src.final_config import FINAL_MEMBERS, FINAL_NAME, FINAL_SEEDS
from src.model_zoo import baseline_lgbm_factory, catboost_factory, lgbm_factory, xgb_factory
from src.submission import write_submission

FACTORIES = {"lgbm": lgbm_factory, "xgb": xgb_factory, "catboost": catboost_factory}


def build_members(columns, members=FINAL_MEMBERS):
    """Turn config dictionaries into harness ``Candidate`` objects and blend weights."""
    groups = get_feature_groups(columns)
    candidates, weights = [], {}
    for spec in members:
        dropped = set(spec.get("drop", []))
        feature_columns = tuple(c for c in groups[spec["feature_set"]] if c not in dropped)
        factory = FACTORIES[spec["family"]](spec["params"])
        candidates.append(Candidate(spec["name"], feature_columns, factory))
        weights[spec["name"]] = spec["weight"]
    return candidates, weights


def validate(labeled, candidates, weights, seeds, output_dir):
    """Score the final model against the original production LightGBM on report windows."""
    reference = Candidate("original_lgbm", tuple(get_feature_groups(labeled.columns)["all"]), baseline_lgbm_factory())
    predictions = run_candidates(labeled, [reference, *candidates], REPORT_WINDOWS, seeds, verbose=False)
    if len(candidates) > 1:
        predictions = add_rank_blend(predictions, weights, FINAL_NAME)
    tables = evaluate_predictions(predictions, "original_lgbm", output_dir, "final_validation", bootstrap_iterations=200)
    return tables


def fit_and_predict(labeled, test, candidates, weights, seeds):
    """Refit each member on all labeled rows per seed; return test scores and fitted models."""
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


def run_pipeline(data_dir=DATA_DIR, output_dir=None, validation_seeds=(0, 1, 2), seeds=FINAL_SEEDS):
    output_dir = PROJECT_ROOT if output_dir is None else Path(output_dir)
    train, test, _, sample_submission = load_data(data_dir)
    labeled = labeled_frame(train)
    candidates, weights = build_members(train.columns)

    tables = None
    if validation_seeds:
        tables = validate(labeled, candidates, weights, tuple(validation_seeds), output_dir / "artifacts" / "final")

    target, fitted = fit_and_predict(labeled, test, candidates, weights, seeds)
    submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": target})
    submission_path = write_submission(
        submission, output_dir / "submissions" / "submission_final.csv", sample_submission=sample_submission
    )
    model_path = output_dir / "models" / "final_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": FINAL_NAME, "weights": weights, "members": fitted, "seeds": list(seeds)}, model_path)
    return tables, submission, submission_path, model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()
    tables, submission, submission_path, model_path = run_pipeline(validation_seeds=tuple(args.validation_seeds))
    if tables is not None:
        print("=== Seed-bagged ROC-AUC by forward window ===")
        print(window_table(tables["window_metrics"]).round(4).to_string())
        print("\n=== Summary vs original production LightGBM ===")
        print(tables["summary"].round(4).to_string(index=False))
    print(f"\nFinal model: {FINAL_NAME}; saved {model_path}")
    print(f"Saved submission: {submission_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
