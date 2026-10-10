"""Fit registered candidates on every labeled row (steps 1-35) and write Kaggle submissions.

Each single-model submission averages ``--seeds`` independently seeded fits. Blends
rank-average their members' test predictions with fixed weights. Only the names passed
on the command line are produced, e.g.:

    python -m src.make_submissions shallow_all shallow_agg blend_shallow_all_agg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import Candidate, fit_model, labeled_frame
from src.features import get_feature_groups
from src.model_zoo import baseline_lgbm_factory, catboost_factory, lgbm_factory, xgb_factory
from src.submission import write_submission

SUBMISSION_DIR = Path("submissions")
TUNED_PARAMS_PATH = Path("artifacts/tuning/best_params.json")


def _tuned_params():
    if TUNED_PARAMS_PATH.exists():
        return json.loads(TUNED_PARAMS_PATH.read_text())
    return {}


def build_registry(columns):
    """Return (single-model candidates, blends) known to the submission writer."""
    groups = get_feature_groups(columns)
    registry = {
        "baseline_all": Candidate("baseline_all", tuple(groups["all"]), baseline_lgbm_factory()),
        "shallow_all": Candidate("shallow_all", tuple(groups["all"]), lgbm_factory()),
        "shallow_agg": Candidate("shallow_agg", tuple(groups["aggregated"]), lgbm_factory()),
    }
    factories = {"lgbm": lgbm_factory, "xgb": xgb_factory, "catboost": catboost_factory}
    for name, spec in _tuned_params().items():
        registry[name] = Candidate(
            name, tuple(spec["columns"]), factories[spec["family"]](spec["params"])
        )
    blends = {"blend_shallow_all_agg": {"shallow_all": 0.5, "shallow_agg": 0.5}}
    blend_path = Path("artifacts/tuning/blends.json")
    if blend_path.exists():
        blends.update(json.loads(blend_path.read_text()))
    return registry, blends


def predict_test(candidate, labeled, test, seeds):
    columns = list(candidate.columns)
    probabilities = [
        fit_model(
            candidate,
            labeled[columns],
            labeled["label"].to_numpy(),
            seed,
            labeled["time_step"].to_numpy(),
        ).predict_proba(test[columns])[:, 1]
        for seed in seeds
    ]
    return np.mean(probabilities, axis=0)


def rank_blend(member_predictions, weights):
    total = sum(weights.values())
    ranks = {
        name: pd.Series(member_predictions[name]).rank(pct=True).to_numpy() for name in weights
    }
    return sum(weights[name] * ranks[name] for name in weights) / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()

    train, test, _, sample_submission = load_data()
    labeled = labeled_frame(train)
    registry, blends = build_registry(train.columns)
    unknown = [name for name in args.names if name not in registry and name not in blends]
    if unknown:
        raise SystemExit(f"Unknown candidates: {unknown}. Known: {sorted([*registry, *blends])}")

    needed = set()
    for name in args.names:
        needed.update(blends[name] if name in blends else [name])
    predictions = {}
    for name in sorted(needed):
        print(f"Fitting {name} on {len(labeled):,} labeled rows x {len(args.seeds)} seeds", flush=True)
        predictions[name] = predict_test(registry[name], labeled, test, args.seeds)

    for name in args.names:
        target = rank_blend(predictions, blends[name]) if name in blends else predictions[name]
        submission = pd.DataFrame({"index": test["index"].to_numpy(), "target": target})
        path = write_submission(
            submission, SUBMISSION_DIR / f"submission_{name}.csv", sample_submission=sample_submission
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
