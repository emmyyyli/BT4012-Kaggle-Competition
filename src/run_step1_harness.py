"""Step 1: re-score the current baseline and the first leads with the evaluation harness.

Candidates (all trained on labeled rows only, time_step excluded):
* baseline_all       - the current production LightGBM on all 165 supplied features
* shallow_all        - regularised LightGBM on all 165 features
* baseline_agg       - production LightGBM on the 72 aggregated features
* shallow_agg        - regularised LightGBM on the 72 aggregated features
* shallow_local      - regularised LightGBM on the 93 local features
* blend_shallow_all_agg - fixed 50/50 rank average of shallow_all and shallow_agg

The 50/50 weight is fixed before looking at results: it is the neutral choice for
combining a model that is strong at short horizons with one strong at long horizons.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import (
    DEFAULT_SEEDS,
    REPORT_WINDOWS,
    Candidate,
    add_rank_blend,
    evaluate_predictions,
    labeled_frame,
    run_candidates,
    window_table,
)
from src.features import get_feature_groups
from src.model_zoo import baseline_lgbm_factory, lgbm_factory

OUTPUT_DIR = Path("artifacts/harness")


def step1_candidates(groups):
    return [
        Candidate("baseline_all", tuple(groups["all"]), baseline_lgbm_factory()),
        Candidate("shallow_all", tuple(groups["all"]), lgbm_factory()),
        Candidate("baseline_agg", tuple(groups["aggregated"]), baseline_lgbm_factory()),
        Candidate("shallow_agg", tuple(groups["aggregated"]), lgbm_factory()),
        Candidate("shallow_local", tuple(groups["local"]), lgbm_factory()),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    train, _, _, _ = load_data()
    labeled = labeled_frame(train)
    groups = get_feature_groups(train.columns)

    predictions = run_candidates(labeled, step1_candidates(groups), REPORT_WINDOWS, tuple(args.seeds))
    predictions = add_rank_blend(
        predictions, {"shallow_all": 0.5, "shallow_agg": 0.5}, "blend_shallow_all_agg"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "step1_predictions.csv.gz", index=False)
    tables = evaluate_predictions(predictions, "baseline_all", OUTPUT_DIR, "step1")

    print("\n=== Seed-bagged ROC-AUC by window ===")
    print(window_table(tables["window_metrics"]).round(4).to_string())
    print("\n=== Summary vs baseline_all ===")
    print(tables["summary"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
