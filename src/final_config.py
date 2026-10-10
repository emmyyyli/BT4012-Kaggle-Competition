"""Version-controlled settings for the final submission model.

Tuning results live in ``artifacts/tuning`` (not tracked), so the selected configuration is
copied here so that ``python -m src.train`` and ``e1304439.ipynb`` reproduce the submission
without re-running the search. Every member is fitted on all labeled rows (steps 1-35) with
each seed in ``FINAL_SEEDS``; the seed-averaged probabilities of each member are then
rank-averaged with the given weights (a single member is used as-is).

Selection evidence: report/findings.md.
"""

from src.model_zoo import SHALLOW_LGBM_PARAMS

FINAL_NAME = "tuned_lgbm_all_no_test_drift5"
FINAL_SEEDS = tuple(range(10))

# Tuned LightGBM settings (Optuna, short-horizon tuning windows; report/findings.md section 8).
TUNED_LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.0252683175248254,
    num_leaves=9,
    min_child_samples=49,
    colsample_bytree=0.644456547178642,
    subsample=0.9826426262765469,
    subsample_freq=1,
    reg_lambda=2.035593509388961,
)

# The five features that best separate labeled training rows from Kaggle test rows
# (adversarial validation, no labels; report/findings.md sections 5.3, 6 and 8).
TEST_DRIFTED_FEATURES = ["feat_103", "feat_136", "feat_101", "feat_2", "feat_139"]

# family: lgbm | xgb | catboost; feature_set: all | local | aggregated; drop: columns removed.
FINAL_MEMBERS = [
    {
        "name": "tuned_lgbm_all_no_test_drift5",
        "family": "lgbm",
        "feature_set": "all",
        "drop": list(TEST_DRIFTED_FEATURES),
        "params": dict(TUNED_LGBM_PARAMS),
        "weight": 1.0,
    },
]

# Safe second pick for the private leaderboard (public 0.94267): shallow_all, i.e.
# feature_set "all", no drops, params SHALLOW_LGBM_PARAMS.
SAFE_ALTERNATIVE = {
    "name": "shallow_all",
    "family": "lgbm",
    "feature_set": "all",
    "drop": [],
    "params": dict(SHALLOW_LGBM_PARAMS),
    "weight": 1.0,
}
