"""Version-controlled settings for the final submission model.

``python -m src.train`` and ``e1304439.ipynb`` read this file to reproduce the submission
without re-running any search. Selection evidence: report/findings.md (sections 14-20).

Final model: two-stage graph stacking (``src/graph_stacking.py``) with tuned LightGBM in both
stages, half-life-14 recency weights, one-hop labeled-neighbour context ("rich": structure
counts and stage-1 score aggregates; "wide": mean/max/std/in-mean/out-mean of 25 key features)
and two-hop context. Kaggle public ROC-AUC 0.96814. Each seed repeats both stages; seed
predictions are averaged.
"""

from src.model_zoo import SHALLOW_LGBM_PARAMS, TUNED_LGBM_PARAMS

FINAL_NAME = "nbr_stack_wide_2hop_hl14"
FINAL_SEEDS = tuple(range(10))

# Arguments for ``graph_stacking.make_neighbour_stack``.
FINAL_STACK = {"half_life_steps": 14, "rich": True, "wide": True, "two_hop": True}

# Simpler alternative (public 0.96668): the same pipeline without the 25-feature "wide" stats.
ALTERNATIVE_NAME = "nbr_stack_rich_2hop_hl14"
ALTERNATIVE_STACK = {"half_life_steps": 14, "rich": True, "two_hop": True}

# Steady second private-leaderboard pick: equal-weight rank average of these submission files
# (``python -m src.run_step8_variants blend blend_top4 ...``); report/findings.md section 23.
SECOND_PICK_BLEND = ["nbr_stack_wide_2hop_hl14", "nbr_stack_rich_2hop_hl14", "nbr_stack_wide_hl14", "nbr_stack_rich_hl14"]

# Non-graph reference used in validation tables: tuned LightGBM on the 165 supplied features
# (public 0.94417). family: lgbm | xgb | catboost; feature_set: all | local | aggregated.
REFERENCE_MEMBERS = [
    {
        "name": "tuned_lgbm_all",
        "family": "lgbm",
        "feature_set": "all",
        "drop": [],
        "params": dict(TUNED_LGBM_PARAMS),
        "weight": 1.0,
    },
]

# The five features that best separate labeled training rows from Kaggle test rows
# (adversarial validation, no labels; sections 5.3, 6, 8, 11). Tested; not used.
TEST_DRIFTED_FEATURES = ["feat_103", "feat_136", "feat_101", "feat_2", "feat_139"]

# Earlier regularised baseline (public 0.94267), kept for reference.
SHALLOW_ALL = {
    "name": "shallow_all",
    "family": "lgbm",
    "feature_set": "all",
    "drop": [],
    "params": dict(SHALLOW_LGBM_PARAMS),
    "weight": 1.0,
}
