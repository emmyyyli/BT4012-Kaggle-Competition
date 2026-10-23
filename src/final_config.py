"""Version-controlled settings for the final submission model.

``python -m src.train`` and ``e1304439.ipynb`` read this file to reproduce the submission
without re-running any search. Selection evidence: report/findings.md (sections 14-20, 35, 37-44).

Final model: two-stage graph stacking (``src/graph_stacking.py``) with tuned LightGBM in both
stages, half-life-14 recency weights, one-hop labeled-neighbour context ("rich": structure
counts and stage-1 score aggregates; "wide": mean/max/std/in-mean/out-mean of 25 key features),
two-hop context (means over all 25 wide features) and label-free payment-chain shape ("chain": longest path in/out, peel-chain
runs), with every fit's sample weights also multiplied by covariate-shift importance weights
("shift_weights": how test-like each training row looks, after removing the 30 most
time-encoding features). In stage 2 only, each tree samples columns in proportion to the 165
supplied features (``stage2_match_features``), about 106 per tree, as in stage 1. Kaggle public
ROC-AUC 0.97190. Each seed repeats both stages; seed predictions are averaged. Without the
stage-2 column matching it scored 0.97097; with two-hop means over 5 features only, 0.9702; without
``shift_weights`` as well, 0.96858 (the second private pick, ``SECOND_PICK_NAME``);
report/findings.md sections 35-41.
"""

from src.model_zoo import SHALLOW_LGBM_PARAMS, TUNED_LGBM_PARAMS

FINAL_NAME = "nbr_stack_wide_2hopw_chain_shift_s2reg_hl14"
FINAL_SEEDS = tuple(range(10))

# Arguments for ``graph_stacking.make_neighbour_stack``.
FINAL_STACK = {"half_life_steps": 14, "rich": True, "wide": True, "two_hop": True, "chain": True, "shift_weights": True,
               "two_hop_wide": True, "stage2_match_features": True}

# Simpler alternative (public 0.96668): the same pipeline without the 25-feature "wide" stats.
ALTERNATIVE_NAME = "nbr_stack_rich_2hop_hl14"
ALTERNATIVE_STACK = {"half_life_steps": 14, "rich": True, "two_hop": True}

# Steady second private-leaderboard pick (public 0.96858): the final model without the shift
# weights (``python -m src.run_step8_variants submit nbr_stack_wide_2hop_chain_hl14``);
# report/findings.md section 39.
SECOND_PICK_NAME = "nbr_stack_wide_2hop_chain_hl14"
SECOND_PICK_STACK = {"half_life_steps": 14, "rich": True, "wide": True, "two_hop": True, "chain": True}

# Earlier steady pick (public 0.96686, superseded in section 39): equal-weight rank average of
# these submission files (``python -m src.run_step8_variants blend blend_top4 ...``).
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
