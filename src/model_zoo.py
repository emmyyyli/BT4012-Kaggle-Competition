"""Model factories used by the evaluation harness, tuning and final submission.

Every factory takes a random seed and returns an unfitted estimator with a
scikit-learn style ``fit``/``predict_proba`` interface. LightGBM handles class
imbalance through ``class_weight="balanced"``; XGBoost and CatBoost receive
class-balanced sample weights from the harness instead (``needs_class_weights``).
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.modeling import make_model


# Original pipeline settings: deep trees and almost all columns per tree.
BASELINE_LGBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    colsample_bytree=0.9,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
)

# Hand-picked regularised setting from the first horizon probe: few leaves, large
# leaves, half the columns per tree. Intended to resist time-specific splits.
SHALLOW_LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=100,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=5.0,
)

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

DEFAULT_XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=5.0,
)

DEFAULT_CATBOOST_PARAMS = dict(
    iterations=600,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=5.0,
    rsm=0.5,
)


def lgbm_factory(params=None):
    params = dict(SHALLOW_LGBM_PARAMS if params is None else params)

    def make(seed):
        return LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

    make.needs_class_weights = False
    return make


def baseline_lgbm_factory():
    def make(seed):
        return make_model(random_state=seed)

    make.needs_class_weights = False
    return make


def xgb_factory(params=None):
    params = dict(DEFAULT_XGB_PARAMS if params is None else params)

    def make(seed):
        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    make.needs_class_weights = True
    return make


def catboost_factory(params=None):
    params = dict(DEFAULT_CATBOOST_PARAMS if params is None else params)

    def make(seed):
        return CatBoostClassifier(
            loss_function="Logloss",
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    make.needs_class_weights = True
    return make
