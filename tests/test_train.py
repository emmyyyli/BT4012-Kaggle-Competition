import numpy as np
import pandas as pd

from src.train import build_members, fit_and_predict


def _labeled(n=120, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "time_step": np.repeat(np.arange(1, 7), n // 6),
            "label": label,
            "feat_2": label + rng.normal(0, 0.3, n),
            "feat_5": rng.normal(0, 1, n),
            "feat_100": label * 0.5 + rng.normal(0, 1, n),
        }
    )


MEMBERS = [
    {"name": "a", "family": "lgbm", "feature_set": "all", "drop": ["feat_2"],
     "params": {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 5}, "weight": 1.0},
    {"name": "b", "family": "xgb", "feature_set": "local", "drop": [],
     "params": {"n_estimators": 20, "max_depth": 2}, "weight": 3.0},
]


def test_build_members_resolves_feature_sets_and_drops():
    candidates, weights = build_members(["feat_2", "feat_5", "feat_100", "label"], MEMBERS)
    assert [c.name for c in candidates] == ["a", "b"]
    assert candidates[0].columns == ("feat_5", "feat_100")
    assert candidates[1].columns == ("feat_2", "feat_5")
    assert weights == {"a": 1.0, "b": 3.0}


def test_fit_and_predict_rank_averages_members_over_seeds():
    labeled = _labeled()
    test = _labeled(seed=1).drop(columns=["label"])
    candidates, weights = build_members(labeled.columns, MEMBERS)

    target, fitted = fit_and_predict(labeled, test, candidates, weights, seeds=(0, 1))

    assert target.shape == (len(test),)
    assert np.all((target > 0) & (target <= 1))
    assert len(fitted["a"]["models"]) == 2
    assert fitted["b"]["columns"] == ["feat_2", "feat_5"]
