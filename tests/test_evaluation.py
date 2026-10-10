import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    Candidate,
    add_rank_blend,
    bag_seeds,
    evaluate_predictions,
    labeled_frame,
    run_candidates,
    summarize,
    window_positions,
)
from src.features import get_feature_groups
from src.model_zoo import catboost_factory, lgbm_factory, xgb_factory


def _synthetic_train(steps=12, rows_per_step=40, seed=0):
    rng = np.random.default_rng(seed)
    n = steps * rows_per_step
    label = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "txId": [str(i) for i in range(n)],
            "time_step": np.repeat(np.arange(1, steps + 1), rows_per_step),
            "label": pd.array(label, dtype="Int8"),
            "feat_1": label + rng.normal(0, 0.5, n),
            "feat_94": rng.normal(0, 1, n),
        }
    )
    frame.loc[frame.index % 7 == 0, "label"] = pd.NA
    return frame


def test_feature_groups_split_local_and_aggregated_columns():
    groups = get_feature_groups(["feat_94", "feat_2", "feat_93", "feat_165", "label"])
    assert groups["local"] == ["feat_2", "feat_93"]
    assert groups["aggregated"] == ["feat_94", "feat_165"]
    assert groups["all"] == ["feat_2", "feat_93", "feat_94", "feat_165"]


def test_window_positions_are_strictly_forward_in_time():
    times = np.array([1, 2, 3, 4, 5, 6])
    train_pos, valid_pos = window_positions(times, (3, 5, 6))
    assert times[train_pos].max() == 3
    assert set(times[valid_pos]) == {5, 6}
    with pytest.raises(ValueError, match="after training"):
        window_positions(times, (4, 4, 6))


def test_run_candidates_excludes_unlabeled_rows_and_scores_each_seed():
    labeled = labeled_frame(_synthetic_train())
    assert labeled["label"].isin([0, 1]).all()
    candidates = [
        Candidate("lgbm", ("feat_1", "feat_94"), lgbm_factory({"n_estimators": 20, "num_leaves": 4, "min_child_samples": 5})),
        Candidate("xgb", ("feat_1",), xgb_factory({"n_estimators": 20, "max_depth": 2})),
        Candidate("cat", ("feat_1",), catboost_factory({"iterations": 20, "depth": 2})),
    ]
    windows = {"w": (6, 7, 12)}
    predictions = run_candidates(labeled, candidates, windows=windows, seeds=(0, 1), verbose=False)

    assert set(predictions["candidate"]) == {"lgbm", "xgb", "cat"}
    assert predictions["time_step"].min() == 7
    assert len(predictions) == 3 * 2 * (labeled["time_step"] >= 7).sum()
    assert predictions["probability"].between(0, 1).all()


def test_rank_blend_and_summary_apply_adoption_rule():
    base = pd.DataFrame(
        {
            "candidate": "base",
            "window": np.repeat(["a", "b"], 4),
            "seed": 0,
            "position": np.tile(np.arange(4), 2),
            "time_step": 1,
            "label": np.tile([0, 0, 1, 1], 2),
            "probability": [0.1, 0.6, 0.4, 0.9, 0.1, 0.2, 0.8, 0.9],
        }
    )
    better = base.assign(candidate="better", probability=np.tile([0.1, 0.2, 0.8, 0.9], 2))
    predictions = add_rank_blend(pd.concat([base, better]), {"base": 1.0, "better": 1.0}, "blend")

    blended = predictions.loc[predictions["candidate"] == "blend"]
    assert len(blended) == 8
    assert blended["probability"].between(0, 1).all()

    tables = evaluate_predictions(predictions, baseline="base", bootstrap_iterations=20)
    summary = tables["summary"].set_index("candidate")
    assert summary.loc["better", "mean_roc_auc"] == pytest.approx(1.0)
    assert summary.loc["better", "adopt"]
    assert not summary.loc["base", "adopt"]
    assert len(bag_seeds(predictions)) == 3 * 8


def test_summary_rejects_candidate_with_large_single_window_loss():
    windows = pd.DataFrame(
        {
            "candidate": ["base", "base", "risky", "risky"],
            "window": ["a", "b", "a", "b"],
            "roc_auc": [0.90, 0.90, 0.99, 0.88],
        }
    )
    summary = summarize(windows, baseline="base").set_index("candidate")
    assert summary.loc["risky", "mean_delta_vs_baseline"] > 0
    assert not summary.loc["risky", "adopt"]
