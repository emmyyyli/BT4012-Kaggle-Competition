import numpy as np
import pandas as pd
import pytest

from src.run_lgbm_robustness_experiments import (
    FEATURE_ABLATIONS,
    _add_baseline_deltas_and_summarize,
    _build_splits,
    exponential_recency_weights,
)


def test_exponential_recency_weights_are_mean_one_and_decay_by_half_life():
    weights = exponential_recency_weights([1, 8, 15], half_life=7)

    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(2.0)
    assert weights[2] / weights[1] == pytest.approx(2.0)


def test_exponential_recency_weights_reject_invalid_inputs():
    with pytest.raises(ValueError, match="half_life"):
        exponential_recency_weights([1, 2], half_life=0)
    with pytest.raises(ValueError, match="non-empty and finite"):
        exponential_recency_weights([1, np.nan], half_life=7)


def test_robustness_splits_are_forward_time_and_include_long_holdout():
    train = pd.DataFrame(
        {
            "time_step": np.repeat(np.arange(1, 36), 2),
            "label": [0, 1] * 35,
        }
    )

    splits = _build_splits(train)

    assert len(splits) == 5
    for split_name, _, train_rows, valid_rows in splits:
        train_end = train.iloc[train_rows]["time_step"].max()
        valid_start = train.iloc[valid_rows]["time_step"].min()
        assert train_end < valid_start
        if split_name == "long_holdout":
            assert train_end == 21
            assert train.iloc[valid_rows]["time_step"].max() == 35


def test_feature_ablation_variants_are_independent():
    assert FEATURE_ABLATIONS == {
        "drop_feat_101": {"feat_101"},
        "drop_feat_103": {"feat_103"},
        "drop_feat_101_103": {"feat_101", "feat_103"},
    }


def test_summary_uses_seed_paired_baseline_deltas_per_fold():
    results = pd.DataFrame(
        [
            {"seed": 1, "variant": "baseline", "split": "cv", "fold": 1, "roc_auc": 0.8, "pr_auc": 0.7},
            {"seed": 2, "variant": "baseline", "split": "cv", "fold": 1, "roc_auc": 0.7, "pr_auc": 0.6},
            {"seed": 1, "variant": "candidate", "split": "cv", "fold": 1, "roc_auc": 0.81, "pr_auc": 0.72},
            {"seed": 2, "variant": "candidate", "split": "cv", "fold": 1, "roc_auc": 0.69, "pr_auc": 0.59},
        ]
    )

    detailed, summary = _add_baseline_deltas_and_summarize(results)
    candidate = summary.loc[summary["variant"] == "candidate"].iloc[0]

    assert detailed.loc[detailed.variant == "candidate", "roc_auc_delta_vs_baseline"].tolist() == pytest.approx([0.01, -0.01])
    assert candidate["roc_auc_delta_mean"] == pytest.approx(0.0)
    assert candidate["roc_auc_delta_std"] == pytest.approx(np.sqrt(0.0002))
    assert candidate["seed_count"] == 2
