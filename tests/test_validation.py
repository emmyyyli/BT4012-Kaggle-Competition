import pandas as pd

from src.validation import holdout_split, temporal_splits


def test_temporal_splits_train_only_on_earlier_time_steps():
    labeled = pd.DataFrame(
        {
            "time_step": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "label": [0, 1] * 6,
        }
    )

    splits = list(temporal_splits(labeled, n_splits=3))

    assert len(splits) == 3
    for train_positions, valid_positions in splits:
        train_times = labeled.iloc[train_positions]["time_step"]
        valid_times = labeled.iloc[valid_positions]["time_step"]
        assert train_times.max() < valid_times.min()
        assert labeled.iloc[train_positions]["label"].nunique() == 2
        assert labeled.iloc[valid_positions]["label"].nunique() == 2


def test_holdout_split_respects_time_step_boundaries():
    labeled = pd.DataFrame(
        {
            "time_step": [1, 2, 20, 21, 22, 25, 35, 36],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train_positions, valid_positions = holdout_split(labeled, train_end=21, valid_end=35)
    train_times = labeled.iloc[train_positions]["time_step"]
    valid_times = labeled.iloc[valid_positions]["time_step"]

    assert train_times.max() <= 21
    assert valid_times.min() >= 22
    assert valid_times.max() <= 35
    assert 36 not in valid_times.values
    assert labeled.iloc[train_positions]["label"].nunique() == 2
    assert labeled.iloc[valid_positions]["label"].nunique() == 2