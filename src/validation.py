import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def temporal_splits(labeled_rows, n_splits=4):
    """Yield expanding-window row positions ordered by distinct time_step."""
    time_steps = np.sort(labeled_rows["time_step"].dropna().unique())
    if len(time_steps) < 3:
        raise ValueError("At least three labeled time steps are needed for validation")

    split_count = min(n_splits, len(time_steps) - 1)
    splitter = TimeSeriesSplit(n_splits=split_count)
    for train_step_positions, valid_step_positions in splitter.split(time_steps):
        train_steps = time_steps[train_step_positions]
        valid_steps = time_steps[valid_step_positions]
        train_positions = np.flatnonzero(labeled_rows["time_step"].isin(train_steps))
        valid_positions = np.flatnonzero(labeled_rows["time_step"].isin(valid_steps))

        train_labels = labeled_rows.iloc[train_positions]["label"].astype(int)
        valid_labels = labeled_rows.iloc[valid_positions]["label"].astype(int)
        if train_labels.nunique() < 2 or valid_labels.nunique() < 2:
            continue
        yield train_positions, valid_positions


def holdout_split(labeled_rows, train_end=21, valid_end=35):
    """Yield train and validation row positions for a specific horizon holdout."""
    train_positions = np.flatnonzero(labeled_rows["time_step"] <= train_end)
    valid_positions = np.flatnonzero(
        (labeled_rows["time_step"] > train_end) & (labeled_rows["time_step"] <= valid_end)
    )

    if len(train_positions) == 0 or len(valid_positions) == 0:
        raise ValueError(
            f"Holdout split produced empty subsets: train={len(train_positions)}, valid={len(valid_positions)}"
        )

    train_labels = labeled_rows.iloc[train_positions]["label"].astype(int)
    valid_labels = labeled_rows.iloc[valid_positions]["label"].astype(int)
    if train_labels.nunique() < 2 or valid_labels.nunique() < 2:
        raise ValueError("Holdout train and validation sets must both contain at least two classes")

    return train_positions, valid_positions