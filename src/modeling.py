import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.validation import holdout_split, temporal_splits


def make_model(random_state=42):
    return LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def evaluate_temporal_cv(train, feature_matrix, n_splits=4):
    known_mask = train["label"].notna().to_numpy()
    labeled_rows = train.loc[known_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = known_mask.nonzero()[0]
    if labeled_rows.empty:
        raise ValueError("No labeled rows with label 0 or 1 were found")

    results = []
    for fold, (train_positions, valid_positions) in enumerate(
        temporal_splits(labeled_rows, n_splits=n_splits), start=1
    ):
        train_rows = labeled_positions[train_positions]
        valid_rows = labeled_positions[valid_positions]
        model = make_model()
        model.fit(
            feature_matrix.iloc[train_rows],
            train["label"].iloc[train_rows].astype(int),
        )
        valid_labels = train["label"].iloc[valid_rows].astype(int)
        probabilities = model.predict_proba(feature_matrix.iloc[valid_rows])[:, 1]
        results.append(
            {
                "fold": fold,
                "train_max_time_step": train["time_step"].iloc[train_rows].max(),
                "valid_min_time_step": train["time_step"].iloc[valid_rows].min(),
                "valid_max_time_step": train["time_step"].iloc[valid_rows].max(),
                "roc_auc": roc_auc_score(valid_labels, probabilities),
                "pr_auc": average_precision_score(valid_labels, probabilities),
                "valid_rows": len(valid_rows),
            }
        )

    if not results:
        raise ValueError(
            "No temporal folds contain both classes in train and validation; "
            "inspect label coverage by time_step or reduce n_splits"
        )
    return pd.DataFrame(results)


def evaluate_holdout(train, feature_matrix, train_end=21, valid_end=35):
    known_mask = train["label"].notna().to_numpy()
    labeled_rows = train.loc[known_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = known_mask.nonzero()[0]
    if labeled_rows.empty:
        raise ValueError("No labeled rows with label 0 or 1 were found")

    train_positions, valid_positions = holdout_split(
        labeled_rows, train_end=train_end, valid_end=valid_end
    )
    train_rows = labeled_positions[train_positions]
    valid_rows = labeled_positions[valid_positions]

    model = make_model()
    model.fit(
        feature_matrix.iloc[train_rows],
        train["label"].iloc[train_rows].astype(int),
    )
    valid_labels = train["label"].iloc[valid_rows].astype(int)
    probabilities = model.predict_proba(feature_matrix.iloc[valid_rows])[:, 1]

    return {
        "train_min_time_step": int(train["time_step"].iloc[train_rows].min()),
        "train_max_time_step": int(train["time_step"].iloc[train_rows].max()),
        "valid_min_time_step": int(train["time_step"].iloc[valid_rows].min()),
        "valid_max_time_step": int(train["time_step"].iloc[valid_rows].max()),
        "roc_auc": float(roc_auc_score(valid_labels, probabilities)),
        "pr_auc": float(average_precision_score(valid_labels, probabilities)),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
    }


def fit_final_model(train, feature_matrix):
    known_mask = train["label"].notna()
    if train.loc[known_mask, "label"].nunique() < 2:
        raise ValueError("Final training requires labeled examples from both classes")
    model = make_model()
    model.fit(
        feature_matrix.loc[known_mask].reset_index(drop=True),
        train.loc[known_mask, "label"].astype(int).reset_index(drop=True),
    )
    return model