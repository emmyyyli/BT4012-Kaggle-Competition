import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.features import build_experiment_features, get_experiment_feature_sets
from src.modeling import make_model
from src.validation import holdout_split, temporal_splits


def _score_split(train, features, feature_columns, train_rows, valid_rows, split, fold):
    model = make_model()
    y_train = train["label"].iloc[train_rows].astype(int)
    y_valid = train["label"].iloc[valid_rows].astype(int)
    model.fit(features.iloc[train_rows][feature_columns], y_train)
    probabilities = model.predict_proba(features.iloc[valid_rows][feature_columns])[:, 1]

    gains = model.booster_.feature_importance(importance_type="gain")
    gain_total = gains.sum()
    if gain_total:
        gains = gains / gain_total
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "normalized_gain": gains,
            "split": split,
            "fold": fold,
        }
    )
    result = {
        "split": split,
        "fold": fold,
        "train_min_time_step": int(train["time_step"].iloc[train_rows].min()),
        "train_max_time_step": int(train["time_step"].iloc[train_rows].max()),
        "valid_min_time_step": int(train["time_step"].iloc[valid_rows].min()),
        "valid_max_time_step": int(train["time_step"].iloc[valid_rows].max()),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "feature_count": len(feature_columns),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
    }
    return result, importance


def run_temporal_ablation(train, feature_frame, n_splits=4):
    """Evaluate fixed feature variants on matching temporal folds and one long holdout."""
    variants = get_experiment_feature_sets(feature_frame, include_out_degree=True)
    known_mask = train["label"].notna().to_numpy()
    labeled_rows = train.loc[known_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = np.flatnonzero(known_mask)
    folds = list(temporal_splits(labeled_rows, n_splits=n_splits))
    holdout_train, holdout_valid = holdout_split(
        labeled_rows, train_end=21, valid_end=35
    )
    splits = [
        (
            "temporal_cv",
            fold,
            labeled_positions[train_pos],
            labeled_positions[valid_pos],
        )
        for fold, (train_pos, valid_pos) in enumerate(folds, start=1)
    ]
    splits.append(
        (
            "long_holdout",
            1,
            labeled_positions[holdout_train],
            labeled_positions[holdout_valid],
        )
    )

    results = []
    importances = []
    for variant, columns in variants.items():
        for split, fold, train_rows, valid_rows in splits:
            result, importance = _score_split(
                train, feature_frame, columns, train_rows, valid_rows, split, fold
            )
            result["variant"] = variant
            importance["variant"] = variant
            results.append(result)
            importances.append(importance)

    return pd.DataFrame(results), pd.concat(importances, ignore_index=True)


def run_adversarial_validation(train, test, feature_frame, random_state=42):
    """Measure feature-only separability of historical labeled rows and Kaggle test rows."""
    labeled_mask = train["label"].notna().to_numpy()
    train_indices = np.flatnonzero(labeled_mask)
    test_indices = np.arange(len(train), len(train) + len(test))
    columns = [
        column
        for column in feature_frame.columns
        if column.startswith("feat_")
        or column
        in {
            "in_degree",
            "out_degree",
            "log_in_degree",
            "log_out_degree",
            "feature_mean",
            "feature_std",
            "feature_min",
            "feature_max",
            "feature_abs_mean",
            "feature_zero_fraction",
        }
    ]
    row_indices = np.concatenate([train_indices, test_indices])
    domain_labels = np.concatenate(
        [np.zeros(len(train_indices), dtype=np.int8), np.ones(len(test_indices), dtype=np.int8)]
    )
    fit_rows, valid_rows = train_test_split(
        np.arange(len(row_indices)),
        test_size=0.25,
        random_state=random_state,
        stratify=domain_labels,
    )

    model = make_model()
    model.fit(feature_frame.iloc[row_indices[fit_rows]][columns], domain_labels[fit_rows])
    probabilities = model.predict_proba(
        feature_frame.iloc[row_indices[valid_rows]][columns]
    )[:, 1]
    auc = float(roc_auc_score(domain_labels[valid_rows], probabilities))

    gains = model.booster_.feature_importance(importance_type="gain")
    importance = pd.DataFrame({"feature": columns, "gain": gains})
    importance = importance.sort_values("gain", ascending=False, ignore_index=True)
    importance["normalized_gain"] = importance["gain"] / importance["gain"].sum()
    return {
        "domain": "labeled_train_vs_test",
        "roc_auc": auc,
        "train_rows": int((domain_labels[fit_rows] == 0).sum()),
        "valid_train_rows": int((domain_labels[valid_rows] == 0).sum()),
        "valid_test_rows": int((domain_labels[valid_rows] == 1).sum()),
        "time_step_included": False,
    }, importance


def run_experiments(output_dir=Path("artifacts")):
    train, test, edges, _ = load_data()
    feature_frame = build_experiment_features(train, test, edges)
    results, importances = run_temporal_ablation(train, feature_frame)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "feature_experiment_results.csv"
    importance_path = output_dir / "feature_gain_importance.csv"
    results.to_csv(results_path, index=False)

    gain_summary = (
        importances.groupby(["variant", "feature"], as_index=False)["normalized_gain"]
        .mean()
        .sort_values(["variant", "normalized_gain"], ascending=[True, False])
    )
    gain_summary.to_csv(importance_path, index=False)

    adversarial_result, adversarial_importance = run_adversarial_validation(
        train, test, feature_frame
    )
    adversarial_path = output_dir / "adversarial_validation.csv"
    adversarial_importance_path = output_dir / "adversarial_feature_importance.csv"
    pd.DataFrame([adversarial_result]).to_csv(adversarial_path, index=False)
    adversarial_importance.to_csv(adversarial_importance_path, index=False)

    print("=== Temporal ablation ROC-AUC / PR-AUC ===")
    print(
        results.groupby(["variant", "split"], as_index=False)[["roc_auc", "pr_auc"]]
        .mean()
        .to_string(index=False)
    )
    print(
        f"\nAdversarial validation ROC-AUC (time_step excluded): "
        f"{adversarial_result['roc_auc']:.5f}"
    )
    print(f"Saved results: {results_path}")
    print(f"Saved gain importance: {importance_path}")
    print(f"Saved adversarial validation: {adversarial_path}")
    print(f"Saved adversarial feature importance: {adversarial_importance_path}")


if __name__ == "__main__":
    run_experiments()
