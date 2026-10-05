"""Matched LightGBM recency-weight and feature-drop experiments."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_feature_columns, load_data
from src.features import build_features
from src.modeling import make_model
from src.validation import holdout_split, temporal_splits


RECENCY_HALF_LIVES = (3, 7, 14)
FEATURE_ABLATIONS = {
    "drop_feat_101": {"feat_101"},
    "drop_feat_103": {"feat_103"},
    "drop_feat_101_103": {"feat_101", "feat_103"},
}


def exponential_recency_weights(time_steps, half_life):
    """Return mean-one weights with exponentially decaying influence by age."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    times = np.asarray(time_steps, dtype=np.float64)
    if times.size == 0 or not np.isfinite(times).all():
        raise ValueError("time_steps must be non-empty and finite")
    cutoff = times.max()
    weights = np.exp2((times - cutoff) / half_life)
    return (weights / weights.mean()).astype(np.float64)


def _build_splits(train, n_splits=4):
    known_mask = train["label"].notna().to_numpy()
    labeled = train.loc[known_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = np.flatnonzero(known_mask)
    splits = [
        (
            "temporal_cv",
            fold,
            labeled_positions[train_positions],
            labeled_positions[valid_positions],
        )
        for fold, (train_positions, valid_positions) in enumerate(
            temporal_splits(labeled, n_splits=n_splits), start=1
        )
    ]
    train_positions, valid_positions = holdout_split(
        labeled, train_end=21, valid_end=35
    )
    splits.append(
        (
            "long_holdout",
            1,
            labeled_positions[train_positions],
            labeled_positions[valid_positions],
        )
    )
    return splits


def _add_baseline_deltas_and_summarize(results):
    baseline_metrics = results.loc[
        results["variant"] == "baseline",
        ["seed", "split", "fold", "roc_auc", "pr_auc"],
    ].rename(
        columns={
            "roc_auc": "baseline_roc_auc",
            "pr_auc": "baseline_pr_auc",
        }
    )
    results = results.merge(
        baseline_metrics, on=["seed", "split", "fold"], validate="many_to_one"
    )
    results["roc_auc_delta_vs_baseline"] = (
        results["roc_auc"] - results["baseline_roc_auc"]
    )
    results["pr_auc_delta_vs_baseline"] = (
        results["pr_auc"] - results["baseline_pr_auc"]
    )
    summary = (
        results.groupby(["variant", "split", "fold"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_delta_mean=("roc_auc_delta_vs_baseline", "mean"),
            roc_auc_delta_std=("roc_auc_delta_vs_baseline", "std"),
            pr_auc_delta_mean=("pr_auc_delta_vs_baseline", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(
            ["split", "fold", "roc_auc_mean"],
            ascending=[True, True, False],
        )
    )
    return results, summary


def run_robustness_experiments(
    train=None,
    test=None,
    edges=None,
    output_dir=Path("artifacts"),
    seeds=(42, 7),
):
    if train is None or test is None or edges is None:
        train, test, edges, _ = load_data()

    feature_frame = build_features(
        train, test, edges, include_in_degree=False
    )
    base_columns = get_feature_columns(train.columns)
    missing_features = {"feat_101", "feat_103"}.difference(base_columns)
    if missing_features:
        raise ValueError(
            f"Expected adversarially shifted features are missing: {sorted(missing_features)}"
        )

    variants = {"baseline": (base_columns, None)}
    for half_life in RECENCY_HALF_LIVES:
        variants[f"recency_half_life_{half_life}"] = (
            base_columns,
            half_life,
        )
    for variant, dropped in FEATURE_ABLATIONS.items():
        variants[variant] = (
            [column for column in base_columns if column not in dropped],
            None,
        )

    splits = _build_splits(train)
    results = []
    predictions = []
    for seed in seeds:
        for split_name, fold, train_rows, valid_rows in splits:
            train_times = train["time_step"].iloc[train_rows].to_numpy()
            valid_times = train["time_step"].iloc[valid_rows].to_numpy()
            y_train = train["label"].iloc[train_rows].astype(int)
            y_valid = train["label"].iloc[valid_rows].astype(int)

            for variant, (columns, half_life) in variants.items():
                model = make_model(random_state=seed)
                fit_kwargs = {}
                if half_life is not None:
                    fit_kwargs["sample_weight"] = exponential_recency_weights(
                        train_times, half_life
                    )
                model.fit(
                    feature_frame.iloc[train_rows][columns],
                    y_train,
                    **fit_kwargs,
                )
                probabilities = model.predict_proba(
                    feature_frame.iloc[valid_rows][columns]
                )[:, 1]
                result = {
                    "seed": seed,
                    "variant": variant,
                    "split": split_name,
                    "fold": fold,
                    "train_min_time_step": int(train_times.min()),
                    "train_max_time_step": int(train_times.max()),
                    "valid_min_time_step": int(valid_times.min()),
                    "valid_max_time_step": int(valid_times.max()),
                    "train_rows": len(train_rows),
                    "valid_rows": len(valid_rows),
                    "feature_count": len(columns),
                    "roc_auc": float(roc_auc_score(y_valid, probabilities)),
                    "pr_auc": float(average_precision_score(y_valid, probabilities)),
                }
                if half_life is not None:
                    result["recency_half_life"] = half_life
                    result["mean_sample_weight"] = float(
                        fit_kwargs["sample_weight"].mean()
                    )
                    result["min_sample_weight"] = float(
                        fit_kwargs["sample_weight"].min()
                    )
                    result["max_sample_weight"] = float(
                        fit_kwargs["sample_weight"].max()
                    )
                results.append(result)
                predictions.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "variant": variant,
                            "split": split_name,
                            "fold": fold,
                            "time_step": valid_times,
                            "label": y_valid.to_numpy(),
                            "probability": probabilities,
                        }
                    )
                )

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    result_frame, summary = _add_baseline_deltas_and_summarize(result_frame)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "lgbm_robustness_results.csv"
    summary_path = output_dir / "lgbm_robustness_summary.csv"
    predictions_path = output_dir / "lgbm_robustness_predictions.csv"
    result_frame.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)

    print("=== Per-window ROC-AUC and paired change vs. baseline ===")
    print(summary.to_string(index=False))
    print(f"\nSaved fold-level results: {result_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved validation predictions: {predictions_path}")
    return result_frame, summary, prediction_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    args = parser.parse_args()
    if not args.seeds:
        parser.error("provide at least one seed")
    run_robustness_experiments(seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
