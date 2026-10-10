"""Step 3: grouped feature ablation, temporal stability of importance, adversarial drops.

Three experiments, all using the regularised LightGBM (``shallow_all``) as the reference:

``clusters``     Group the 165 supplied features into correlated clusters (Spearman |rho|,
                 average linkage; no labels used). Drop one cluster at a time and score the
                 reduced model on every report window. Grouping matters because correlated
                 features mask one another under single-column permutation.

``stability``    Fit on each *tuning* window (validation <= step 28) and seed, compute
                 TreeSHAP mean |contribution| per feature on the validation rows, and rank
                 features by how consistently important they are across windows. Feature
                 subsets selected from that ranking are then scored on the *confirmation*
                 windows (validation 29-35), which played no part in the selection.

``adversarial``  For each report window, train a classifier to separate training-period
                 rows from validation-period rows (features only, no fraud labels), then
                 drop the k most separating features and re-score. At submission time the
                 same procedure uses training rows vs test rows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import (
    CONFIRM_WINDOWS,
    REPORT_WINDOWS,
    TUNING_WINDOWS,
    Candidate,
    evaluate_predictions,
    fit_model,
    labeled_frame,
    run_candidates,
    window_positions,
    window_table,
)
from src.features import get_feature_groups
from src.model_zoo import lgbm_factory

OUTPUT_DIR = Path("artifacts/features")
ABLATION_SEEDS = (0, 1, 2)


def correlation_clusters(frame, columns, distance_threshold=0.3):
    """Cluster columns by 1 - |Spearman rho| with average linkage."""
    rho = frame[columns].rank().corr().abs().fillna(0).to_numpy()
    np.fill_diagonal(rho, 1.0)
    distance = np.clip(1.0 - rho, 0.0, None)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=distance_threshold, criterion="distance")
    return pd.DataFrame({"feature": columns, "cluster": labels})


def run_cluster_ablation(train, labeled, groups):
    clusters = correlation_clusters(labeled, groups["all"])
    clusters.to_csv(OUTPUT_DIR / "feature_clusters.csv", index=False)
    sizes = clusters["cluster"].value_counts()
    print(f"{len(sizes)} clusters; sizes: {sizes.describe().round(1).to_dict()}")

    candidates = [Candidate("shallow_all", tuple(groups["all"]), lgbm_factory())]
    for cluster_id, members in clusters.groupby("cluster")["feature"]:
        kept = [column for column in groups["all"] if column not in set(members)]
        candidates.append(Candidate(f"drop_cluster_{cluster_id:03d}", tuple(kept), lgbm_factory()))

    predictions = run_candidates(labeled, candidates, REPORT_WINDOWS, ABLATION_SEEDS, verbose=False)
    tables = evaluate_predictions(predictions, "shallow_all", OUTPUT_DIR, "cluster_ablation", bootstrap_iterations=100)
    summary = tables["summary"].copy()
    summary["cluster"] = summary["candidate"].str.extract(r"drop_cluster_(\d+)").astype(float)
    membership = clusters.groupby("cluster")["feature"].agg(
        size="count", members=lambda s: " ".join(s)
    )
    summary = summary.merge(membership, left_on="cluster", right_index=True, how="left")
    summary.to_csv(OUTPUT_DIR / "cluster_ablation_with_members.csv", index=False)
    print(summary.drop(columns=["members"]).round(4).to_string(index=False))
    return summary


def shap_importance(model, X):
    contributions = model.predict_proba(X, pred_contrib=True)
    return np.abs(contributions[:, :-1]).mean(axis=0)


def run_stability(train, labeled, groups):
    columns = groups["all"]
    reference = Candidate("shallow_all", tuple(columns), lgbm_factory())
    times = labeled["time_step"].to_numpy()
    rows = []
    for window_name, window in TUNING_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        for seed in ABLATION_SEEDS:
            model = fit_model(reference, labeled.iloc[train_pos][columns], labeled["label"].to_numpy()[train_pos], seed)
            importance = shap_importance(model, labeled.iloc[valid_pos][columns])
            rows.append(pd.DataFrame({"window": window_name, "seed": seed, "feature": columns, "mean_abs_shap": importance}))
    long = pd.concat(rows, ignore_index=True)
    long["share"] = long["mean_abs_shap"] / long.groupby(["window", "seed"])["mean_abs_shap"].transform("sum")
    per_window = long.groupby(["feature", "window"])["share"].mean().unstack()
    stability = pd.DataFrame(
        {
            "mean_share": per_window.mean(axis=1),
            "min_share": per_window.min(axis=1),
            "cv_share": per_window.std(axis=1) / per_window.mean(axis=1).replace(0, np.nan),
        }
    )
    stability["min_share_rank"] = stability["min_share"].rank(ascending=False)
    stability = stability.join(per_window.add_prefix("share_")).sort_values("min_share", ascending=False)
    stability.to_csv(OUTPUT_DIR / "shap_stability_tuning_windows.csv")
    print(stability.head(25).round(4).to_string())

    # Subsets chosen on tuning windows only; scored on confirmation + report windows.
    candidates = [Candidate("shallow_all", tuple(columns), lgbm_factory())]
    ordered = stability.index.tolist()
    for k in (20, 40, 60, 90, 120):
        candidates.append(Candidate(f"stable_top{k}", tuple(ordered[:k]), lgbm_factory()))
    unstable = stability.loc[stability["cv_share"] > stability["cv_share"].quantile(0.8)].index
    candidates.append(Candidate("drop_unstable_cv_top20pct", tuple(c for c in columns if c not in set(unstable)), lgbm_factory()))

    windows = {**CONFIRM_WINDOWS, **REPORT_WINDOWS}
    predictions = run_candidates(labeled, candidates, windows, ABLATION_SEEDS, verbose=False)
    tables = evaluate_predictions(predictions, "shallow_all", OUTPUT_DIR, "stability_selection", bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    print(table.round(4).to_string())
    table.to_csv(OUTPUT_DIR / "stability_selection_table.csv")
    return stability, tables


def adversarial_ranking(early, late, columns, seed=0):
    """Rank columns by LightGBM gain when separating early rows from late rows."""
    X = pd.concat([early[columns], late[columns]], ignore_index=True)
    y = np.r_[np.zeros(len(early)), np.ones(len(late))]
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.8, random_state=seed, verbosity=-1, n_jobs=-1)
    model.fit(X, y)
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(gain, index=columns).sort_values(ascending=False)


def run_adversarial(train, test, labeled, groups):
    columns = groups["all"]
    times = labeled["time_step"].to_numpy()
    drop_counts = (0, 2, 5, 10, 20, 40)
    frames, rankings = [], []
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        # Separate the window's training-period rows from its validation-period rows,
        # using every row in those steps (labeled or not); fraud labels are never used.
        early = train.loc[train["time_step"] <= window[0]]
        late = train.loc[train["time_step"].between(window[1], window[2])]
        ranking = adversarial_ranking(early, late, columns)
        rankings.append(ranking.rename(window_name))
        candidates = [
            Candidate(f"adv_drop_{k:02d}", tuple(c for c in columns if c not in set(ranking.index[:k])), lgbm_factory())
            for k in drop_counts
        ]
        frames.append(run_candidates(labeled, candidates, {window_name: window}, ABLATION_SEEDS, verbose=False))
        print(f"  {window_name} top separating: {ranking.index[:5].tolist()}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    tables = evaluate_predictions(predictions, "adv_drop_00", OUTPUT_DIR, "adversarial_drop", bootstrap_iterations=100)
    table = window_table(tables["window_metrics"])
    print(table.round(4).to_string())
    table.to_csv(OUTPUT_DIR / "adversarial_drop_table.csv")

    test_ranking = adversarial_ranking(train, test, columns).rename("train_vs_test")
    pd.concat([*rankings, test_ranking], axis=1).to_csv(OUTPUT_DIR / "adversarial_rankings.csv")
    print("train vs test top separating:", test_ranking.index[:10].tolist())
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", choices=["clusters", "stability", "adversarial"])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, _, _ = load_data()
    labeled = labeled_frame(train)
    groups = get_feature_groups(train.columns)
    for experiment in args.experiments:
        print(f"\n===== {experiment} =====", flush=True)
        if experiment == "clusters":
            run_cluster_ablation(train, labeled, groups)
        elif experiment == "stability":
            run_stability(train, labeled, groups)
        else:
            run_adversarial(train, test, labeled, groups)


if __name__ == "__main__":
    main()
