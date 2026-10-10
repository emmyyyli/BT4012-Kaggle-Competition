"""Step 5: stress-test finalist models under simulated shocks.

Every candidate is fitted once per report window and seed, then scored on perturbed
copies of the validation rows. Blends are re-formed inside each scenario from their
members' perturbed predictions. Scenarios:

``clean``                 unmodified validation rows (reference)
``drift_0.5`` / ``drift_1.0``
                          add alpha x (test median - labeled-train median) to every feature,
                          i.e. push validation rows further along the observed train->test
                          drift direction
``noise_local`` / ``noise_aggregated``
                          Gaussian noise with sd = 0.25 x the training-row sd, applied to one
                          feature group only
``withhold_high_illicit`` refit without training steps whose illicit share exceeds 25%,
                          testing dependence on a few fraud-heavy periods
``rare_positives``        clean predictions, but validation positives subsampled to a 2%
                          illicit share (200 draws), mimicking the post-shutdown regime;
                          the mean and 5th percentile AUC are reported

The selection criterion is the worst scenario, not the clean score.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import REPORT_WINDOWS, add_rank_blend, fit_model, labeled_frame, window_positions
from src.features import get_feature_groups
from src.make_submissions import build_registry

OUTPUT_DIR = Path("artifacts/stress")
STRESS_SEEDS = (0, 1, 2)
HIGH_ILLICIT_SHARE = 0.25
RARE_SHARE = 0.02


def scenario_frames(X_valid, X_train, drift, groups, rng):
    yield "clean", X_valid
    for alpha in (0.5, 1.0):
        yield f"drift_{alpha}", X_valid + alpha * drift[X_valid.columns]
    for group in ("local", "aggregated"):
        columns = [c for c in groups[group] if c in X_valid.columns]
        noisy = X_valid.copy()
        if columns:
            scale = X_train[columns].std().to_numpy() * 0.25
            noisy[columns] = noisy[columns].to_numpy() + rng.normal(0, 1, (len(noisy), len(columns))) * scale
        yield f"noise_{group}", noisy


def rare_positive_auc(labels, scores, rng, draws=200):
    labels = np.asarray(labels)
    negatives = np.flatnonzero(labels == 0)
    positives = np.flatnonzero(labels == 1)
    keep = max(5, int(round(RARE_SHARE / (1 - RARE_SHARE) * len(negatives))))
    keep = min(keep, len(positives))
    aucs = []
    for _ in range(draws):
        index = np.r_[negatives, rng.choice(positives, keep, replace=False)]
        aucs.append(roc_auc_score(labels[index], scores[index]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 5))


def run(names):
    train, test, _, _ = load_data()
    labeled = labeled_frame(train)
    groups = get_feature_groups(train.columns)
    registry, blends = build_registry(train.columns)
    members = sorted({m for name in names for m in (blends[name] if name in blends else [name])})
    all_columns = groups["all"]
    drift = test[all_columns].median() - labeled[all_columns].median()

    step_share = labeled.groupby("time_step")["label"].mean()
    high_steps = set(step_share.index[step_share > HIGH_ILLICIT_SHARE])
    times = labeled["time_step"].to_numpy()
    labels = labeled["label"].to_numpy()

    frames = []
    for window_name, window in REPORT_WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        reduced_pos = train_pos[~np.isin(times[train_pos], list(high_steps))]
        for name in members:
            candidate = registry[name]
            columns = list(candidate.columns)
            X_train = labeled.iloc[train_pos][columns]
            X_valid = labeled.iloc[valid_pos][columns]
            for seed in STRESS_SEEDS:
                rng = np.random.default_rng(seed)
                model = fit_model(candidate, X_train, labels[train_pos], seed, times[train_pos])
                outputs = {
                    scenario: model.predict_proba(frame)[:, 1]
                    for scenario, frame in scenario_frames(X_valid, X_train, drift, groups, rng)
                }
                reduced = fit_model(candidate, labeled.iloc[reduced_pos][columns], labels[reduced_pos], seed, times[reduced_pos])
                outputs["withhold_high_illicit"] = reduced.predict_proba(X_valid)[:, 1]
                for scenario, probability in outputs.items():
                    frames.append(pd.DataFrame({
                        "scenario": scenario, "candidate": name, "window": window_name, "seed": seed,
                        "position": valid_pos, "time_step": times[valid_pos], "label": labels[valid_pos],
                        "probability": probability,
                    }))
            print(f"  {window_name} {name}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)

    rows = []
    rng = np.random.default_rng(0)
    for scenario, scenario_frame in predictions.groupby("scenario"):
        scenario_frame = scenario_frame.drop(columns="scenario")
        for name in names:
            if name in blends:
                scenario_frame = add_rank_blend(scenario_frame, blends[name], name)
        bagged = scenario_frame.groupby(["candidate", "window", "position"], as_index=False).agg(
            label=("label", "first"), probability=("probability", "mean")
        )
        for (name, window_name), group in bagged.groupby(["candidate", "window"]):
            if name not in names:
                continue
            rows.append({"candidate": name, "window": window_name, "scenario": scenario,
                         "roc_auc": roc_auc_score(group["label"], group["probability"])})
            if scenario == "clean":
                mean_auc, p5_auc = rare_positive_auc(group["label"].to_numpy(), group["probability"].to_numpy(), rng)
                rows.append({"candidate": name, "window": window_name, "scenario": "rare_positives", "roc_auc": mean_auc})
                rows.append({"candidate": name, "window": window_name, "scenario": "rare_positives_p5", "roc_auc": p5_auc})
    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "stress_window_results.csv", index=False)

    table = results.pivot_table(index="candidate", columns="scenario", values="roc_auc", aggfunc="mean")
    table["worst_scenario_auc"] = table.min(axis=1)
    table["worst_scenario"] = table.drop(columns="worst_scenario_auc").idxmin(axis=1)
    worst_window = results.groupby("candidate")["roc_auc"].min().rename("worst_window_any_scenario")
    table = table.join(worst_window).sort_values("worst_scenario_auc", ascending=False)
    table.to_csv(OUTPUT_DIR / "stress_summary.csv")
    print(f"High-illicit training steps withheld: {sorted(high_steps)}")
    print(table.round(4).to_string())
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    run(args.names)


if __name__ == "__main__":
    main()
