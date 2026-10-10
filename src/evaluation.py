"""Multi-window, multi-seed temporal evaluation harness.

Single-seed, single-holdout comparisons proved too noisy on this dataset (seed-to-seed
ROC-AUC spread of 0.01-0.02 on the steps 1-21 -> 22-35 holdout), so every candidate is
scored the same way:

* several forward-in-time windows (train on steps <= t, validate on a later block);
* several random seeds per window, plus the seed-averaged ("bagged") prediction, which
  mirrors how the final submission averages seeds;
* ROC-AUC per validation time step, because pooled AUC can hide step-specific failures;
* a paired row bootstrap of the bagged AUC difference against a named baseline.

Adoption rule: a candidate replaces the baseline only if its mean bagged ROC-AUC across
windows improves and no single window loses more than ``MAX_WINDOW_LOSS``.

Three window sets are defined. ``REPORT_WINDOWS`` is used for reporting and adoption.
For hyperparameter and blend-weight search, ``TUNING_WINDOWS`` validates only up to
step 28, and ``CONFIRM_WINDOWS`` (validating from step 26 or 29 onward) is held back to
check that tuned choices transfer to later steps. Only steps 29-35 are completely unseen
by tuning; steps 26-28 also fall inside the t21_v22-28 tuning window.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


# name -> (last training step, first validation step, last validation step)
REPORT_WINDOWS = {
    "t14_v15-35": (14, 15, 35),
    "t21_v22-35": (21, 22, 35),
    "t21_v22-28": (21, 22, 28),
    "t25_v26-35": (25, 26, 35),
    "t28_v29-35": (28, 29, 35),
}
# Short-horizon windows (about 7 steps). The first Kaggle scores ranked all-features >
# all/aggregated blend > aggregated-only. Only the short report windows (t25, t28) ranked
# the three models the same way; the long windows ranked them in reverse. So tuning and
# confirmation use short horizons. See report/findings.md section 6.
TUNING_WINDOWS = {
    "t14_v15-21": (14, 15, 21),
    "t18_v19-25": (18, 19, 25),
    "t21_v22-28": (21, 22, 28),
}
CONFIRM_WINDOWS = {
    "t25_v26-35": (25, 26, 35),
    "t28_v29-35": (28, 29, 35),
}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
MAX_WINDOW_LOSS = 0.01


@dataclass(frozen=True)
class Candidate:
    """A named model configuration: feature columns plus a seeded model factory."""

    name: str
    columns: tuple
    make_model: Callable
    sample_weight: Optional[Callable] = None  # f(train time steps) -> per-row weights


def labeled_frame(train):
    """Return only rows with a confirmed 0/1 label, with an integer label column."""
    labeled = train.loc[train["label"].notna()].reset_index(drop=True)
    labeled["label"] = labeled["label"].astype(int)
    return labeled


def window_positions(time_steps, window):
    train_end, valid_start, valid_end = window
    if valid_start <= train_end:
        raise ValueError(f"Validation must start after training ends: {window}")
    time_steps = np.asarray(time_steps)
    train_positions = np.flatnonzero(time_steps <= train_end)
    valid_positions = np.flatnonzero((time_steps >= valid_start) & (time_steps <= valid_end))
    if len(train_positions) == 0 or len(valid_positions) == 0:
        raise ValueError(f"Window {window} produced an empty split")
    return train_positions, valid_positions


def fit_model(candidate, X_train, y_train, seed, train_time_steps=None):
    """Fit one seeded model, combining recency and class-balancing weights if needed."""
    model = candidate.make_model(seed)
    weights = None
    if candidate.sample_weight is not None:
        weights = np.asarray(candidate.sample_weight(train_time_steps), dtype=np.float64)
    if getattr(candidate.make_model, "needs_class_weights", False):
        class_weights = compute_sample_weight("balanced", y_train)
        weights = class_weights if weights is None else weights * class_weights
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def run_candidates(labeled, candidates, windows=REPORT_WINDOWS, seeds=DEFAULT_SEEDS, verbose=True):
    """Fit every candidate on every window and seed; return long-format predictions."""
    time_steps = labeled["time_step"].to_numpy()
    labels = labeled["label"].to_numpy()
    frames = []
    for window_name, window in windows.items():
        train_positions, valid_positions = window_positions(time_steps, window)
        for candidate in candidates:
            columns = list(candidate.columns)
            X_train = labeled.iloc[train_positions][columns]
            X_valid = labeled.iloc[valid_positions][columns]
            y_train = labels[train_positions]
            for seed in seeds:
                model = fit_model(
                    candidate, X_train, y_train, seed, time_steps[train_positions]
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "candidate": candidate.name,
                            "window": window_name,
                            "seed": seed,
                            "position": valid_positions,
                            "time_step": time_steps[valid_positions],
                            "label": labels[valid_positions],
                            "probability": model.predict_proba(X_valid)[:, 1],
                        }
                    )
                )
            if verbose:
                print(f"  {window_name:12s} {candidate.name}", flush=True)
    return pd.concat(frames, ignore_index=True)


def add_rank_blend(predictions, members, name):
    """Append a weighted average of per-(window, seed) percentile ranks of ``members``.

    Ranks rather than raw probabilities are averaged so that models with different
    calibration (e.g. class-weighted LightGBM vs CatBoost) contribute on equal terms.
    Weights must be fixed in advance, not fitted on the windows being reported.
    """
    subset = predictions.loc[predictions["candidate"].isin(members)].copy()
    missing = set(members).difference(subset["candidate"].unique())
    if missing:
        raise ValueError(f"Blend members have no predictions: {sorted(missing)}")
    subset["rank"] = subset.groupby(["candidate", "window", "seed"])["probability"].rank(pct=True)
    subset["weighted"] = subset["rank"] * subset["candidate"].map(members)
    blended = subset.groupby(["window", "seed", "position"], as_index=False).agg(
        time_step=("time_step", "first"),
        label=("label", "first"),
        probability=("weighted", "sum"),
    )
    blended["probability"] /= sum(members.values())
    blended["candidate"] = name
    return pd.concat([predictions, blended[predictions.columns]], ignore_index=True)


def bag_seeds(predictions):
    """Average predictions over seeds for each candidate, window and validation row."""
    return predictions.groupby(["candidate", "window", "position"], as_index=False).agg(
        time_step=("time_step", "first"),
        label=("label", "first"),
        probability=("probability", "mean"),
    )


def _safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def seed_metrics(predictions):
    rows = []
    for (candidate, window, seed), group in predictions.groupby(["candidate", "window", "seed"]):
        rows.append(
            {
                "candidate": candidate,
                "window": window,
                "seed": seed,
                "roc_auc": _safe_auc(group["label"], group["probability"]),
                "pr_auc": float(average_precision_score(group["label"], group["probability"])),
            }
        )
    return pd.DataFrame(rows)


def window_metrics(bagged, seeds_frame=None):
    rows = []
    for (candidate, window), group in bagged.groupby(["candidate", "window"]):
        rows.append(
            {
                "candidate": candidate,
                "window": window,
                "roc_auc": _safe_auc(group["label"], group["probability"]),
                "pr_auc": float(average_precision_score(group["label"], group["probability"])),
                "valid_rows": len(group),
                "valid_positives": int(group["label"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if seeds_frame is not None:
        spread = (
            seeds_frame.groupby(["candidate", "window"])["roc_auc"]
            .agg(seed_auc_mean="mean", seed_auc_std="std")
            .reset_index()
        )
        result = result.merge(spread, on=["candidate", "window"], how="left")
    return result


def step_metrics(bagged):
    rows = []
    for (candidate, window, step), group in bagged.groupby(["candidate", "window", "time_step"]):
        rows.append(
            {
                "candidate": candidate,
                "window": window,
                "time_step": int(step),
                "roc_auc": _safe_auc(group["label"], group["probability"]),
                "rows": len(group),
                "positives": int(group["label"].sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(bagged, baseline, iterations=300, random_state=0):
    """Bootstrap validation rows per window; CI of bagged AUC(candidate) - AUC(baseline).

    The interval conditions on the fitted models (it resamples rows, not training runs),
    so it complements rather than replaces the seed spread.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for window, window_frame in bagged.groupby("window"):
        pivot = window_frame.pivot(index="position", columns="candidate", values="probability")
        labels = window_frame.drop_duplicates("position").set_index("position").loc[pivot.index, "label"].to_numpy()
        if baseline not in pivot:
            continue
        samples = [rng.integers(0, len(labels), len(labels)) for _ in range(iterations)]
        samples = [index for index in samples if len(np.unique(labels[index])) == 2]
        base_scores = pivot[baseline].to_numpy()
        base_boot = np.array([roc_auc_score(labels[i], base_scores[i]) for i in samples])
        for candidate in pivot.columns:
            if candidate == baseline:
                continue
            scores = pivot[candidate].to_numpy()
            deltas = np.array([roc_auc_score(labels[i], scores[i]) for i in samples]) - base_boot
            rows.append(
                {
                    "candidate": candidate,
                    "window": window,
                    "delta_ci_2_5": float(np.percentile(deltas, 2.5)),
                    "delta_ci_97_5": float(np.percentile(deltas, 97.5)),
                    "prob_delta_positive": float((deltas > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize(windows_frame, baseline, max_window_loss=MAX_WINDOW_LOSS):
    """One row per candidate: mean/worst bagged AUC, deltas and the adoption decision."""
    pivot = windows_frame.pivot(index="window", columns="candidate", values="roc_auc")
    rows = []
    for candidate in pivot.columns:
        delta = pivot[candidate] - pivot[baseline]
        seed_std = windows_frame.loc[windows_frame["candidate"] == candidate].get("seed_auc_std")
        rows.append(
            {
                "candidate": candidate,
                "mean_roc_auc": float(pivot[candidate].mean()),
                "worst_roc_auc": float(pivot[candidate].min()),
                "mean_delta_vs_baseline": float(delta.mean()),
                "min_window_delta": float(delta.min()),
                "windows_improved": int((delta > 0).sum()),
                "windows": int(delta.notna().sum()),
                "mean_seed_auc_std": float(seed_std.mean()) if seed_std is not None else np.nan,
                "adopt": bool(
                    candidate != baseline and delta.mean() > 0 and delta.min() >= -max_window_loss
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False, ignore_index=True)


def evaluate_predictions(predictions, baseline, output_dir=None, prefix=None, bootstrap_iterations=300):
    """Compute every metric table and optionally write them as ``<prefix>_<table>.csv``."""
    seeds_frame = seed_metrics(predictions)
    bagged = bag_seeds(predictions)
    windows_frame = window_metrics(bagged, seeds_frame)
    tables = {
        "seed_metrics": seeds_frame,
        "window_metrics": windows_frame,
        "step_metrics": step_metrics(bagged),
        "bootstrap": paired_bootstrap(bagged, baseline, iterations=bootstrap_iterations),
        "summary": summarize(windows_frame, baseline),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for table_name, table in tables.items():
            table.to_csv(output_dir / f"{prefix}_{table_name}.csv", index=False)
    return tables


def window_table(windows_frame, candidates=None):
    """Candidate x window bagged ROC-AUC table, for printing and the findings log."""
    table = windows_frame.pivot(index="candidate", columns="window", values="roc_auc")
    if candidates is not None:
        table = table.loc[[c for c in candidates if c in table.index]]
    table["mean"] = table.mean(axis=1)
    return table.sort_values("mean", ascending=False)
