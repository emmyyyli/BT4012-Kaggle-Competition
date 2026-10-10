"""Step 4: hyperparameter search and a fixed-weight, seed-bagged rank ensemble.

``tune``      Optuna (TPE) search per (model family, feature set). The objective is the
              mean seed ROC-AUC over ``TUNING_WINDOWS`` (validation never later than step 28),
              so steps 29-35 stay unseen by the search. Best settings are written to
              ``artifacts/tuning/best_params.json`` in the format read by make_submissions.

``evaluate``  Score every tuned model, the step-1 references and pre-declared rank blends
              on the confirmation windows (validation 29-35) and report windows with five
              seeds. Blend weights are fixed by rule (equal weights) or chosen on tuning
              windows only, never on the windows they are reported on.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import (
    CONFIRM_WINDOWS,
    DEFAULT_SEEDS,
    REPORT_WINDOWS,
    TUNING_WINDOWS,
    Candidate,
    add_rank_blend,
    bag_seeds,
    evaluate_predictions,
    labeled_frame,
    run_candidates,
    window_metrics,
    window_table,
)
from src.features import get_feature_groups
from src.model_zoo import catboost_factory, lgbm_factory, xgb_factory

OUTPUT_DIR = Path("artifacts/tuning")
TUNING_SEEDS = (0, 1)
FACTORIES = {"lgbm": lgbm_factory, "xgb": xgb_factory, "catboost": catboost_factory}
TRIALS = {"lgbm": 40, "xgb": 25, "catboost": 15}
STABILITY_PATH = Path("artifacts/features/shap_stability_tuning_windows.csv")
ADVERSARIAL_PATH = Path("artifacts/features/adversarial_rankings.csv")
# Kaggle scored aggregated-only far below all-features (0.880 vs 0.943), and the 50/50 blend
# in between (0.926). Tuning therefore centres on all-feature and local-feature models.
DEFAULT_FEATURE_SETS = ("all", "all_no_feat2", "local")
TEST_DRIFT_DROP = 5


def extended_feature_sets(groups):
    """Feature groups plus the step-3 leads.

    ``all_no_feat2`` drops feat_2: the best single-cluster drop in step 3 and one of the most
    drifted features between training and test. ``stable20`` is the 20 features with the
    highest minimum TreeSHAP share across the step-3 stability windows.
    ``all_no_test_drift5`` drops the five features that best separate labeled training rows
    from Kaggle test rows (no labels used). Window validation cannot measure the train->test
    drift that motivates it, so only a Kaggle score can judge this set.
    """
    sets = dict(groups)
    sets["all_no_feat2"] = [column for column in groups["all"] if column != "feat_2"]
    if STABILITY_PATH.exists():
        ranking = pd.read_csv(STABILITY_PATH, index_col="feature")
        sets["stable20"] = ranking.sort_values("min_share", ascending=False).index[:20].tolist()
    if ADVERSARIAL_PATH.exists():
        drifted = pd.read_csv(ADVERSARIAL_PATH, index_col=0)["train_vs_test"].nlargest(TEST_DRIFT_DROP).index
        sets["all_no_test_drift5"] = [column for column in groups["all"] if column not in set(drifted)]
    return sets


def suggest_params(trial, family):
    if family == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 32, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        )
    if family == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        )
    return dict(
        iterations=trial.suggest_int("iterations", 300, 1000, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        depth=trial.suggest_int("depth", 3, 7),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        rsm=trial.suggest_float("rsm", 0.2, 0.9),
    )


def tune(labeled, groups, families, feature_sets, trials_override=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / "best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    for family in families:
        for feature_set in feature_sets:
            name = f"tuned_{family}_{feature_set}"
            columns = tuple(groups[feature_set])

            def objective(trial):
                params = suggest_params(trial, family)
                candidate = Candidate(name, columns, FACTORIES[family](params))
                predictions = run_candidates(labeled, [candidate], TUNING_WINDOWS, TUNING_SEEDS, verbose=False)
                aucs = window_metrics(bag_seeds(predictions))["roc_auc"]
                return float(aucs.mean())

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            # Enqueue the hand-picked regularised setting so the search starts from it.
            if family == "lgbm":
                study.enqueue_trial(dict(n_estimators=600, learning_rate=0.03, num_leaves=8, min_child_samples=100, colsample_bytree=0.5, subsample=0.8, reg_lambda=5.0))
            n_trials = trials_override or TRIALS[family]
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            study.trials_dataframe().to_csv(OUTPUT_DIR / f"{name}_trials.csv", index=False)
            params = suggest_params(optuna.trial.FixedTrial(study.best_params), family)
            best[name] = {
                "family": family,
                "feature_set": feature_set,
                "columns": list(columns),
                "params": params,
                "tuning_objective": study.best_value,
                "trials": n_trials,
            }
            best_path.write_text(json.dumps(best, indent=2))
            print(f"{name}: best tuning-window AUC {study.best_value:.4f} with {params}", flush=True)
    return best


def choose_blend_weight(labeled, candidates, short_name, long_name):
    """Pick the weight on ``long_name`` from a 0..1 grid using tuning windows only."""
    predictions = run_candidates(labeled, candidates, TUNING_WINDOWS, TUNING_SEEDS, verbose=False)
    scores = {}
    for weight in np.linspace(0, 1, 11):
        blended = add_rank_blend(predictions, {short_name: 1 - weight + 1e-9, long_name: weight + 1e-9}, "b")
        aucs = window_metrics(bag_seeds(blended.loc[blended["candidate"] == "b"]))["roc_auc"]
        scores[round(float(weight), 1)] = float(aucs.mean())
    return max(scores, key=scores.get), scores


def evaluate(labeled, groups):
    best = json.loads((OUTPUT_DIR / "best_params.json").read_text())
    candidates = [Candidate("shallow_all", tuple(groups["all"]), lgbm_factory())]
    for name, spec in best.items():
        candidates.append(Candidate(name, tuple(spec["columns"]), FACTORIES[spec["family"]](spec["params"])))
    if "tuned_lgbm_all" in best and "all_no_test_drift5" in groups:
        # Not tuned separately: the tuned all-feature LightGBM minus the 5 most test-drifted features.
        params = best["tuned_lgbm_all"]["params"]
        candidates.append(Candidate("tuned_lgbm_all_no_test_drift5", tuple(groups["all_no_test_drift5"]), lgbm_factory(params)))
        best["tuned_lgbm_all_no_test_drift5"] = {**best["tuned_lgbm_all"], "feature_set": "all_no_test_drift5", "columns": list(groups["all_no_test_drift5"])}
        (OUTPUT_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    by_name = {candidate.name: candidate for candidate in candidates}

    blends = {}
    weight_log = {}
    # Cross-family ensembles: diversity from model families on one feature set, equal weights
    # fixed in advance.
    for feature_set in ("all", "all_no_feat2"):
        members = [f"tuned_{family}_{feature_set}" for family in FACTORIES if f"tuned_{family}_{feature_set}" in by_name]
        if len(members) > 1:
            blends[f"ensemble_families_{feature_set}"] = {name: 1.0 for name in members}
    # All-feature + local-feature pairing per family; the local weight is chosen on tuning windows.
    for family in FACTORIES:
        all_name, local_name = f"tuned_{family}_all", f"tuned_{family}_local"
        if all_name in by_name and local_name in by_name:
            weight, scores = choose_blend_weight(labeled, [by_name[all_name], by_name[local_name]], all_name, local_name)
            weight_log[family] = {"chosen_local_weight": weight, "tuning_scores": scores}
            if 0 < weight < 1:
                blends[f"blend_{family}_all_local_tunedweight"] = {all_name: 1 - weight, local_name: weight}
    (OUTPUT_DIR / "blend_weight_selection.json").write_text(json.dumps(weight_log, indent=2))

    windows = {**CONFIRM_WINDOWS, **REPORT_WINDOWS}
    predictions = run_candidates(labeled, candidates, windows, DEFAULT_SEEDS)
    for blend_name, members in blends.items():
        predictions = add_rank_blend(predictions, members, blend_name)
    predictions.to_csv(OUTPUT_DIR / "step4_predictions.csv.gz", index=False)
    tables = evaluate_predictions(predictions, "shallow_all", OUTPUT_DIR, "step4")
    (OUTPUT_DIR / "blends.json").write_text(json.dumps(blends, indent=2))

    table = window_table(tables["window_metrics"])
    table.to_csv(OUTPUT_DIR / "step4_window_table.csv")
    print(table.round(4).to_string())
    print(tables["summary"].round(4).to_string(index=False))
    print(json.dumps(weight_log, indent=2))
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "evaluate"])
    parser.add_argument("--families", nargs="+", default=list(FACTORIES))
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, _, _, _ = load_data()
    labeled = labeled_frame(train)
    groups = extended_feature_sets(get_feature_groups(train.columns))
    if args.command == "tune":
        tune(labeled, groups, args.families, args.feature_sets, args.trials)
    else:
        evaluate(labeled, groups)


if __name__ == "__main__":
    main()
