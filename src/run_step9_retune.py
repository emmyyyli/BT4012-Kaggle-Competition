"""Step 9: re-tune the shared LightGBM settings on the validation window that tracks Kaggle.

The settings in ``TUNED_LGBM_PARAMS`` were tuned (step 4) on the mean of three windows, one of which
(train <= 21 -> validate 22-28) ranks models almost opposite to the public leaderboard
(report/findings.md section 46). Here the full final stacked model is re-tuned on train <= 28 ->
validate 29-35, the window whose ranking agrees with Kaggle (Spearman +0.81).

Usage:
    python -m src.run_step9_retune tune 20      # Optuna trials, seed 0 (current settings enqueued first)
    python -m src.run_step9_retune confirm      # 3 seeds on t28_v29-35 and t25_v26-35: best trial, top-5 consensus (+ half-life 7)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import REPORT_WINDOWS, labeled_frame, window_positions
from src.features import get_feature_groups
from src.final_config import FINAL_STACK
from src.graph_stacking import TUNED_LGBM_PARAMS, build_graph, make_neighbour_stack

OUTPUT = Path("artifacts/tuning/step9_retune.json")


def window_auc(labeled, columns, graph, params, window, seed, half_life=None):
    train_pos, valid_pos = window_positions(labeled["time_step"].to_numpy(), REPORT_WINDOWS[window])
    train, target = labeled.iloc[train_pos], labeled.iloc[valid_pos]
    stack = dict(FINAL_STACK) if half_life is None else {**FINAL_STACK, "half_life_steps": half_life}
    model = make_neighbour_stack(**stack, base_params=params)
    return roc_auc_score(target["label"].to_numpy(), model(train, target, columns, seed, graph))


def suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 5, 40, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "confirm"])
    parser.add_argument("trials", nargs="?", type=int, default=20)
    args = parser.parse_args()
    train, test, edges, _ = load_data()
    labeled = labeled_frame(train)
    columns = get_feature_groups(train.columns)["all"]
    graph = build_graph(edges)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "tune":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.enqueue_trial({k: v for k, v in TUNED_LGBM_PARAMS.items() if k != "subsample_freq"})

        def objective(trial):
            auc = window_auc(labeled, columns, graph, suggest(trial), "t28_v29-35", 0)
            print(f"trial {trial.number}: {auc:.5f} {trial.params}", flush=True)
            return auc

        study.optimize(objective, n_trials=args.trials)
        trials = [{"number": t.number, "auc": t.value, "params": t.params} for t in study.trials]
        OUTPUT.write_text(json.dumps({"best": study.best_trial.params, "best_auc": study.best_value, "trials": trials}, indent=2))
        print("best", study.best_value, study.best_trial.params)
    else:
        saved = json.loads(OUTPUT.read_text())
        candidates = {
            "best_trial": ({**saved["best"], "subsample_freq": 1}, 14),
            "consensus_top5": (saved["consensus_top5"], 14),
            "consensus_top5_hl7": (saved["consensus_top5"], 7),
        }
        for window in ("t28_v29-35", "t25_v26-35"):
            for name, (params, half_life) in candidates.items():
                aucs = [window_auc(labeled, columns, graph, params, window, seed, half_life) for seed in range(3)]
                print(f"{window} {name}: mean {np.mean(aucs):.5f} sd {np.std(aucs):.5f}", flush=True)


if __name__ == "__main__":
    main()
