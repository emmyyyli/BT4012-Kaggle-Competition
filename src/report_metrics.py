"""Figures and triage metrics for the written report (report/e1304439.pdf).

Outputs:
* ``artifacts/report/precision_recall.csv``: ROC-AUC, PR-AUC and triage operating points for
  the final graph stack vs the non-graph tuned LightGBM, on the two Kaggle-aligned validation
  windows (3 seeds, seed-averaged predictions). The operating points are: precision at
  k = number of illicit rows (precision = recall = F1 there); precision when 90% / 95% of
  illicit rows are caught; and the best-F1 threshold with its precision and recall. These
  adapt to each window's illicit share (about 18%, versus about 6% in the test period), so
  fixed review budgets would saturate.
* ``report/figures/fig1_public_auc.png``: Kaggle public ROC-AUC of every submission, with
  the best-so-far line and the changes that moved it.
* ``report/figures/fig2_context_coverage.png``: share of labeled training rows vs test rows
  with labeled graph context at one and two hops.

Kaggle scores are not recomputable locally (test labels are hidden); they are transcribed
from the leaderboard as recorded in report/findings.md.

Run from the project root: ``python -m src.report_metrics``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.evaluation import Candidate, fit_model, labeled_frame, window_positions
from src.features import get_feature_groups
from src.final_config import FINAL_NAME, FINAL_STACK, REFERENCE_MEMBERS
from src.graph_stacking import build_graph, make_neighbour_stack, two_hop_features
from src.model_zoo import lgbm_factory

FIGURE_DIR = Path("report/figures")
TABLE_DIR = Path("artifacts/report")
WINDOWS = {"train <= 25, validate 26-35": (25, 26, 35), "train <= 28, validate 29-35": (28, 29, 35)}
SEEDS = (0, 1, 2)
RECALL_TARGETS = (0.90, 0.95)

# Palette (validated categorical slots 1-2 of the reference palette; text uses ink tokens).
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#a3a29c"

# (round, label, public ROC-AUC, adopted as the new best?) in submission order.
SUBMISSIONS = [
    (0, "original LightGBM (+time_step, in_degree)", 0.94246, True),
    (0, "0.7 LightGBM + 0.3 GraphSAGE", 0.91059, False),
    (1, "regularised LightGBM", 0.94267, True),
    (1, "all + aggregated-only blend", 0.92624, False),
    (1, "aggregated-only", 0.88023, False),
    (2, "minus 5 drifted features", 0.94365, True),
    (2, "minus feat_2", 0.93673, False),
    (2, "tuned LightGBM", 0.94417, True),
    (3, "recency weights", 0.95066, True),
    (3, "temporal bagging", 0.94672, False),
    (3, "neighbour stacking", 0.95389, True),
    (4, "stack + non-graph blend", 0.95328, False),
    (4, "stack + recency", 0.95858, True),
    (4, "stack + rich context", 0.95871, True),
    (5, "rich + recency", 0.96342, True),
    (5, "wide + recency", 0.96506, True),
    (5, "rich + two-hop", 0.96668, True),
    (6, "3-hop propagation", 0.96108, False),
    (6, "wide + two-hop", 0.96814, True),
    (6, "blend of 3 graph models", 0.96587, False),
    (7, "final model, 30 seeds", 0.96785, False),
    (7, "half-life 21", 0.96670, False),
    (7, "blend of 4 graph models (pick 2)", 0.96686, False),
    (8, "extra collective round", 0.96698, False),
    (9, "direction-split two-hop", 0.96787, False),
    (10, "forward-only stage-1 scores", 0.96503, False),
    (11, "+ centrality", 0.96759, False),
    (11, "+ peel-chain shape", 0.96858, True),
    (11, "+ hub-weighted means", 0.96766, False),
    (12, "+ test-similarity weights (final)", 0.9702, True),
]
LEADER_AUC = 0.97012
PUBLIC_SE = 0.0077  # Hanley-McNeil SE of one public AUC (~270 illicit rows), findings 20.2


def operating_points(labels, scores):
    """Precision/recall/F1 at operating points an investigator could choose."""
    positives = int(labels.sum())
    top = np.argsort(-scores)[:positives]
    result = {"precision=recall@k=illicit": float(labels[top].mean())}
    precision, recall, _ = precision_recall_curve(labels, scores)
    for target in RECALL_TARGETS:
        result[f"precision@recall{target:.0%}"] = float(precision[recall >= target].max())
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    best = int(np.argmax(f1))
    result.update({"best_f1": float(f1[best]), "best_f1_precision": float(precision[best]), "best_f1_recall": float(recall[best])})
    return result


def precision_recall_table(labeled, graph):
    columns = get_feature_groups(labeled.columns)["all"]
    spec = REFERENCE_MEMBERS[0]
    reference = Candidate(spec["name"], tuple(columns), lgbm_factory(spec["params"]))
    stack = make_neighbour_stack(**FINAL_STACK)
    times, labels = labeled["time_step"].to_numpy(), labeled["label"].to_numpy()
    rows = []
    for window_name, window in WINDOWS.items():
        train_pos, valid_pos = window_positions(times, window)
        train, target = labeled.iloc[train_pos], labeled.iloc[valid_pos]
        y = labels[valid_pos]
        predictions = {
            spec["name"]: np.mean([
                fit_model(reference, train[columns], labels[train_pos], seed).predict_proba(target[columns])[:, 1]
                for seed in SEEDS
            ], axis=0),
            FINAL_NAME: np.mean([stack(train, target, columns, seed, graph) for seed in SEEDS], axis=0),
        }
        for model, scores in predictions.items():
            row = {"window": window_name, "model": model, "valid_rows": len(y), "illicit": int(y.sum()),
                   "roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores)}
            row.update(operating_points(y, scores))
            rows.append(row)
        print(f"  {window_name}", flush=True)
    return pd.DataFrame(rows)


def coverage_table(labeled, test, graph):
    rows = pd.concat([labeled[["txId"]], test[["txId"]]], ignore_index=True)
    scored = set(rows["txId"])
    one_hop = graph[graph["u"].isin(scored) & graph["v"].isin(scored)].groupby("u").size()
    placeholder = rows.assign(**{c: 0.0 for c in ["feat_3", "feat_5", "feat_53", "feat_59", "feat_61"]})
    two_hop = two_hop_features(graph, placeholder, pd.Series(0.0, index=rows["txId"]))["hop2_labeled_walks"]
    result = {}
    for name, frame in [("Labeled training rows", labeled), ("Test rows", test)]:
        has_one = one_hop.reindex(frame["txId"]).fillna(0).to_numpy() > 0
        has_two = two_hop.reindex(frame["txId"]).fillna(0).to_numpy() > 0
        result[name] = {"1-hop labeled neighbour": has_one.mean(), "1- or 2-hop labeled context": (has_one | has_two).mean()}
    return pd.DataFrame(result).T


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_public_auc(path):
    frame = pd.DataFrame(SUBMISSIONS, columns=["round", "label", "auc", "adopted"])
    frame["x"] = np.arange(1, len(frame) + 1)
    frame["best"] = frame["auc"].cummax()
    frame["new_best"] = frame["auc"] >= frame["best"]
    fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.axhline(LEADER_AUC, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.6, LEADER_AUC + 0.0013, f"leaderboard top {LEADER_AUC:.4f}", color=INK_SECONDARY, fontsize=8)
    ax.step(frame["x"], frame["best"], where="post", color=BLUE, linewidth=2, label="best public score so far")
    other = frame[~frame["new_best"]]
    ax.scatter(other["x"], other["auc"], s=34, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=3,
               label="submission (not a new best)")
    best = frame[frame["new_best"]]
    ax.scatter(best["x"], best["auc"], s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="submission (new best)")
    final = frame.loc[frame["auc"].idxmax()]
    ax.errorbar(final["x"], final["auc"], yerr=PUBLIC_SE, color=BLUE, linewidth=1, capsize=3, zorder=2)
    notes = {
        "tuned LightGBM": (6, -11, "left"), "recency weights": (-5, 9, "right"), "neighbour stacking": (4, -12, "left"),
        "stack + rich context": (-5, 9, "right"), "wide + two-hop": (-8, 15, "right"),
        "+ test-similarity weights (final)": (4, -42, "right"),
        "aggregated-only": (7, 0, "left"), "0.7 LightGBM + 0.3 GraphSAGE": (7, 0, "left"), "3-hop propagation": (0, -13, "center"),
    }
    for _, row in frame.iterrows():
        if row["label"] in notes:
            dx, dy, ha = notes[row["label"]]
            ax.annotate(f"{row['label']} {row['auc']:.4f}", (row["x"], row["auc"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7.2, color=INK, ha=ha, va="center")
    bounds = frame.groupby("round")["x"].agg(["min", "max"])
    for round_id, (lo, hi) in bounds.iterrows():
        if round_id:
            ax.axvline(lo - 0.5, color=GRID, linewidth=0.8)
        ax.text((lo + hi) / 2, 0.8745, f"R{round_id}", ha="center", color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(0.4, len(frame) + 0.6)
    ax.set_ylim(0.872, 0.981)
    ax.set_xticks([])
    ax.set_ylabel("Kaggle public ROC-AUC", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Public ROC-AUC by submission (R = submission round; bar = \u00b11 SE of one public score)",
                 color=INK, fontsize=9.5, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.41, 0.07), frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def figure_coverage(coverage, path):
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    categories = list(coverage.columns)
    y = np.arange(len(categories))
    height = 0.3
    for offset, (group, colour) in zip((-height / 2 - 0.01, height / 2 + 0.01), [("Labeled training rows", ORANGE), ("Test rows", BLUE)]):
        values = coverage.loc[group].to_numpy()
        ax.barh(y + offset, values, height=height, color=colour, label=group)
        for yi, value in zip(y + offset, values):
            ax.text(value + 0.008, yi, f"{value:.1%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, categories, color=INK, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Rows with labeled graph context", color=INK, fontsize=10, loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5,
              labelcolor=INK_SECONDARY, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figure_public_auc(FIGURE_DIR / "fig1_public_auc.png")
    train, test, edges, _ = load_data()
    labeled = labeled_frame(train)
    graph = build_graph(edges)
    coverage = coverage_table(labeled, test, graph)
    coverage.to_csv(TABLE_DIR / "context_coverage.csv")
    figure_coverage(coverage, FIGURE_DIR / "fig2_context_coverage.png")
    table = precision_recall_table(labeled, graph)
    table.to_csv(TABLE_DIR / "precision_recall.csv", index=False)
    print(coverage.round(3).to_string())
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
