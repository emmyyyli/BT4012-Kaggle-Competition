import numpy as np
import pandas as pd
import pytest

from src.run_step8_variants import build_graph, neighbour_score_features, neighbour_stack, recent_steps


def test_neighbour_features_use_only_scored_neighbours_and_keep_direction():
    edges = pd.DataFrame({"txId1": ["a", "a", "x", "c"], "txId2": ["b", "c", "b", "b"]})
    scores = pd.Series({"a": 0.2, "b": 0.9, "c": 0.4})  # "x" is unscored (unlabeled / featureless)
    features = neighbour_score_features(build_graph(edges), scores)

    assert features.loc["b", "nbr_score_mean"] == pytest.approx(0.3)  # from a and c, not x
    assert features.loc["b", "nbr_score_mean_in"] == pytest.approx(0.3)
    assert np.isnan(features.loc["b", "nbr_score_mean_out"])
    assert features.loc["a", "nbr_score_max"] == pytest.approx(0.9)
    assert features.loc["a", "nbr_score_mean_out"] == pytest.approx(0.65)


def _frame(steps, seed):
    rng = np.random.default_rng(seed)
    n = len(steps)
    label = rng.integers(0, 2, n)
    return pd.DataFrame({"txId": [f"{seed}_{i}" for i in range(n)], "time_step": steps, "label": label,
                         "feat_1": label + rng.normal(0, 0.5, n), "feat_2": rng.normal(0, 1, n)})


def test_recent_steps_and_stacking_return_one_score_per_target_row():
    train = _frame(np.repeat(np.arange(1, 11), 30), 0)
    target = _frame(np.repeat([11, 12], 20), 1)
    edges = pd.DataFrame({"txId1": train["txId"].iloc[:-1].to_numpy(), "txId2": train["txId"].iloc[1:].to_numpy()})

    recent = recent_steps(3)(train, target, ["feat_1", "feat_2"], seed=0)
    stacked = neighbour_stack(train, target, ["feat_1", "feat_2"], seed=0, graph=build_graph(edges))

    assert recent.shape == stacked.shape == (len(target),)
    assert np.all((stacked >= 0) & (stacked <= 1))


def test_round4_stack_variants_return_one_score_per_target_row():
    from src.run_step8_variants import RICH_NEIGHBOUR_FEATURES, make_neighbour_stack, neighbour_context_features

    columns = ["feat_1", "feat_2", *RICH_NEIGHBOUR_FEATURES]
    train = _frame(np.repeat(np.arange(1, 11), 30), 0)
    target = _frame(np.repeat([11, 12], 20), 1)
    for frame in (train, target):
        for column in RICH_NEIGHBOUR_FEATURES:
            frame[column] = np.arange(len(frame), dtype=float)
    edges = pd.DataFrame({"txId1": [*train["txId"].iloc[:-1], "ghost"], "txId2": [*train["txId"].iloc[1:], train["txId"].iloc[0]]})
    graph = build_graph(edges)

    context = neighbour_context_features(graph, pd.concat([train, target], ignore_index=True))
    first = train["txId"].iloc[0]
    assert context.loc[first, "in_degree"] == 1  # the edge from "ghost" counts toward degree
    assert context.loc[first, "n_labeled_nbrs"] == 1  # ...but "ghost" is not a labeled neighbour
    assert context.loc[first, "frac_labeled_nbrs"] == pytest.approx(0.5)

    for variant in (make_neighbour_stack(half_life_steps=14), make_neighbour_stack(rich=True)):
        scores = variant(train, target, columns, 0, graph)
        assert scores.shape == (len(target),)
        assert np.all((scores >= 0) & (scores <= 1))
