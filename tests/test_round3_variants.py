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


def test_two_hop_features_reach_through_unlabeled_nodes_but_not_back_to_self():
    from src.run_step8_variants import two_hop_features

    # a - x - c - d : x is unlabeled (not in frame); a's two-hop labeled node is c only.
    edges = pd.DataFrame({"txId1": ["a", "x", "c"], "txId2": ["x", "c", "d"]})
    frame = pd.DataFrame({"txId": ["a", "c", "d"], "feat_1": [1.0, 3.0, 5.0]})
    scores = pd.Series({"a": 0.1, "c": 0.7, "d": 0.9})
    out = two_hop_features(build_graph(edges), frame, scores, mean_features=["feat_1"])

    assert out.loc["a", "hop2_labeled_walks"] == 1
    assert out.loc["a", "hop2_score_mean"] == pytest.approx(0.7)
    assert out.loc["a", "hop2_mean_feat_1"] == pytest.approx(3.0)
    assert out.loc["c", "hop2_labeled_walks"] == 1  # c -> x -> a (c -> d -> c is excluded)
    assert out.loc["c", "hop2_score_mean"] == pytest.approx(0.1)
    assert out.loc["d", "hop2_labeled_walks"] == 0  # d -> c -> x reaches only an unlabeled node
    assert np.isnan(out.loc["d", "hop2_score_mean"])


def test_wide_features_cover_stats_and_directions():
    from src.run_step8_variants import neighbour_wide_features

    edges = pd.DataFrame({"txId1": ["a", "b"], "txId2": ["c", "c"]})
    frame = pd.DataFrame({"txId": ["a", "b", "c"], "feat_9": [1.0, 3.0, 10.0]})
    out = neighbour_wide_features(build_graph(edges), frame, features=["feat_9"])

    assert out.loc["c", "nbrw_mean_feat_9"] == pytest.approx(2.0)
    assert out.loc["c", "nbrw_max_feat_9"] == pytest.approx(3.0)
    assert out.loc["c", "nbrw_in_mean_feat_9"] == pytest.approx(2.0)
    assert np.isnan(out.loc["c", "nbrw_out_mean_feat_9"])
    assert out.loc["a", "nbrw_out_mean_feat_9"] == pytest.approx(10.0)


def test_propagation_passes_through_featureless_nodes_with_decay():
    from src.graph_stacking import propagation_features

    # a - x - c : x has no features (not in frame). One hop from a reaches only x (no mass);
    # two hops reach a (itself) and c with equal probability.
    edges = pd.DataFrame({"txId1": ["a", "x"], "txId2": ["x", "c"]})
    frame = pd.DataFrame({"txId": ["a", "c"]})
    scores = pd.Series({"a": 0.2, "c": 0.8})
    out = propagation_features(build_graph(edges), frame, scores, hops=2, decay=0.5)

    # hop1: P m at a = 0 ; hop2: P^2 m at a = 1 (half to a, half to c), weighted by 0.5
    assert out.loc["a", "prop_reach"] == pytest.approx(0.5)
    assert out.loc["a", "prop_score"] == pytest.approx(0.5)  # (0.5*0.2 + 0.5*0.8) / 1


def test_extra_round_of_collective_classification_returns_scores():
    from src.graph_stacking import RICH_NEIGHBOUR_FEATURES, make_neighbour_stack

    columns = ["feat_1", "feat_2", *RICH_NEIGHBOUR_FEATURES]
    train = _frame(np.repeat(np.arange(1, 11), 30), 0)
    target = _frame(np.repeat([11, 12], 20), 1)
    for frame in (train, target):
        for column in RICH_NEIGHBOUR_FEATURES:
            frame[column] = np.arange(len(frame), dtype=float)
    ids = [*train["txId"], *target["txId"]]
    graph = build_graph(pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]}))

    plain = make_neighbour_stack(half_life_steps=14, rich=True, two_hop=True)(train, target, columns, 0, graph)
    iterated = make_neighbour_stack(half_life_steps=14, rich=True, two_hop=True, extra_rounds=1)(train, target, columns, 0, graph)
    assert iterated.shape == plain.shape == (len(target),)
    assert np.all((iterated >= 0) & (iterated <= 1))


def test_directional_two_hop_separates_payment_patterns():
    from src.graph_stacking import directional_two_hop_features

    # a -> b -> c (b unlabeled); d -> b ; b -> e.  For a: fwd reaches c and e, cospent reaches d.
    edges = pd.DataFrame({"txId1": ["a", "b", "d", "b"], "txId2": ["b", "c", "b", "e"]})
    frame = pd.DataFrame({"txId": ["a", "c", "d", "e"]})
    scores = pd.Series({"a": 0.1, "c": 0.5, "d": 0.9, "e": 0.7})
    out = directional_two_hop_features(build_graph(edges), frame, scores)

    assert out.loc["a", "dir2_fwd_walks"] == 2
    assert out.loc["a", "dir2_fwd_score_mean"] == pytest.approx(0.6)
    assert out.loc["a", "dir2_cospent_walks"] == 1  # d also spends into b; a itself excluded
    assert out.loc["a", "dir2_cospent_score_mean"] == pytest.approx(0.9)
    assert out.loc["a", "dir2_bwd_walks"] == 0
    assert out.loc["c", "dir2_bwd_walks"] == 2  # a and d are grandparents of c
    assert out.loc["c", "dir2_cofunded_walks"] == 1  # e is c's sibling via b
    assert out.loc["c", "dir2_cofunded_score_mean"] == pytest.approx(0.7)


def test_forward_oof_scores_each_block_from_earlier_steps_only(monkeypatch):
    import src.graph_stacking as gs

    columns = ["feat_1", "feat_2", *gs.RICH_NEIGHBOUR_FEATURES]
    train = _frame(np.repeat(np.arange(1, 11), 30), 0)
    target = _frame(np.repeat([11, 12], 20), 1)
    for frame in (train, target):
        for column in gs.RICH_NEIGHBOUR_FEATURES:
            frame[column] = np.arange(len(frame), dtype=float)
    ids = [*train["txId"], *target["txId"]]
    graph = build_graph(pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]}))

    fitted_steps = []
    real_fit = gs._fit

    def recording_fit(frame, cols, seed, weights=None):
        fitted_steps.append(sorted(frame["time_step"].unique()))
        return real_fit(frame, cols, seed, weights)

    monkeypatch.setattr(gs, "_fit", recording_fit)
    scores = gs.make_neighbour_stack(half_life_steps=14, rich=True, forward_oof=True)(train, target, columns, 0, graph)

    # Blocks of steps 1-2, 3-4, ..., 9-10. Block 1 has no past and keeps the ordinary
    # out-of-fold model; later blocks see only earlier steps. Then a full stage-1 model scores
    # the target rows, and stage 2 is fitted on every training row.
    assert [max(steps) for steps in fitted_steps] == [10, 2, 4, 6, 8, 10, 10]
    assert 1 not in fitted_steps[0] and 2 not in fitted_steps[0]
    assert scores.shape == (len(target),)
    assert np.all((scores >= 0) & (scores <= 1))
