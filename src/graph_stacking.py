"""Graph stacking: label-free neighbourhood context for the illicit-transaction classifier.

Stage 1 scores every labeled training transaction out-of-fold (folds are blocks of time steps)
and every target transaction with the mean of the fold models. Stage 2 adds, for each
transaction, aggregates over its *labeled* neighbours (one hop) and over labeled nodes two hops
away. The only transactions with features in the Kaggle test period (steps 36-49) are labeled
ones, so restricting context to labeled nodes, and letting walks pass through featureless
nodes, computes every graph feature the same way at training and prediction time
(report/findings.md sections 1.3, 11.1, 17.1 and 20.1). No label of a target row is used.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.model_zoo import TUNED_LGBM_PARAMS, lgbm_factory

STACK_FOLDS = 5


def _fit(frame, columns, seed, weights=None):
    model = lgbm_factory(TUNED_LGBM_PARAMS)(seed)
    kwargs = {} if weights is None else {"sample_weight": weights}
    return model.fit(frame[columns], frame["label"].to_numpy(), **kwargs)


def build_graph(edges):
    """Directed edge table in both orientations, tagged with the direction seen from ``u``."""
    edges = edges[["txId1", "txId2"]].drop_duplicates()
    outgoing = edges.rename(columns={"txId1": "u", "txId2": "v"}).assign(direction="out")
    incoming = edges.rename(columns={"txId2": "u", "txId1": "v"}).assign(direction="in")
    return pd.concat([outgoing, incoming], ignore_index=True)


def neighbour_score_features(graph, scores):
    """Aggregate ``scores`` (Series indexed by txId) over each node's scored neighbours."""
    linked = graph.loc[graph["u"].isin(scores.index) & graph["v"].isin(scores.index)].copy()
    linked["score"] = linked["v"].map(scores)
    grouped = linked.groupby("u")["score"]
    features = pd.DataFrame({"nbr_score_mean": grouped.mean(), "nbr_score_max": grouped.max()})
    for direction in ("in", "out"):
        part = linked.loc[linked["direction"] == direction].groupby("u")["score"].mean()
        features[f"nbr_score_mean_{direction}"] = part
    return features.reindex(scores.index)  # NaN where a node has no scored neighbour


RICH_NEIGHBOUR_FEATURES = ["feat_53", "feat_59", "feat_61", "feat_5", "feat_3"]


def _recency_weights(times, reference_step, steps):
    """2^((t - reference)/steps), normalised to mean one; reference = last training step."""
    weights = np.exp2((np.asarray(times, dtype=float) - reference_step) / steps)
    return weights / weights.mean()


def neighbour_context_features(graph, frame, mean_features=RICH_NEIGHBOUR_FEATURES):
    """Structure and labeled-neighbour feature means for the rows of ``frame``.

    Degrees count every edge in the edge list (complete for train and test). Neighbour counts
    and means use only neighbours present in ``frame`` (labeled training rows plus the target
    rows), the same rule as the neighbour scores.
    """
    ids = frame["txId"]
    degree = graph.groupby(["u", "direction"]).size().unstack(fill_value=0).reindex(ids).fillna(0)
    linked = graph.loc[graph["u"].isin(ids) & graph["v"].isin(ids)]
    lookup = frame.set_index("txId")[list(mean_features)]
    values = lookup.loc[linked["v"]].set_axis(linked.index)
    values["u"] = linked["u"]
    means = values.groupby("u")[list(mean_features)].mean().add_prefix("nbr_mean_")
    features = pd.DataFrame(index=ids)
    features["in_degree"] = degree.get("in", 0)
    features["out_degree"] = degree.get("out", 0)
    features["n_labeled_nbrs"] = linked.groupby("u").size().reindex(ids).fillna(0).to_numpy()
    total = features["in_degree"] + features["out_degree"]
    features["frac_labeled_nbrs"] = (features["n_labeled_nbrs"] / total.replace(0, np.nan)).fillna(0)
    return features.join(means)


# Top 25 supplied features by mean |TreeSHAP| on test rows of tuned_lgbm_all (3 seeds, fitted on
# all labeled rows; test features only, no labels). They carry 65% of that model's contribution.
WIDE_NEIGHBOUR_FEATURES = [
    "feat_53", "feat_59", "feat_125", "feat_61", "feat_5", "feat_3", "feat_161", "feat_132", "feat_16",
    "feat_90", "feat_58", "feat_131", "feat_106", "feat_76", "feat_8", "feat_2", "feat_55", "feat_163",
    "feat_65", "feat_138", "feat_47", "feat_23", "feat_67", "feat_52", "feat_136",
]


def neighbour_wide_features(graph, frame, features=WIDE_NEIGHBOUR_FEATURES):
    """Mean, max and std over all labeled neighbours, plus incoming and outgoing means."""
    ids = frame["txId"]
    linked = graph.loc[graph["u"].isin(ids) & graph["v"].isin(ids)]
    values = frame.set_index("txId")[list(features)].loc[linked["v"]].set_axis(linked.index)
    values["u"] = linked["u"].to_numpy()
    values["direction"] = linked["direction"].to_numpy()
    grouped = values.groupby("u")[list(features)]
    parts = [grouped.mean().add_prefix("nbrw_mean_"), grouped.max().add_prefix("nbrw_max_"), grouped.std().add_prefix("nbrw_std_")]
    for direction in ("in", "out"):
        subset = values.loc[values["direction"] == direction]
        parts.append(subset.groupby("u")[list(features)].mean().add_prefix(f"nbrw_{direction}_mean_"))
    return pd.concat(parts, axis=1).reindex(ids)


_ADJACENCY_CACHE = {}


def _undirected_adjacency(graph):
    """Binary undirected adjacency over every node in the edge list (cached per graph)."""
    key = id(graph)
    if key not in _ADJACENCY_CACHE:
        nodes = pd.Index(pd.unique(graph["u"]))
        rows = nodes.get_indexer(graph["u"])
        cols = nodes.get_indexer(graph["v"])
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
        matrix.data[:] = 1.0  # collapse duplicate edges
        _ADJACENCY_CACHE[key] = (nodes, matrix)
    return _ADJACENCY_CACHE[key]


def two_hop_features(graph, frame, scores, mean_features=RICH_NEIGHBOUR_FEATURES):
    """Walk-weighted means over labeled nodes two steps away (paths may pass through any node).

    Uses A(Ax) minus the walk back to the node itself, so a node never counts as its own
    two-hop neighbour. Only nodes in ``frame`` (labeled training rows plus target rows) are
    endpoints, the same rule as the one-hop features.
    """
    nodes, adjacency = _undirected_adjacency(graph)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    position = nodes.get_indexer(frame["txId"])
    present = position >= 0
    mask = np.zeros(len(nodes))
    mask[position[present]] = 1.0

    def walk_sum(values):
        return adjacency @ (adjacency @ values) - degree * values

    counts = walk_sum(mask)
    out = pd.DataFrame(index=frame["txId"])
    out["hop2_labeled_walks"] = np.where(present, counts[np.clip(position, 0, None)], 0.0)
    columns = {"hop2_score_mean": scores.reindex(frame["txId"]).to_numpy()}
    for feature in mean_features:
        columns[f"hop2_mean_{feature}"] = frame[feature].to_numpy()
    for name, values in columns.items():
        vector = np.zeros(len(nodes))
        vector[position[present]] = values[present]
        sums = walk_sum(vector)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / counts, np.nan)
        out[name] = np.where(present, mean[np.clip(position, 0, None)], np.nan)
    return out


_DIRECTED_CACHE = {}


def _directed_adjacency(graph):
    """Binary directed adjacency (row = spending source u, column = spender v), cached per graph."""
    key = id(graph)
    if key not in _DIRECTED_CACHE:
        nodes, _ = _undirected_adjacency(graph)
        out_edges = graph.loc[graph["direction"] == "out"]
        rows, cols = nodes.get_indexer(out_edges["u"]), nodes.get_indexer(out_edges["v"])
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
        matrix.data[:] = 1.0
        _DIRECTED_CACHE[key] = (nodes, matrix)
    return _DIRECTED_CACHE[key]


def directional_two_hop_features(graph, frame, scores):
    """Two-hop stage-1 score means split by the direction of the two payment steps.

    With D the directed adjacency (u -> v when an output of u is spent by v), for node a:
    * ``fwd``  a -> b -> c   descendants two steps on (where the money went)
    * ``bwd``  c -> b -> a   ancestors two steps back (where it came from)
    * ``cospent``  a -> b <- c   co-inputs: outputs spent together in one transaction b
    * ``cofunded`` a <- b -> c   siblings paid by the same transaction b (fan-out)
    Paths may pass through featureless nodes; only nodes in ``frame`` count as endpoints, and a
    node never counts as its own co-input or sibling. Returns a walk-weighted mean score and
    walk count per pattern.
    """
    nodes, directed = _directed_adjacency(graph)
    transpose = directed.T.tocsr()
    out_degree = np.asarray(directed.sum(axis=1)).ravel()
    in_degree = np.asarray(transpose.sum(axis=1)).ravel()
    position = nodes.get_indexer(frame["txId"])
    present = position >= 0
    index = np.clip(position, 0, None)
    mask = np.zeros(len(nodes))
    mask[position[present]] = 1.0
    values = np.zeros(len(nodes))
    values[position[present]] = scores.reindex(frame["txId"]).to_numpy()[present]
    patterns = {
        "fwd": lambda x: directed @ (directed @ x),
        "bwd": lambda x: transpose @ (transpose @ x),
        "cospent": lambda x: directed @ (transpose @ x) - out_degree * x,
        "cofunded": lambda x: transpose @ (directed @ x) - in_degree * x,
    }
    out = pd.DataFrame(index=frame["txId"])
    for name, walk in patterns.items():
        counts = walk(mask)
        sums = walk(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0.5, sums / counts, np.nan)
        out[f"dir2_{name}_walks"] = np.where(present, counts[index], 0.0)
        out[f"dir2_{name}_score_mean"] = np.where(present, mean[index], np.nan)
    return out


def propagation_features(graph, frame, scores, hops=3, decay=0.5):
    """Label-free score propagation over the full graph (featureless nodes pass the walk on).

    With P the row-normalised undirected adjacency and m the indicator of nodes in ``frame``,
    returns sum_k decay^(k-1) P^k (m*s) / sum_k decay^(k-1) P^k m for k = 1..hops: a
    random-walk-weighted mean of stage-1 scores over labeled nodes up to ``hops`` steps away,
    plus the walk mass that reached them (``prop_reach``). Walk returns can include the node's
    own stage-1 score, which is itself label-free.
    """
    nodes, adjacency = _undirected_adjacency(graph)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    transition = sp.diags(inverse) @ adjacency
    position = nodes.get_indexer(frame["txId"])
    present = position >= 0
    mask = np.zeros(len(nodes))
    mask[position[present]] = 1.0
    values = np.zeros(len(nodes))
    values[position[present]] = scores.reindex(frame["txId"]).to_numpy()[present]
    numerator, denominator = np.zeros(len(nodes)), np.zeros(len(nodes))
    walk_values, walk_mask = values, mask
    for hop in range(hops):
        walk_values, walk_mask = transition @ walk_values, transition @ walk_mask
        numerator += decay ** hop * walk_values
        denominator += decay ** hop * walk_mask
    with np.errstate(invalid="ignore", divide="ignore"):
        propagated = np.where(denominator > 0, numerator / denominator, np.nan)
    index = np.clip(position, 0, None)
    return pd.DataFrame(
        {
            "prop_score": np.where(present, propagated[index], np.nan),
            "prop_reach": np.where(present, denominator[index], 0.0),
        },
        index=frame["txId"],
    )


def make_neighbour_stack(half_life_steps=None, rich=False, wide=False, two_hop=False, propagate=False, extra_rounds=0, directional=False):
    """Two-stage stacking over labeled neighbours.

    ``half_life_steps``: recency weights in every stage (relative to the last training step).
    ``rich``: add structure + neighbour feature means, and score target rows with the mean of
    the out-of-fold models so that training and target neighbour scores come from the same
    kind of model (instead of a separate model fitted on all training rows).
    ``wide``: replace the 5 rich neighbour means with mean/max/std/in-mean/out-mean of the 25
    WIDE_NEIGHBOUR_FEATURES over labeled neighbours (requires ``rich``).
    ``two_hop``: add walk-weighted two-hop means of stage-1 scores and the rich features.
    ``propagate``: add 3-hop label-free score propagation (``propagation_features``).
    ``directional``: add two-hop score means split by payment direction
    (``directional_two_hop_features``: forward, backward, co-spent, co-funded).
    ``extra_rounds``: iterative (collective) classification. Before the final stage, repeat
    this many times: score every row out-of-fold with a model that uses the current graph
    context, then rebuild the context from those (more accurate) scores. 0 = plain two-stage.
    """
    def fit(frame, columns, seed, reference_step):
        weights = None if half_life_steps is None else _recency_weights(frame["time_step"], reference_step, half_life_steps)
        return _fit(frame, columns, seed, weights)

    def out_of_fold(train, target, columns, seed, reference, score_target):
        """Out-of-fold scores for ``train`` (folds = blocks of time steps) + fold models' target scores."""
        oof = np.full(len(train), np.nan)
        fold_target_scores = []
        for fold_steps in np.array_split(np.sort(train["time_step"].unique()), STACK_FOLDS):
            held = train["time_step"].isin(fold_steps).to_numpy()
            model = fit(train.loc[~held], columns, seed, reference)
            oof[held] = model.predict_proba(train.loc[held, columns])[:, 1]
            if score_target:
                fold_target_scores.append(model.predict_proba(target[columns])[:, 1])
        return oof, fold_target_scores

    def graph_context(train, target, scores, graph):
        neighbour = neighbour_score_features(graph, scores)
        if rich:
            both = pd.concat([train, target], ignore_index=True)
            context = neighbour_context_features(graph, both, mean_features=[] if wide else RICH_NEIGHBOUR_FEATURES)
            neighbour = neighbour.join(context)
            if wide:
                neighbour = neighbour.join(neighbour_wide_features(graph, both))
            if two_hop:
                neighbour = neighbour.join(two_hop_features(graph, both, scores))
            if propagate:
                neighbour = neighbour.join(propagation_features(graph, both, scores))
            if directional:
                neighbour = neighbour.join(directional_two_hop_features(graph, both, scores))
        return neighbour

    def with_context(train, target, columns, neighbour):
        stacked_columns = [*columns, *neighbour.columns]
        train_x = pd.concat(
            [train[columns], neighbour.loc[train["txId"]].reset_index(drop=True), train[["txId", "time_step", "label"]]], axis=1
        )
        target_x = pd.concat([target[columns], neighbour.loc[target["txId"]].reset_index(drop=True), target[["txId"]]], axis=1)
        return train_x, target_x, stacked_columns

    def fit_predict(train, target, columns, seed, graph):
        train = train.reset_index(drop=True)
        target = target.reset_index(drop=True)
        reference = train["time_step"].max()
        oof, fold_target_scores = out_of_fold(train, target, columns, seed, reference, score_target=rich)
        if rich:
            target_scores = np.mean(fold_target_scores, axis=0)
        else:
            target_scores = fit(train, columns, seed, reference).predict_proba(target[columns])[:, 1]
        scores = pd.concat([pd.Series(oof, index=train["txId"]), pd.Series(target_scores, index=target["txId"])])
        for _ in range(extra_rounds):
            train_x, target_x, stacked_columns = with_context(train, target, columns, graph_context(train, target, scores, graph))
            oof, fold_target_scores = out_of_fold(train_x, target_x, stacked_columns, seed, reference, score_target=True)
            scores = pd.concat([pd.Series(oof, index=train["txId"]), pd.Series(np.mean(fold_target_scores, axis=0), index=target["txId"])])
        train_x, target_x, stacked_columns = with_context(train, target, columns, graph_context(train, target, scores, graph))
        model = fit(train_x, stacked_columns, seed, reference)
        return model.predict_proba(target_x[stacked_columns])[:, 1]

    return fit_predict


neighbour_stack = make_neighbour_stack()


