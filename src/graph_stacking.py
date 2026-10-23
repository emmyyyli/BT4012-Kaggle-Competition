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
import scipy.sparse.csgraph  # noqa: F401  (registers sp.csgraph)

from src.model_zoo import TUNED_LGBM_PARAMS, lgbm_factory

STACK_FOLDS = 5


def _fit(frame, columns, seed, weights=None, params=None):
    model = lgbm_factory(TUNED_LGBM_PARAMS if params is None else params)(seed)
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


_TWO_HOP_SET_CACHE = {}


def _two_hop_sets(graph):
    """Binary matrix of distinct nodes reachable by an undirected two-step walk (self excluded)."""
    key = id(graph)
    if key not in _TWO_HOP_SET_CACHE:
        _, adjacency = _undirected_adjacency(graph)
        reach = (adjacency @ adjacency).tocsr()
        reach.setdiag(0)
        reach.eliminate_zeros()
        reach.data[:] = 1.0
        _TWO_HOP_SET_CACHE[key] = reach
    return _TWO_HOP_SET_CACHE[key]


def two_hop_max_features(graph, frame, scores):
    """Highest stage-1 score among labeled transactions exactly two steps away.

    The two-hop means average over every labeled node two steps away, so one suspicious
    transaction among many licit ones is diluted. The max asks whether *any* labeled
    transaction within two steps looks illicit. Walks may pass through featureless nodes; only
    nodes in ``frame`` count as endpoints, and a node is never its own two-hop endpoint. Also
    returns the number of distinct labeled two-hop endpoints and how many score above 0.5.
    """
    nodes, _ = _undirected_adjacency(graph)
    reach = _two_hop_sets(graph)
    position = nodes.get_indexer(frame["txId"])
    present = position >= 0
    labeled = np.zeros(len(nodes), dtype=bool)
    labeled[position[present]] = True
    values = np.full(len(nodes), -np.inf)
    values[position[present]] = scores.reindex(frame["txId"]).to_numpy()[present]
    restricted = (reach @ sp.diags(labeled.astype(float))).tocsr()
    restricted.eliminate_zeros()
    counts = np.diff(restricted.indptr)
    endpoint_scores = values[restricted.indices]
    best = np.full(len(nodes), np.nan)
    has = counts > 0
    starts = restricted.indptr[:-1][has]
    best[has] = np.maximum.reduceat(endpoint_scores, starts)
    high = np.add.reduceat((endpoint_scores > 0.5).astype(float), starts) if has.any() else np.array([])
    n_high = np.zeros(len(nodes))
    n_high[has] = high
    index = np.clip(position, 0, None)
    return pd.DataFrame(
        {
            "hop2_score_max": np.where(present, best[index], np.nan),
            "hop2_distinct_labeled": np.where(present, counts[index], 0).astype(float),
            "hop2_n_high_score": np.where(present, n_high[index], 0.0),
        },
        index=frame["txId"],
    )


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


def hub_weighted_features(graph, frame, scores, wide_features=WIDE_NEIGHBOUR_FEATURES, hop2_features=RICH_NEIGHBOUR_FEATURES):
    """Neighbour means in which high-degree (hub) transactions count less.

    One hop: mean over labeled neighbours v weighted by 1/deg(v), of the stage-1 score and the
    25 wide features. Two hops: paths a - b - c weighted by 1/deg(b), so a hub in the middle
    (e.g. a sweep transaction with hundreds of links) no longer contributes hundreds of
    endpoints; the walk back to the node itself is excluded. Degrees count every edge in the
    edge list, including edges to featureless nodes.
    """
    nodes, adjacency = _undirected_adjacency(graph)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    position = nodes.get_indexer(frame["txId"])
    present = position >= 0
    index = np.clip(position, 0, None)
    mask = np.zeros(len(nodes))
    mask[position[present]] = 1.0
    self_return = adjacency @ inverse  # sum over b of 1/deg(b): weight of the paths a - b - a

    def one_hop(x):
        return adjacency @ (inverse * x)

    def two_hop(x):
        return adjacency @ (inverse * (adjacency @ x)) - self_return * x

    columns = {"score": scores.reindex(frame["txId"]).to_numpy()}
    for feature in dict.fromkeys([*wide_features, *hop2_features]):
        columns[feature] = frame[feature].to_numpy()
    one_hop_names = ["score", *wide_features]
    two_hop_names = ["score", *hop2_features]
    counts = {"hub1": one_hop(mask), "hub2": two_hop(mask)}
    out = {}
    for prefix, walk, names in (("hub1", one_hop, one_hop_names), ("hub2", two_hop, two_hop_names)):
        for name in names:
            vector = np.zeros(len(nodes))
            vector[position[present]] = columns[name][present]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(counts[prefix] > 1e-12, walk(vector) / counts[prefix], np.nan)
            out[f"{prefix}_mean_{name}"] = np.where(present, mean[index], np.nan)
    return pd.DataFrame(out, index=frame["txId"])


_STRUCTURE_CACHE = {}


def _components(graph):
    nodes, adjacency = _undirected_adjacency(graph)
    _, labels = sp.csgraph.connected_components(adjacency, directed=False)
    return labels


def _longest_paths(directed):
    """Longest directed path ending at (``up``) and starting from (``down``) each node of a DAG."""
    n = directed.shape[0]
    transpose = directed.T.tocsr()
    in_degree = np.diff(transpose.indptr).copy()
    order, frontier = [], list(np.flatnonzero(in_degree == 0))
    while frontier:  # Kahn's algorithm; the payment graph has no cycles
        node = frontier.pop()
        order.append(node)
        for child in directed.indices[directed.indptr[node]:directed.indptr[node + 1]]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                frontier.append(child)
    if len(order) != n:
        raise ValueError("graph has a cycle")
    up, down = np.zeros(n), np.zeros(n)
    for node in order:
        parents = transpose.indices[transpose.indptr[node]:transpose.indptr[node + 1]]
        if len(parents):
            up[node] = up[parents].max() + 1
    for node in reversed(order):
        children = directed.indices[directed.indptr[node]:directed.indptr[node + 1]]
        if len(children):
            down[node] = down[children].max() + 1
    return np.asarray(order), up, down


def chain_features(graph, ids):
    """Label-free payment-chain shape for each id, from the full edge list.

    * ``depth_up`` / ``depth_down``: longest directed path into / out of the transaction;
    * ``peel_like``: one funding transaction and at most two spenders (pay one party, keep the
      change), the building block of a peeling chain;
    * ``peel_run_up`` / ``peel_run_down``: length of the unbroken run of peel-like transactions
      leading into / out of this one (0 when the transaction itself is not peel-like).
    Every node of the edge list takes part (featureless ones included), and the edge list is
    complete in both periods, so the features mean the same at training and prediction time.
    Component sizes are deliberately not used: each time step is exactly one component, so
    they would encode the time step.
    """
    key = ("chain", id(graph))
    if key not in _STRUCTURE_CACHE:
        nodes, directed = _directed_adjacency(graph)
        out_degree = np.diff(directed.indptr)
        in_degree = np.diff(directed.T.tocsr().indptr)
        order, up, down = _longest_paths(directed)
        transpose = directed.T.tocsr()
        peel = (in_degree == 1) & (out_degree <= 2)
        run_up, run_down = np.zeros(len(nodes)), np.zeros(len(nodes))
        for node in order:
            if peel[node]:
                parents = transpose.indices[transpose.indptr[node]:transpose.indptr[node + 1]]
                run_up[node] = 1 + (run_up[parents].max() if len(parents) else 0)
        for node in order[::-1]:
            if peel[node]:
                children = directed.indices[directed.indptr[node]:directed.indptr[node + 1]]
                run_down[node] = 1 + (run_down[children].max() if len(children) else 0)
        _STRUCTURE_CACHE[key] = pd.DataFrame(
            {"depth_up": up, "depth_down": down, "peel_like": peel.astype(float), "peel_run_up": run_up, "peel_run_down": run_down},
            index=nodes,
        )
    return _STRUCTURE_CACHE[key].reindex(pd.Index(ids))


def _pagerank(matrix, damping=0.85, iterations=100, tol=1e-10):
    """PageRank over ``matrix`` (row = source); dangling mass is spread uniformly."""
    n = matrix.shape[0]
    out_degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0)
    transition_t = (sp.diags(inverse) @ matrix).T.tocsr()
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        dangling = rank[out_degree == 0].sum()
        new = damping * (transition_t @ rank + dangling / n) + (1 - damping) / n
        if np.abs(new - rank).sum() < tol:
            return new
        rank = new
    return rank


def centrality_features(graph, ids):
    """Label-free centrality for each id, from the full edge list.

    * ``pagerank_fwd``: PageRank along the payment direction (where funds accumulate);
      ``pagerank_bwd``: against it (where funds originate). Each is rescaled to mean one within
      its connected component, because each time step is one component and raw PageRank
      shrinks with component size (it would otherwise encode the time step);
    * ``reach2``: distinct transactions within two hops (undirected);
    * ``nbr_degree_mean`` / ``nbr_degree_max``: degree of direct neighbours, featureless included;
    * ``triangles``: number of triangles through the node (undirected), e.g. a -> b -> c with a -> c.
    """
    key = ("centrality", id(graph))
    if key not in _STRUCTURE_CACHE:
        nodes, adjacency = _undirected_adjacency(graph)
        _, directed = _directed_adjacency(graph)
        labels = _components(graph)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        features = {}
        for name, matrix in (("pagerank_fwd", directed), ("pagerank_bwd", directed.T.tocsr())):
            rank = _pagerank(matrix)
            component_mean = np.bincount(labels, weights=rank) / np.bincount(labels)
            features[name] = rank / component_mean[labels]
        two_step = adjacency @ adjacency
        reach = ((two_step + adjacency) > 0).astype(float)
        features["reach2"] = np.asarray(reach.sum(axis=1)).ravel() - (reach.diagonal() > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            features["nbr_degree_mean"] = np.where(degree > 0, (adjacency @ degree) / degree, np.nan)
        features["nbr_degree_max"] = np.asarray(adjacency.multiply(degree[None, :]).max(axis=1).todense()).ravel()
        features["triangles"] = np.asarray(two_step.multiply(adjacency).sum(axis=1)).ravel() / 2
        _STRUCTURE_CACHE[key] = pd.DataFrame(features, index=nodes)
    return _STRUCTURE_CACHE[key].reindex(pd.Index(ids))


_SHIFT_CACHE = {}


def covariate_shift_weights(train, target, columns, drop_rounds=3, drop_per_round=10, power=0.5, clip=(0.2, 5.0)):
    """Importance weights p(target | x) / p(train | x) for the rows of ``train`` (Series by txId).

    An adversarial LightGBM separates training rows from target rows (5-fold out-of-fold, no
    labels). With all supplied features the two periods separate perfectly through a few
    time-trend features, which would make every weight ~0 and only repeat the recency weights.
    So ``drop_rounds`` times the ``drop_per_round`` features with the largest adversarial gain
    are removed first (report/findings.md section 37: AUC 1.000 -> 0.94 after 30 features,
    weight-vs-step correlation 0.62 -> 0.37). The remaining odds are tempered by ``power``,
    normalised to mean one, clipped to ``clip`` and renormalised. The removed features are
    dropped only for the weights; the fraud classifier keeps all of them.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    key = (tuple(train["txId"]), tuple(target["txId"]), tuple(columns), drop_rounds, drop_per_round, power, clip)
    if key in _SHIFT_CACHE:
        return _SHIFT_CACHE[key]
    features = pd.concat([train[list(columns)], target[list(columns)]], ignore_index=True)
    is_target = np.r_[np.zeros(len(train)), np.ones(len(target))]
    kept = list(columns)
    for round_ in range(drop_rounds + 1):
        oof, gain = np.zeros(len(features)), np.zeros(len(kept))
        for fit_rows, held_rows in StratifiedKFold(5, shuffle=True, random_state=0).split(features, is_target):
            model = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=15, learning_rate=0.05, colsample_bytree=0.5, min_child_samples=100,
                random_state=0, verbose=-1,
            ).fit(features.iloc[fit_rows][kept], is_target[fit_rows])
            oof[held_rows] = model.predict_proba(features.iloc[held_rows][kept])[:, 1]
            gain += model.booster_.feature_importance("gain")
        if round_ < drop_rounds:
            dropped = set(pd.Series(gain, index=kept).nlargest(min(drop_per_round, len(kept) - 1)).index)
            kept = [column for column in kept if column not in dropped]
    probability = np.clip(oof[: len(train)], 1e-6, 1 - 1e-6)
    weights = (probability / (1 - probability)) ** power
    weights = np.clip(weights / weights.mean(), *clip)
    result = pd.Series(weights / weights.mean(), index=train["txId"].to_numpy())
    _SHIFT_CACHE[key] = result
    return result


def make_neighbour_stack(
    half_life_steps=None, rich=False, wide=False, two_hop=False, propagate=False, extra_rounds=0, directional=False, forward_oof=False,
    hub=False, chain=False, centrality=False, shift_weights=False, two_hop_wide=False, two_hop_max=False,
    stage2_match_features=False, stage2_overrides=None,
):
    """Two-stage stacking over labeled neighbours.

    ``half_life_steps``: recency weights in every stage (relative to the last training step).
    ``rich``: add structure + neighbour feature means, and score target rows with the mean of
    the out-of-fold models so that training and target neighbour scores come from the same
    kind of model (instead of a separate model fitted on all training rows).
    ``wide``: replace the 5 rich neighbour means with mean/max/std/in-mean/out-mean of the 25
    WIDE_NEIGHBOUR_FEATURES over labeled neighbours (requires ``rich``).
    ``two_hop``: add walk-weighted two-hop means of stage-1 scores and the rich features.
    ``two_hop_wide``: take the two-hop means over all 25 WIDE_NEIGHBOUR_FEATURES instead of
    the 5 rich ones (requires ``two_hop``).
    ``two_hop_max``: add the highest stage-1 score among labeled two-hop endpoints
    (``two_hop_max_features``).
    ``stage2_match_features``: in the final stage only, scale ``colsample_bytree`` by
    (supplied features / stage-2 features), so each stage-2 tree samples as many columns as a
    stage-1 tree (about 106) instead of twice as many from the overlapping graph statistics.
    ``stage2_overrides``: dict of LightGBM settings applied to the final stage only, after
    ``stage2_match_features`` (e.g. ``{"reg_lambda": 8.0}``).
    ``propagate``: add 3-hop label-free score propagation (``propagation_features``).
    ``directional``: add two-hop score means split by payment direction
    (``directional_two_hop_features``: forward, backward, co-spent, co-funded).
    ``extra_rounds``: iterative (collective) classification. Before the final stage, repeat
    this many times: score every row out-of-fold with a model that uses the current graph
    context, then rebuild the context from those (more accurate) scores. 0 = plain two-stage.
    ``forward_oof``: score each time block of training rows with a model fitted only on earlier
    steps (the first block, which has no past, keeps its ordinary out-of-fold scores), and score
    target rows with a model fitted on all training rows. Ordinary out-of-fold models also see
    later steps, so training-row neighbour scores are far more accurate than the forward-only
    scores target rows get (AUC 0.997 vs 0.985 on steps 8-35, report/findings.md section 32);
    this option gives stage 2 neighbour scores of the quality it will meet at prediction time.
    ``hub``: add neighbour means down-weighted by degree (``hub_weighted_features``).
    ``chain``: add label-free payment-chain shape (``chain_features``).
    ``centrality``: add label-free PageRank, two-hop reach and neighbour degree
    (``centrality_features``).
    ``shift_weights``: multiply every fit's sample weights by covariate-shift importance weights
    (``covariate_shift_weights``: how test-like each training row looks), computed once per
    train/target pair.
    """
    state = {"shift": None}

    def fit(frame, columns, seed, reference_step, params=None):
        weights = None if half_life_steps is None else _recency_weights(frame["time_step"], reference_step, half_life_steps)
        if state["shift"] is not None:
            shift = state["shift"].reindex(frame["txId"]).to_numpy()
            weights = shift if weights is None else np.asarray(weights) * shift
            weights = weights / weights.mean()
        return _fit(frame, columns, seed, weights, params)

    def out_of_fold(train, target, columns, seed, reference, score_target):
        """Out-of-fold scores for ``train`` (folds = blocks of time steps) + fold models' target scores."""
        oof = np.full(len(train), np.nan)
        fold_target_scores = []
        for block, fold_steps in enumerate(np.array_split(np.sort(train["time_step"].unique()), STACK_FOLDS)):
            held = train["time_step"].isin(fold_steps).to_numpy()
            fit_rows = (train["time_step"] < fold_steps[0]).to_numpy() if forward_oof and block > 0 else ~held
            model = fit(train.loc[fit_rows], columns, seed, reference)
            oof[held] = model.predict_proba(train.loc[held, columns])[:, 1]
            if score_target and not forward_oof:
                fold_target_scores.append(model.predict_proba(target[columns])[:, 1])
        if score_target and forward_oof:
            fold_target_scores.append(fit(train, columns, seed, reference).predict_proba(target[columns])[:, 1])
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
                hop2_features = WIDE_NEIGHBOUR_FEATURES if two_hop_wide else RICH_NEIGHBOUR_FEATURES
                neighbour = neighbour.join(two_hop_features(graph, both, scores, mean_features=hop2_features))
            if two_hop_max:
                neighbour = neighbour.join(two_hop_max_features(graph, both, scores))
            if propagate:
                neighbour = neighbour.join(propagation_features(graph, both, scores))
            if directional:
                neighbour = neighbour.join(directional_two_hop_features(graph, both, scores))
            if hub:
                neighbour = neighbour.join(hub_weighted_features(graph, both, scores))
            if chain:
                neighbour = neighbour.join(chain_features(graph, both["txId"]).set_axis(both["txId"]))
            if centrality:
                neighbour = neighbour.join(centrality_features(graph, both["txId"]).set_axis(both["txId"]))
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
        state["shift"] = covariate_shift_weights(train, target, columns) if shift_weights else None
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
        stage2_params = None
        if stage2_match_features or stage2_overrides:
            stage2_params = dict(TUNED_LGBM_PARAMS)
            if stage2_match_features:
                stage2_params["colsample_bytree"] *= len(columns) / len(stacked_columns)
            stage2_params.update(stage2_overrides or {})
        model = fit(train_x, stacked_columns, seed, reference, stage2_params)
        return model.predict_proba(target_x[stacked_columns])[:, 1]

    return fit_predict


neighbour_stack = make_neighbour_stack()


