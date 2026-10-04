import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data import get_feature_columns


def audit_graph_coverage(train, test, edges):
    """Compute exact node and edge coverage statistics comparing Option A (induced) vs Option B (union)."""
    feat_train_ids = set(train["txId"])
    feat_test_ids = set(test["txId"])
    feat_all_ids = feat_train_ids | feat_test_ids

    dedup_edges = edges[["txId1", "txId2"]].drop_duplicates()
    edge_all_ids = set(dedup_edges["txId1"]) | set(dedup_edges["txId2"])

    union_all_ids = feat_all_ids | edge_all_ids
    missing_feat_ids = edge_all_ids - feat_all_ids
    orphan_feat_ids = feat_all_ids - edge_all_ids

    src_in_feat = dedup_edges["txId1"].isin(feat_all_ids)
    dst_in_feat = dedup_edges["txId2"].isin(feat_all_ids)

    both_in_feat = src_in_feat & dst_in_feat
    touching_missing = ~both_in_feat

    return {
        "num_train_nodes": len(feat_train_ids),
        "num_test_nodes": len(feat_test_ids),
        "num_feature_nodes": len(feat_all_ids),
        "num_edge_endpoints": len(edge_all_ids),
        "num_union_nodes": len(union_all_ids),
        "num_missing_feature_nodes": len(missing_feat_ids),
        "num_orphan_feature_nodes": len(orphan_feat_ids),
        "num_total_edges": len(dedup_edges),
        "num_edges_induced": int(both_in_feat.sum()),
        "num_edges_lost_induced": int(touching_missing.sum()),
        "pct_edges_lost_induced": float(touching_missing.sum() / len(dedup_edges) * 100),
    }


def build_union_graph_data(train, test, edges, fill_value=0.0):
    """
    Construct memory-efficient sparse union graph representation without dense matrices.
    
    Includes all 203,769 nodes and 234,355 edges:
    - Feature nodes carry original 165 features with is_missing=0.
    - Missing-feature nodes carry fill_value with is_missing=1.
    - Time-steps for missing nodes are established via connected component membership.
    """
    feature_cols = get_feature_columns(train.columns)
    dedup_edges = edges[["txId1", "txId2"]].drop_duplicates()

    # 1. Gather all unique node IDs
    feat_train_ids = set(train["txId"])
    feat_test_ids = set(test["txId"])
    feat_ids = set(train["txId"]).union(test["txId"])
    edge_ids = set(dedup_edges["txId1"]).union(dedup_edges["txId2"])
    all_node_ids = sorted(list(feat_ids.union(edge_ids)))
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    num_nodes = len(all_node_ids)

    # 2. Known metadata & time_step lookup
    time_map = pd.concat(
        [train.set_index("txId")["time_step"], test.set_index("txId")["time_step"]]
    ).to_dict()

    # Establish time step for missing nodes via connected component membership
    # In Elliptic data, components are strictly intra-timestep
    adj_for_comp = {}
    for src, dst in zip(dedup_edges["txId1"], dedup_edges["txId2"]):
        adj_for_comp.setdefault(src, []).append(dst)
        adj_for_comp.setdefault(dst, []).append(src)

    # Propagate time_step to any connected node with missing time
    visited = set()
    for start_node in all_node_ids:
        if start_node in visited:
            continue
        # BFS / component exploration
        component = []
        queue = [start_node]
        visited.add(start_node)
        comp_time = None
        while queue:
            curr = queue.pop()
            component.append(curr)
            if comp_time is None and curr in time_map:
                comp_time = time_map[curr]
            for neighbor in adj_for_comp.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if comp_time is not None:
            for node in component:
                time_map.setdefault(node, comp_time)

    # 3. Build Node Metadata DataFrame
    labels_map = train.set_index("txId")["label"].to_dict()
    node_df = pd.DataFrame(
        {
            "node_idx": np.arange(num_nodes, dtype=np.int32),
            "txId": all_node_ids,
            "time_step": [time_map.get(nid, np.nan) for nid in all_node_ids],
            "label": [labels_map.get(nid, np.nan) for nid in all_node_ids],
            "is_train": [nid in feat_train_ids for nid in all_node_ids],
            "is_test": [nid in feat_test_ids for nid in all_node_ids],
            "is_missing_features": [nid not in feat_ids for nid in all_node_ids],
        }
    )
    node_df["time_step"] = pd.to_numeric(node_df["time_step"], errors="coerce").astype("Int16")
    node_df["label"] = pd.to_numeric(node_df["label"], errors="coerce").astype("Int8")

    # 4. Feature Matrix (N x 166) with missing indicator
    X = np.full((num_nodes, len(feature_cols) + 1), fill_value, dtype=np.float32)

    # Set missing indicator column (last column) to 1 for missing nodes, 0 for known nodes
    is_missing = node_df["is_missing_features"].to_numpy()
    X[:, -1] = is_missing.astype(np.float32)

    # Populate known features
    combined_feats = pd.concat(
        [train[["txId", *feature_cols]], test[["txId", *feature_cols]]],
        ignore_index=True,
    )
    known_indices = combined_feats["txId"].map(node_to_idx).to_numpy()
    X[known_indices, : len(feature_cols)] = combined_feats[feature_cols].to_numpy(dtype=np.float32)

    # 5. Sparse Edge Indices & SciPy Matrix
    src_idx = dedup_edges["txId1"].map(node_to_idx).to_numpy(dtype=np.int32)
    dst_idx = dedup_edges["txId2"].map(node_to_idx).to_numpy(dtype=np.int32)
    edge_index = np.stack([src_idx, dst_idx], axis=0)

    sparse_adj = sp.csr_matrix(
        (np.ones(len(src_idx), dtype=np.float32), (src_idx, dst_idx)),
        shape=(num_nodes, num_nodes),
    )

    feature_names = [*feature_cols, "is_missing_features"]
    return {
        "node_df": node_df,
        "features": X,
        "feature_names": feature_names,
        "edge_index": edge_index,
        "sparse_adj": sparse_adj,
    }


def build_induced_graph_data(train, test, edges):
    """
    Construct Option A (induced graph): Drops all missing-feature nodes and their edges.
    """
    feature_cols = get_feature_columns(train.columns)
    feat_train_ids = set(train["txId"])
    feat_test_ids = set(test["txId"])
    feat_ids = sorted(list(feat_train_ids.union(feat_test_ids)))
    node_to_idx = {nid: i for i, nid in enumerate(feat_ids)}
    num_nodes = len(feat_ids)

    dedup_edges = edges[["txId1", "txId2"]].drop_duplicates()
    valid_mask = dedup_edges["txId1"].isin(node_to_idx) & dedup_edges["txId2"].isin(node_to_idx)
    filtered_edges = dedup_edges.loc[valid_mask]

    # Combine known features
    combined_feats = pd.concat(
        [train[["txId", "time_step", *feature_cols]], test[["txId", "time_step", *feature_cols]]],
        ignore_index=True,
    )
    labels_map = train.set_index("txId")["label"].to_dict()

    node_df = pd.DataFrame(
        {
            "node_idx": np.arange(num_nodes, dtype=np.int32),
            "txId": feat_ids,
            "time_step": combined_feats.set_index("txId").loc[feat_ids, "time_step"].values,
            "label": [labels_map.get(nid, np.nan) for nid in feat_ids],
            "is_train": [nid in feat_train_ids for nid in feat_ids],
            "is_test": [nid in feat_test_ids for nid in feat_ids],
            "is_missing_features": False,
        }
    )

    src_idx = filtered_edges["txId1"].map(node_to_idx).to_numpy(dtype=np.int32)
    dst_idx = filtered_edges["txId2"].map(node_to_idx).to_numpy(dtype=np.int32)
    edge_index = np.stack([src_idx, dst_idx], axis=0)

    sparse_adj = sp.csr_matrix(
        (np.ones(len(src_idx), dtype=np.float32), (src_idx, dst_idx)),
        shape=(num_nodes, num_nodes),
    )

    X = combined_feats.set_index("txId").loc[feat_ids, feature_cols].to_numpy(dtype=np.float32)

    return {
        "node_df": node_df,
        "features": X,
        "feature_names": feature_cols,
        "edge_index": edge_index,
        "sparse_adj": sparse_adj,
    }
