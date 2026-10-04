import numpy as np
import pandas as pd
import pytest

from src.graph import audit_graph_coverage, build_induced_graph_data, build_union_graph_data


@pytest.fixture
def sample_graph_data():
    train = pd.DataFrame(
        {
            "txId": ["t1", "t2"],
            "time_step": [1, 1],
            "feat_1": [0.5, 0.6],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "txId": ["t3"],
            "time_step": [1],
            "feat_1": [0.7],
        }
    )
    # t4 is in edges, but NOT in train or test (missing feature node)
    # t1 -> t2 (both in feat)
    # t2 -> t4 (t4 missing)
    # t4 -> t3 (t4 missing)
    edges = pd.DataFrame(
        {
            "txId1": ["t1", "t2", "t4"],
            "txId2": ["t2", "t4", "t3"],
        }
    )
    return train, test, edges


def test_audit_graph_coverage(sample_graph_data):
    train, test, edges = sample_graph_data
    coverage = audit_graph_coverage(train, test, edges)

    assert coverage["num_train_nodes"] == 2
    assert coverage["num_test_nodes"] == 1
    assert coverage["num_feature_nodes"] == 3
    assert coverage["num_edge_endpoints"] == 4
    assert coverage["num_union_nodes"] == 4
    assert coverage["num_missing_feature_nodes"] == 1
    assert coverage["num_total_edges"] == 3
    assert coverage["num_edges_induced"] == 1
    assert coverage["num_edges_lost_induced"] == 2
    assert coverage["pct_edges_lost_induced"] == pytest.approx(66.6666, 0.01)


def test_build_union_graph_data(sample_graph_data):
    train, test, edges = sample_graph_data
    graph_data = build_union_graph_data(train, test, edges, fill_value=0.0)

    node_df = graph_data["node_df"]
    features = graph_data["features"]
    edge_index = graph_data["edge_index"]
    sparse_adj = graph_data["sparse_adj"]

    assert len(node_df) == 4
    assert set(node_df["txId"]) == {"t1", "t2", "t3", "t4"}
    
    # Missing node t4 should have time_step=1 established via connected component
    t4_row = node_df[node_df["txId"] == "t4"].iloc[0]
    assert t4_row["time_step"] == 1
    assert t4_row["is_missing_features"] == True
    
    # Features matrix: 4 nodes, 1 feature + 1 missing indicator column
    assert features.shape == (4, 2)
    # Check missing indicator column is 1 for t4 and 0 for others
    t4_idx = t4_row["node_idx"]
    assert features[t4_idx, 0] == 0.0  # filled
    assert features[t4_idx, 1] == 1.0  # missing indicator
    
    other_indices = [idx for idx in range(4) if idx != t4_idx]
    assert (features[other_indices, 1] == 0.0).all()

    # Edge index & sparse adj
    assert edge_index.shape == (2, 3)
    assert sparse_adj.nnz == 3
    assert sparse_adj.shape == (4, 4)


def test_build_induced_graph_data(sample_graph_data):
    train, test, edges = sample_graph_data
    graph_data = build_induced_graph_data(train, test, edges)

    node_df = graph_data["node_df"]
    features = graph_data["features"]
    edge_index = graph_data["edge_index"]
    sparse_adj = graph_data["sparse_adj"]

    assert len(node_df) == 3
    assert "t4" not in node_df["txId"].values
    assert features.shape == (3, 1)
    assert edge_index.shape == (2, 1)  # Only t1 -> t2 retained
    assert sparse_adj.nnz == 1
