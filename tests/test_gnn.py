import numpy as np
import pandas as pd
import pytest
import torch

from src.gnn import GraphSAGENet, compute_normalized_adjacency, train_graphsage
from src.gt_ssl_extractor import (
    GTSSLConfig,
    GTSSLPretrainer,
    _bootstrap_auc_delta,
    _contrastive_loss,
    build_walk_sequences,
)
from src.graph import build_union_graph_data


@pytest.fixture
def sample_graph_for_gnn():
    train = pd.DataFrame(
        {
            "txId": [f"t{i}" for i in range(1, 7)],
            "time_step": [1, 1, 1, 2, 2, 2],
            "feat_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feat_2": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "txId": ["t7", "t8"],
            "time_step": [3, 3],
            "feat_1": [0.7, 0.8],
            "feat_2": [0.3, 0.2],
        }
    )
    edges = pd.DataFrame(
        {
            "txId1": ["t1", "t2", "t4", "t5", "t7"],
            "txId2": ["t2", "t3", "t5", "t6", "t8"],
        }
    )
    return train, test, edges


def test_compute_normalized_adjacency(sample_graph_for_gnn):
    train, test, edges = sample_graph_for_gnn
    graph_data = build_union_graph_data(train, test, edges)
    sparse_adj = graph_data["sparse_adj"]

    norm_adj = compute_normalized_adjacency(sparse_adj)
    assert norm_adj.is_sparse
    assert norm_adj.shape == sparse_adj.shape

    dense = norm_adj.to_dense().numpy()
    row_sums = dense.sum(axis=1)
    # Each row with self-loop should sum to 1.0
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_graphsage_forward_pass():
    n_nodes = 10
    in_dim = 5
    hidden_dim = 16

    x = torch.randn(n_nodes, in_dim)
    # Simple identity adjacency
    indices = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)])
    values = torch.ones(n_nodes)
    adj = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))

    model = GraphSAGENet(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=2, dropout=0.0)
    logits, embeddings = model(x, adj)

    assert logits.shape == (n_nodes, 2)
    assert embeddings.shape == (n_nodes, hidden_dim)


def test_train_graphsage_pipeline(sample_graph_for_gnn):
    train, test, edges = sample_graph_for_gnn
    graph_data = build_union_graph_data(train, test, edges)

    model, prob_series, eval_metrics = train_graphsage(
        graph_data,
        train_time_end=1,
        valid_time_end=2,
        hidden_dim=16,
        epochs=10,
        lr=0.05,
    )

    assert len(prob_series) == len(graph_data["node_df"])
    assert (prob_series >= 0.0).all() and (prob_series <= 1.0).all()
    assert eval_metrics["train_rows"] == 3  # t1, t2, t3
    assert eval_metrics["valid_rows"] == 3  # t4, t5, t6
    assert 0.0 <= eval_metrics["roc_auc"] <= 1.0


def test_gt_ssl_walks_are_fixed_length_and_seeded():
    config = GTSSLConfig(hidden_dim=16, attention_heads=4, encoder_layers=1)
    edge_index = np.array([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=np.int32)
    features = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]],
        dtype=np.float32,
    )

    sequences_a, edges_a = build_walk_sequences(
        edge_index, features, np.arange(4, dtype=np.int32), 4, config
    )
    sequences_b, edges_b = build_walk_sequences(
        edge_index, features, np.arange(4, dtype=np.int32), 4, config
    )

    assert sequences_a.shape == (4, 8)
    assert edges_a.shape == (4, 7)
    assert np.array_equal(sequences_a[:, 0], np.arange(4))
    assert np.array_equal(sequences_a, sequences_b)
    assert np.array_equal(edges_a, edges_b)


def test_gt_ssl_encoder_and_contrastive_loss_backpropagate():
    config = GTSSLConfig(hidden_dim=16, attention_heads=4, encoder_layers=1)
    model = GTSSLPretrainer(input_dim=4, reconstruction_dim=3, config=config)
    features = torch.randn(4, 8, 4)
    features[..., -1] = 0.0
    edge_keep = torch.ones(4, 7, dtype=torch.bool)

    reconstructed, first_view, hidden = model(features, edge_keep)
    second_view = model(features + 0.01, edge_keep)[1]
    loss = reconstructed.square().mean() + _contrastive_loss(
        first_view, second_view, config.contrastive_temperature
    )
    loss.backward()

    assert reconstructed.shape == (4, 8, 3)
    assert hidden.shape == (4, 8, config.hidden_dim)
    assert torch.isfinite(loss)
    assert model.input_projection.weight.grad is not None


def test_gt_ssl_auc_delta_bootstrap_is_zero_for_identical_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    low, high = _bootstrap_auc_delta(
        labels, probabilities, probabilities, random_state=7, iterations=20
    )

    assert low == 0.0
    assert high == 0.0
