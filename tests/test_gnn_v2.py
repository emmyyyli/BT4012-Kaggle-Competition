import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.gnn_v2 import mask_unlabeled_features, standardize, train_and_predict


def _tiny_graph():
    node_df = pd.DataFrame(
        {
            "txId": list("abcdef"),
            "time_step": pd.array([1, 1, 1, 2, 2, 2], dtype="Int16"),
            "label": pd.array([0, 1, pd.NA, 0, pd.NA, pd.NA], dtype="Int8"),
            "is_test": [False, False, False, False, False, True],
        }
    )
    features = np.array(
        [[0.1, 1.0, 0], [2.0, 3.0, 0], [5.0, 5.0, 0], [0.2, 0.9, 0], [0.0, 0.0, 1], [1.5, 2.5, 0]],
        dtype=np.float32,
    )
    adjacency = sp.csr_matrix(([1, 1, 1, 1], ([0, 1, 3, 4], [2, 2, 5, 5])), shape=(6, 6), dtype=np.float32)
    return {"node_df": node_df, "features": features, "sparse_adj": adjacency}


def test_masking_blanks_unlabeled_nodes_but_keeps_labeled_and_test_nodes():
    graph = _tiny_graph()
    masked = mask_unlabeled_features(graph["features"], graph["node_df"])

    assert masked[2].tolist() == [0.0, 0.0, 1.0]  # unlabeled training-step node
    assert masked[4].tolist() == [0.0, 0.0, 1.0]  # already missing
    np.testing.assert_array_equal(masked[[0, 1, 3, 5]], graph["features"][[0, 1, 3, 5]])


def test_standardize_uses_only_fit_rows_and_leaves_blank_nodes_zero():
    graph = _tiny_graph()
    masked = mask_unlabeled_features(graph["features"], graph["node_df"])
    fit_rows = np.array([True, True, False, False, False, False])
    scaled = standardize(masked, fit_rows)

    np.testing.assert_allclose(scaled[[0, 1], 0], [-1.0, 1.0], atol=1e-6)
    assert scaled[2].tolist() == [0.0, 0.0, 1.0]


def test_train_and_predict_returns_probability_per_node():
    probabilities = train_and_predict(_tiny_graph(), train_end=1, seed=0, epochs=5, hidden_dim=4)
    assert probabilities.shape == (6,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
