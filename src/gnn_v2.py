"""Directed GraphSAGE trained under the same neighbourhood conditions as the test period.

In steps 36-49 only labeled transactions have features, so every unlabeled neighbour of a
test node is a blank (zero-feature) node. In steps 1-35 unlabeled neighbours keep their
features. The first GraphSAGE was trained on the second regime and scored on the first.

This version applies ``mask_unlabeled_features``: in every step, nodes without a label have
their features zeroed and their missing-feature flag set, so training and test
neighbourhoods follow one rule. 22% of training-step nodes are labeled versus 24.7% of
test-step nodes having features, so the blank-neighbour rates also roughly match.

Other changes from ``gnn.py``:
* incoming and outgoing neighbours are aggregated separately (payment direction kept);
* LayerNorm replaces BatchNorm, whose batch statistics mixed all time steps;
* features are standardised with statistics from training-period nodes only;
* a fixed epoch budget, with no early stopping on validation labels.

Edges never connect different time steps (checked: 0 of 175,918 feature-node edges), so
full-graph message passing does not carry information across steps.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_unlabeled_features(features, node_df):
    """Zero features of unlabeled nodes and set their missing flag (last column).

    Test rows have no label in ``node_df`` but are labeled transactions in the original
    data (that is why they have features), so they are kept.
    """
    masked = features.copy()
    unlabeled = (node_df["label"].isna() & ~node_df["is_test"]).to_numpy()
    masked[unlabeled, :-1] = 0.0
    masked[unlabeled, -1] = 1.0
    return masked


def standardize(features, fit_rows):
    """Standardise feature columns (not the flag) using statistics from ``fit_rows``."""
    scaled = features.copy()
    reference = features[fit_rows, :-1]
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std == 0] = 1.0
    present = features[:, -1] == 0
    scaled[present, :-1] = (features[present, :-1] - mean) / std
    return scaled.astype(np.float32)


def row_normalized(sparse_matrix):
    """Return D^-1 A as a torch sparse tensor (rows with no neighbours stay zero)."""
    matrix = sp.csr_matrix(sparse_matrix, dtype=np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    normalized = sp.diags(inverse).dot(matrix).tocoo()
    indices = torch.from_numpy(np.vstack((normalized.row, normalized.col))).long()
    values = torch.from_numpy(normalized.data).float()
    return torch.sparse_coo_tensor(indices, values, torch.Size(normalized.shape)).coalesce()


class DirectedSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim, hidden_dim, hidden_dim]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.ModuleDict({
                "self": nn.Linear(d_in, d_out),
                "incoming": nn.Linear(d_in, d_out, bias=False),
                "outgoing": nn.Linear(d_in, d_out, bias=False),
            }))
            self.norms.append(nn.LayerNorm(d_out))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj_in, adj_out):
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = (
                layer["self"](h)
                + layer["incoming"](torch.sparse.mm(adj_in, h))
                + layer["outgoing"](torch.sparse.mm(adj_out, h))
            )
            h = self.dropout(F.relu(norm(h)))
        return self.classifier(h).squeeze(-1)


def train_and_predict(graph_data, train_end, seed=0, epochs=200, hidden_dim=64, lr=0.01, weight_decay=5e-4, dropout=0.3):
    """Train on labeled nodes with time_step <= train_end; return probabilities for all nodes."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_df = graph_data["node_df"]
    times = node_df["time_step"].to_numpy(dtype=np.float64, na_value=np.nan)
    labels = node_df["label"].to_numpy(dtype=np.float64, na_value=np.nan)

    masked = mask_unlabeled_features(graph_data["features"], node_df)
    in_training_period = times <= train_end
    fit_rows = in_training_period & (masked[:, -1] == 0)
    x = torch.from_numpy(standardize(masked, fit_rows))

    adjacency = graph_data["sparse_adj"]  # row = source, column = destination
    adj_out = row_normalized(adjacency)          # aggregate over successors
    adj_in = row_normalized(adjacency.T)         # aggregate over predecessors

    train_rows = np.flatnonzero(in_training_period & ~np.isnan(labels))
    y = torch.from_numpy(labels[train_rows].astype(np.float32))
    positives = float(y.sum())
    pos_weight = torch.tensor((len(y) - positives) / max(positives, 1.0))
    train_index = torch.from_numpy(train_rows)

    model = DirectedSAGE(x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj_in, adj_out)
        loss = F.binary_cross_entropy_with_logits(logits[train_index], y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, adj_in, adj_out)).numpy()
