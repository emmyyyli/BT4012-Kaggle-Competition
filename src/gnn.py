import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


class GraphSAGENet(nn.Module):
    """Inductive 2-layer GraphSAGE network using sparse matrix message passing."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.w_self1 = nn.Linear(in_dim, hidden_dim)
        self.w_neigh1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.w_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.w_neigh2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        # Layer 1
        h_neigh1 = torch.sparse.mm(adj_norm, x)
        h1 = self.w_self1(x) + self.w_neigh1(h_neigh1)
        h1 = F.relu(self.bn1(h1))
        h1 = self.dropout(h1)

        # Layer 2
        h_neigh2 = torch.sparse.mm(adj_norm, h1)
        h2 = self.w_self2(h1) + self.w_neigh2(h_neigh2)
        h2 = F.relu(self.bn2(h2))
        h2 = self.dropout(h2)

        out = self.classifier(h2)
        return out, h2


def compute_normalized_adjacency(sparse_adj):
    """Compute row-normalized adjacency with self-loops: D^{-1}(A + I) as a torch sparse COO tensor."""
    n = sparse_adj.shape[0]
    adj_loop = sparse_adj + sp.eye(n, dtype=np.float32)
    deg = np.array(adj_loop.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1.0, where=deg > 0)
    deg_inv[deg == 0] = 0.0
    d_mat = sp.diags(deg_inv)
    norm_adj = d_mat.dot(adj_loop).tocoo()

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
    values = torch.from_numpy(norm_adj.data).float()
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))


def train_graphsage(
    graph_data,
    train_time_end=21,
    valid_time_end=35,
    hidden_dim=64,
    epochs=100,
    lr=0.01,
    weight_decay=1e-4,
    dropout=0.2,
    random_state=42,
    device="cpu",
):
    """
    Train GraphSAGE model respecting strict temporal topology isolation.
    Training loss is computed strictly on labeled nodes where time_step <= train_time_end.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    node_df = graph_data["node_df"]
    X = graph_data["features"]
    sparse_adj = graph_data["sparse_adj"]

    norm_adj_torch = compute_normalized_adjacency(sparse_adj).to(device)
    X_torch = torch.from_numpy(X).float().to(device)

    time_steps = node_df["time_step"].to_numpy()
    is_labeled = node_df["label"].notna().to_numpy()
    labels = np.where(node_df["label"].isna(), -1, node_df["label"].to_numpy()).astype(np.int64)

    tr_mask = is_labeled & (time_steps <= train_time_end)
    va_mask = is_labeled & (time_steps > train_time_end) & (time_steps <= valid_time_end)

    y_torch = torch.from_numpy(labels).to(device)
    tr_indices = torch.from_numpy(np.where(tr_mask)[0]).to(device)
    va_indices = torch.from_numpy(np.where(va_mask)[0]).to(device)

    # Class imbalance weighting
    tr_labels_np = labels[tr_mask]
    pos_count = max(1, int((tr_labels_np == 1).sum()))
    neg_count = max(1, int((tr_labels_np == 0).sum()))
    weight = torch.tensor([1.0, float(neg_count / pos_count)]).float().to(device)

    model = GraphSAGENet(
        in_dim=X.shape[1], hidden_dim=hidden_dim, out_dim=2, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_roc = 0.0
    best_pr = 0.0

    for _ in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_torch, norm_adj_torch)
        loss = criterion(logits[tr_indices], y_torch[tr_indices])
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(X_torch, norm_adj_torch)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    va_probs = probs[va_mask]
    va_y = labels[va_mask]
    if len(va_y) > 0 and len(np.unique(va_y)) >= 2:
        roc = float(roc_auc_score(va_y, va_probs))
        pr = float(average_precision_score(va_y, va_probs))
    else:
        roc = None
        pr = None

    # Build node-level probability series mapped to txId
    prob_series = pd.Series(probs, index=node_df["txId"], name="graphsage_prob")

    eval_metrics = {
        "train_max_time_step": int(train_time_end),
        "valid_min_time_step": int(train_time_end + 1) if valid_time_end is not None else None,
        "valid_max_time_step": int(valid_time_end) if valid_time_end is not None else None,
        "roc_auc": roc,
        "pr_auc": pr,
        "train_rows": int(tr_mask.sum()),
        "valid_rows": int(va_mask.sum()),
    }

    return model, prob_series, eval_metrics
