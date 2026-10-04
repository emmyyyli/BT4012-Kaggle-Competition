"""Unsupervised graph-Transformer embedding pilot for the Elliptic graph."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import get_feature_columns, load_data
from src.features import build_features
from src.graph import build_union_graph_data
from src.modeling import make_model
from src.validation import holdout_split


@dataclass(frozen=True)
class GTSSLConfig:
    walk_length: int = 8
    restart_probability: float = 0.15
    degree_penalty: float = 0.5
    similarity_weight: float = 1.0
    feature_similarity_temperature: float = 2.0
    hidden_dim: int = 64
    attention_heads: int = 4
    encoder_layers: int = 2
    dropout: float = 0.1
    mask_probability: float = 0.15
    edge_drop_probability: float = 0.1
    feature_noise_std: float = 0.05
    contrastive_temperature: float = 0.1
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    random_state: int = 42


def _outgoing_neighbors(edge_index, num_nodes):
    sources, destinations = edge_index
    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(offsets, sorted_sources + 1, 1)
    np.cumsum(offsets, out=offsets)
    return [destinations[order[offsets[i] : offsets[i + 1]]] for i in range(num_nodes)]


def build_walk_sequences(
    edge_index,
    node_features,
    center_indices,
    num_nodes,
    config=GTSSLConfig(),
    random_state=None,
):
    """Serialize each center as an eight-node directed, restarted walk."""
    rng = np.random.default_rng(
        config.random_state if random_state is None else random_state
    )
    neighbors = _outgoing_neighbors(edge_index, num_nodes)
    sources, destinations = edge_index
    degree = np.bincount(sources, minlength=num_nodes) + np.bincount(
        destinations, minlength=num_nodes
    )

    transaction_features = np.nan_to_num(
        node_features[:, :-1], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(transaction_features, axis=1)
    normalized = transaction_features / np.maximum(norms[:, None], 1e-8)
    edge_similarity = np.einsum(
        "ij,ij->i", normalized[sources], normalized[destinations]
    )

    edge_order = np.argsort(sources, kind="stable")
    sorted_sources = sources[edge_order]
    edge_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(edge_offsets, sorted_sources + 1, 1)
    np.cumsum(edge_offsets, out=edge_offsets)
    edge_weights = np.exp(
        np.clip(config.feature_similarity_temperature * edge_similarity, -10, 10)
        * config.similarity_weight
    ) / np.power(np.maximum(degree[destinations], 1), config.degree_penalty)
    edge_weights = edge_weights[edge_order]

    sequences = np.empty((len(center_indices), config.walk_length), dtype=np.int32)
    walk_edges = np.zeros(
        (len(center_indices), config.walk_length - 1), dtype=np.bool_
    )
    sequences[:, 0] = center_indices
    for row, center in enumerate(center_indices):
        current = int(center)
        for step in range(1, config.walk_length):
            traversed_edge = False
            if rng.random() < config.restart_probability:
                current = int(center)
            else:
                start, end = edge_offsets[current : current + 2]
                if start == end:
                    current = int(center)
                else:
                    weights = edge_weights[start:end]
                    cumulative = np.cumsum(weights)
                    draw = rng.random() * cumulative[-1]
                    current = int(neighbors[current][np.searchsorted(cumulative, draw)])
                    traversed_edge = True
            sequences[row, step] = current
            walk_edges[row, step - 1] = traversed_edge
    return sequences, walk_edges


class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hop_attenuation = nn.Parameter(torch.tensor(0.55))

    def forward(self, hidden, edge_keep):
        batch_size, sequence_length, hidden_dim = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        direct = torch.zeros(
            batch_size,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        )
        direct[:, :-1, 1:] = torch.diag_embed(edge_keep.bool())
        direct[:, 1:, :-1] = torch.diag_embed(edge_keep.bool())
        distance = torch.full(
            (batch_size, sequence_length, sequence_length),
            float("inf"),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        diagonal = torch.arange(sequence_length, device=hidden.device)
        distance[:, diagonal, diagonal] = 0.0
        distance = torch.where(direct, torch.ones_like(distance), distance)

        power = direct.to(hidden.dtype)
        for hop in (2, 3):
            power = torch.bmm(power, direct.to(hidden.dtype))
            newly_reached = (power > 0) & ~torch.isfinite(distance)
            distance = torch.where(newly_reached, float(hop), distance)

        # Keep soft attention to all tokens; attenuate by walk-graph hop distance.
        finite_distance = torch.where(
            torch.isfinite(distance),
            distance,
            torch.full_like(distance, 4.0),
        )
        structural_bias = -F.softplus(self.hop_attenuation) * finite_distance
        structural_bias = torch.where(
            distance == 0, torch.zeros_like(structural_bias), structural_bias
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores + structural_bias[:, None, :, :]
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = StructureAwareAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_keep):
        hidden = self.norm1(hidden + self.dropout(self.attention(hidden, edge_keep)))
        return self.norm2(hidden + self.dropout(self.feedforward(hidden)))


class GTSSLPretrainer(nn.Module):
    """Transformer encoder with masked reconstruction and InfoNCE objectives."""

    def __init__(self, input_dim, reconstruction_dim, config=GTSSLConfig()):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(reconstruction_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_dim, config.attention_heads, config.dropout
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.reconstruction_head = nn.Linear(
            config.hidden_dim, reconstruction_dim
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def encode(self, features, edge_keep):
        hidden = self.input_projection(features)
        for layer in self.layers:
            hidden = layer(hidden, edge_keep)
        return hidden

    def forward(self, features, edge_keep):
        hidden = self.encode(features, edge_keep)
        reconstructed = self.reconstruction_head(hidden)
        projected_center = F.normalize(
            self.contrastive_projection(hidden[:, 0]), dim=1
        )
        return reconstructed, projected_center, hidden


def _normalize_node_features(graph_data, node_df, cutoff):
    raw = graph_data["features"].astype(np.float32, copy=True)
    known = node_df["is_missing_features"].to_numpy() == 0
    times = pd.to_numeric(node_df["time_step"], errors="coerce").to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    fit_mask = known & np.isfinite(times) & (times <= cutoff)
    if not fit_mask.any():
        raise ValueError("No known-feature nodes are available before the pretraining cutoff")

    scaler_values = raw[fit_mask, :-1]
    mean = scaler_values.mean(axis=0)
    scale = scaler_values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    raw[:, :-1] = np.clip((raw[:, :-1] - mean) / scale, -8.0, 8.0)
    raw[~known, :-1] = 0.0
    raw[:, -1] = (~known).astype(np.float32)
    return raw, mean.astype(np.float32), scale.astype(np.float32), fit_mask


def _view_edge_keep(edge_flags, drop_probability, generator):
    random_keep = torch.rand(
        edge_flags.shape, device=edge_flags.device, generator=generator
    ).ge(drop_probability)
    return edge_flags & random_keep


def _contrastive_loss(first, second, temperature):
    logits = first @ second.T / temperature
    targets = torch.arange(len(first), device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def _train_epoch(
    model, optimizer, features, sequences, walk_edges, center_mask, config, device, epoch
):
    selected = np.flatnonzero(center_mask)
    rng = np.random.default_rng(config.random_state + epoch)
    rng.shuffle(selected)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.random_state + epoch)
    model.train()
    batch_losses = []

    for start in range(0, len(selected), config.batch_size):
        centers = selected[start : start + config.batch_size]
        sequence_indices = sequences[centers]
        batch_walk_edges = torch.from_numpy(walk_edges[centers]).to(device)
        batch = torch.from_numpy(features[sequence_indices]).to(device)
        missing_feature_node = batch[:, :, -1].ge(0.5)
        eligible = ~missing_feature_node
        mask_positions = (
            torch.rand(
                eligible.shape, device=device, generator=generator
            ).lt(config.mask_probability)
            & eligible
        )

        masked = batch.clone()
        masked[..., :-1] = torch.where(
            mask_positions.unsqueeze(-1),
            model.mask_token.view(1, 1, -1),
            masked[..., :-1],
        )
        edge_keep_a = _view_edge_keep(
            batch_walk_edges,
            config.edge_drop_probability,
            generator,
        )
        edge_keep_b = _view_edge_keep(
            batch_walk_edges,
            config.edge_drop_probability,
            generator,
        )

        reconstructed, _, _ = model(masked, edge_keep_a)
        reconstruction_mask = mask_positions.unsqueeze(-1).expand_as(reconstructed)
        if reconstruction_mask.any():
            reconstruction_loss = F.mse_loss(
                reconstructed[reconstruction_mask],
                batch[..., :-1][reconstruction_mask],
            )
        else:
            reconstruction_loss = reconstructed.sum() * 0.0

        view_a = batch.clone()
        view_b = batch.clone()
        noise_a = torch.randn(
            view_a[..., :-1].shape, device=device, generator=generator
        )
        noise_b = torch.randn(
            view_b[..., :-1].shape, device=device, generator=generator
        )
        view_a[..., :-1] += config.feature_noise_std * noise_a
        view_b[..., :-1] += config.feature_noise_std * noise_b
        _, projected_a, _ = model(view_a, edge_keep_a)
        _, projected_b, _ = model(view_b, edge_keep_b)
        contrastive_loss = _contrastive_loss(
            projected_a, projected_b, config.contrastive_temperature
        )

        loss = reconstruction_loss + contrastive_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_losses.append(
            (float(reconstruction_loss.detach()), float(contrastive_loss.detach()))
        )

    if not batch_losses:
        raise ValueError("No pretraining batches were produced")
    return np.asarray(batch_losses).mean(axis=0)


def train_pretrainer(
    graph_data,
    pretrain_cutoff=None,
    config=GTSSLConfig(),
    device=None,
):
    """Pretrain without labels; restrict SSL centers and walks to the cutoff."""
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    node_df = graph_data["node_df"]
    times = pd.to_numeric(node_df["time_step"], errors="coerce").to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    if pretrain_cutoff is None:
        pretrain_cutoff = int(np.nanmax(times))
    features, mean, scale, _ = _normalize_node_features(
        graph_data, node_df, pretrain_cutoff
    )
    center_mask = np.isfinite(times) & (times <= pretrain_cutoff)
    edge_index = graph_data["edge_index"]
    edge_times_ok = (
        np.isfinite(times[edge_index[0]])
        & np.isfinite(times[edge_index[1]])
        & (times[edge_index[0]] <= pretrain_cutoff)
        & (times[edge_index[1]] <= pretrain_cutoff)
    )
    restricted_edges = edge_index[:, edge_times_ok]
    sequences, walk_edges = build_walk_sequences(
        restricted_edges,
        features,
        np.arange(len(node_df), dtype=np.int32),
        len(node_df),
        config,
    )

    model = GTSSLPretrainer(
        input_dim=features.shape[1],
        reconstruction_dim=features.shape[1] - 1,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    history = []
    for epoch in range(config.epochs):
        reconstruction_loss, contrastive_loss = _train_epoch(
            model,
            optimizer,
            features,
            sequences,
            walk_edges,
            center_mask,
            config,
            device,
            epoch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "reconstruction_loss": float(reconstruction_loss),
                "contrastive_loss": float(contrastive_loss),
            }
        )
        print(
            f"epoch {epoch + 1}/{config.epochs}: "
            f"reconstruction={reconstruction_loss:.5f}, "
            f"contrastive={contrastive_loss:.5f}"
        )

    return model, features, mean, scale, sequences, history, device


@torch.no_grad()
def extract_embeddings(
    model, features, sequences, walk_edges, node_ids, config, device
):
    model.eval()
    output = np.empty(
        (len(sequences), config.hidden_dim), dtype=np.float32
    )
    for start in range(0, len(sequences), config.batch_size):
        batch_ids = node_ids[start : start + config.batch_size]
        indices = sequences[start : start + len(batch_ids)]
        batch = torch.from_numpy(features[indices]).to(device)
        edge_keep = torch.from_numpy(walk_edges[start : start + len(batch_ids)]).to(
            device
        )
        hidden = model.encode(batch, edge_keep)
        output[start : start + len(batch_ids)] = hidden[:, 0].cpu().numpy()
    return output


def _bootstrap_auc_delta(labels, baseline, augmented, random_state, iterations=500):
    rng = np.random.default_rng(random_state)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    deltas = np.empty(iterations, dtype=np.float32)
    for iteration in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sampled_labels = labels[sample]
        deltas[iteration] = roc_auc_score(
            sampled_labels, augmented[sample]
        ) - roc_auc_score(sampled_labels, baseline[sample])
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def run_holdout_pilot(
    output_dir=Path("artifacts"),
    config=GTSSLConfig(),
    pretrain_scope="causal",
):
    """Compare the supplied-feature baseline with SSL embeddings on the long holdout."""
    if pretrain_scope not in {"causal", "full_transductive"}:
        raise ValueError("pretrain_scope must be 'causal' or 'full_transductive'")
    train, test, edges, _ = load_data()
    graph_data = build_union_graph_data(train, test, edges)
    pretrain_cutoff = 21 if pretrain_scope == "causal" else None
    model, normalized_features, _, _, _, history, device = train_pretrainer(
        graph_data, pretrain_cutoff=pretrain_cutoff, config=config
    )

    node_df = graph_data["node_df"]
    all_node_indices = np.arange(len(node_df), dtype=np.int32)
    full_edge_index = graph_data["edge_index"]
    all_sequences, all_walk_edges = build_walk_sequences(
        full_edge_index,
        normalized_features,
        all_node_indices,
        len(node_df),
        config,
        random_state=config.random_state + 1,
    )
    feature_node_df = pd.concat(
        [train[["txId", "time_step", *get_feature_columns(train.columns)]],
         test[["txId", "time_step", *get_feature_columns(test.columns)]]],
        ignore_index=True,
    )
    node_index = pd.Series(
        node_df["node_idx"].to_numpy(), index=node_df["txId"]
    )
    feature_node_indices = feature_node_df["txId"].map(node_index).to_numpy()
    embeddings = extract_embeddings(
        model,
        normalized_features,
        all_sequences[feature_node_indices],
        all_walk_edges[feature_node_indices],
        feature_node_indices,
        config,
        device,
    )
    embedding_columns = [
        f"gt_ssl_{index:03d}" for index in range(config.hidden_dim)
    ]
    embedding_frame = pd.DataFrame(embeddings, columns=embedding_columns)
    embedding_frame.insert(0, "txId", feature_node_df["txId"].to_numpy())

    base_features = build_features(train, test, edges, include_in_degree=False)
    feature_columns = get_feature_columns(train.columns)
    combined = base_features[["txId", *feature_columns]].merge(
        embedding_frame, on="txId", how="left", validate="one_to_one"
    )
    if combined[embedding_columns].isna().any().any():
        raise ValueError("Failed to align one or more SSL embeddings by txId")

    known_mask = train["label"].notna().to_numpy()
    labeled = train.loc[known_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = np.flatnonzero(known_mask)
    train_positions, valid_positions = holdout_split(
        labeled, train_end=21, valid_end=35
    )
    train_rows = labeled_positions[train_positions]
    valid_rows = labeled_positions[valid_positions]
    labels_train = train["label"].iloc[train_rows].astype(int)
    labels_valid = train["label"].iloc[valid_rows].astype(int)

    predictions = {}
    for name, columns in (
        ("supplied_features", feature_columns),
        ("supplied_plus_gt_ssl", [*feature_columns, *embedding_columns]),
    ):
        classifier = make_model()
        classifier.fit(combined.iloc[train_rows][columns], labels_train)
        probabilities = classifier.predict_proba(
            combined.iloc[valid_rows][columns]
        )[:, 1]
        predictions[name] = probabilities

    baseline_probabilities = predictions["supplied_features"]
    augmented_probabilities = predictions["supplied_plus_gt_ssl"]
    auc_delta_ci = _bootstrap_auc_delta(
        labels_valid.to_numpy(),
        baseline_probabilities,
        augmented_probabilities,
        random_state=config.random_state,
    )
    valid_times = train["time_step"].iloc[valid_rows].to_numpy()
    metrics = []
    for name, probabilities in predictions.items():
        for window, lower, upper in (
            ("22-35", 22, 35),
            ("22-28", 22, 28),
            ("29-35", 29, 35),
        ):
            window_mask = (valid_times >= lower) & (valid_times <= upper)
            y_window = labels_valid.to_numpy()[window_mask]
            p_window = probabilities[window_mask]
            row = {
                "model": name,
                "pretraining_scope": pretrain_scope,
                "pretrain_cutoff": 21 if pretrain_scope == "causal" else "all",
                "validation_window": window,
                "embedding_dim": config.hidden_dim
                if name.endswith("gt_ssl")
                else 0,
                "roc_auc": float(roc_auc_score(y_window, p_window)),
                "pr_auc": float(average_precision_score(y_window, p_window)),
                "train_rows": len(train_rows),
                "valid_rows": int(window_mask.sum()),
            }
            if name == "supplied_plus_gt_ssl":
                row["auc_delta_vs_baseline"] = float(
                    row["roc_auc"]
                    - roc_auc_score(y_window, baseline_probabilities[window_mask])
                )
                if window == "22-35":
                    row["auc_delta_ci_2_5"] = auc_delta_ci[0]
                    row["auc_delta_ci_97_5"] = auc_delta_ci[1]
            metrics.append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_suffix = f"{pretrain_scope}_seed{config.random_state}"
    metrics_path = output_dir / f"gt_ssl_pilot_holdout_{artifact_suffix}.csv"
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_path, index=False)
    history_path = output_dir / f"gt_ssl_pilot_pretraining_{artifact_suffix}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    embedding_path = output_dir / f"gt_ssl_pilot_embeddings_{artifact_suffix}.npz"
    np.savez_compressed(
        embedding_path,
        txId=feature_node_df["txId"].to_numpy(dtype=str),
        embeddings=embeddings,
        feature_columns=np.asarray(feature_columns, dtype=str),
        embedding_columns=np.asarray(embedding_columns, dtype=str),
        config=np.asarray([str(asdict(config))]),
    )
    predictions_path = (
        output_dir / f"gt_ssl_pilot_predictions_{artifact_suffix}.npz"
    )
    np.savez_compressed(
        predictions_path,
        labels=labels_valid.to_numpy(dtype=np.int8),
        time_step=valid_times.astype(np.int16),
        baseline=baseline_probabilities,
        augmented=augmented_probabilities,
    )

    print(f"\n=== Matched holdout comparison ({pretrain_scope}) ===")
    print(metrics_frame.to_string(index=False))
    print(f"Saved holdout results: {metrics_path}")
    print(f"Saved pretraining losses: {history_path}")
    print(f"Saved feature-node embeddings: {embedding_path}")
    print(f"Saved validation predictions: {predictions_path}")
    return metrics_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=GTSSLConfig.epochs)
    parser.add_argument("--hidden-dim", type=int, default=GTSSLConfig.hidden_dim)
    parser.add_argument("--batch-size", type=int, default=GTSSLConfig.batch_size)
    parser.add_argument("--seed", type=int, default=GTSSLConfig.random_state)
    parser.add_argument(
        "--pretrain-scope",
        choices=("causal", "full_transductive"),
        default="causal",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden_dim < 4 or args.batch_size < 2:
        parser.error("epochs >= 1, hidden_dim >= 4, and batch_size >= 2 are required")
    config = GTSSLConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    run_holdout_pilot(config=config, pretrain_scope=args.pretrain_scope)


if __name__ == "__main__":
    main()
