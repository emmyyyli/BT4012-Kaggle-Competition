import numpy as np
import pandas as pd

from src.data import get_feature_columns


IDENTIFIER_COLUMNS = {"index", "txId", "label", "time_step"}

# The Elliptic documentation lists 94 local features (the first is the time step)
# followed by 72 one-hop aggregated features. With time_step supplied as its own
# column, feat_1-feat_93 are the local features and feat_94-feat_165 the aggregates.
# An attempt to reconstruct the aggregates from neighbour statistics was inconclusive,
# so this split rests on the dataset documentation (see report/findings.md).
LAST_LOCAL_FEATURE = 93


def get_feature_groups(columns):
    """Split supplied feat_ columns into documented local and aggregated groups."""
    supplied = get_feature_columns(columns)
    local = [column for column in supplied if int(column.split("_")[1]) <= LAST_LOCAL_FEATURE]
    aggregated = [column for column in supplied if int(column.split("_")[1]) > LAST_LOCAL_FEATURE]
    return {"all": supplied, "local": local, "aggregated": aggregated}


def build_features(train, test, edges=None, include_in_degree=True):
    """Create tabular features plus causal incoming degree for each transaction."""
    feature_columns = get_feature_columns(train.columns)
    shared_columns = ["txId", "time_step", *feature_columns]
    combined = pd.concat(
        [train[shared_columns], test[shared_columns]],
        ignore_index=True,
    )

    if include_in_degree and edges is not None:
        incoming_degree = (
            edges.drop_duplicates(["txId1", "txId2"])
            .groupby("txId2", sort=False)
            .size()
            .rename("in_degree")
        )
        combined = combined.join(incoming_degree, on="txId")
        combined["in_degree"] = combined["in_degree"].fillna(0).astype("int32")
    return combined


def build_experiment_features(train, test, edges=None):
    """Add experimental graph-degree and row-summary features to supplied features."""
    combined = build_features(train, test, edges, include_in_degree=True)
    feature_columns = get_feature_columns(train.columns)
    feature_values = combined[feature_columns].to_numpy(dtype=np.float32)

    combined["feature_mean"] = feature_values.mean(axis=1)
    combined["feature_std"] = feature_values.std(axis=1)
    combined["feature_min"] = feature_values.min(axis=1)
    combined["feature_max"] = feature_values.max(axis=1)
    combined["feature_abs_mean"] = np.abs(feature_values).mean(axis=1)
    combined["feature_zero_fraction"] = (feature_values == 0).mean(axis=1)

    if edges is not None:
        outgoing_degree = (
            edges.drop_duplicates(["txId1", "txId2"])
            .groupby("txId1", sort=False)
            .size()
            .rename("out_degree")
        )
        combined = combined.join(outgoing_degree, on="txId")
        combined["out_degree"] = combined["out_degree"].fillna(0).astype("int32")

    if "in_degree" in combined:
        combined["log_in_degree"] = np.log1p(combined["in_degree"])
    if "out_degree" in combined:
        combined["log_out_degree"] = np.log1p(combined["out_degree"])
    return combined


def get_experiment_feature_sets(feature_frame, include_out_degree=False):
    """Return fixed feature sets for controlled temporal ablations."""
    supplied = get_feature_columns(feature_frame.columns)
    row_summaries = [
        "feature_mean",
        "feature_std",
        "feature_min",
        "feature_max",
        "feature_abs_mean",
        "feature_zero_fraction",
    ]

    variants = {
        "supplied": supplied,
        "time_step": [*supplied, "time_step"],
        "in_degree": [*supplied, "in_degree"],
        "log_in_degree": [*supplied, "log_in_degree"],
        "time_and_in_degree": [*supplied, "time_step", "in_degree"],
        "row_summaries": [*supplied, *row_summaries],
        "in_degree_and_row_summaries": [
            *supplied,
            "in_degree",
            *row_summaries,
        ],
    }
    if include_out_degree and "out_degree" in feature_frame:
        variants["out_degree_diagnostic"] = [*supplied, "out_degree"]
        variants["degree_and_row_summaries"] = [
            *supplied,
            "in_degree",
            "out_degree",
            *row_summaries,
        ]
    return variants


def get_model_feature_columns(feature_frame, include_in_degree=False):
    excluded = set(IDENTIFIER_COLUMNS)
    if not include_in_degree:
        excluded.add("in_degree")
    return [
        column
        for column in feature_frame.columns
        if column not in excluded
    ]