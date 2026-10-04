import numpy as np
import pandas as pd

from src.features import (
    build_experiment_features,
    build_features,
    get_experiment_feature_sets,
    get_model_feature_columns,
)
from src.data import get_feature_columns


def test_feature_columns_are_numerically_ordered():
    assert get_feature_columns(["feat_10", "feat_2", "label"]) == ["feat_2", "feat_10"]


def test_build_features_adds_directed_in_degree_without_future_out_degree():
    train = pd.DataFrame(
        {
            "index": [0, 1],
            "txId": ["a", "b"],
            "time_step": [1, 2],
            "feat_1": [0.2, 0.3],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "index": [0],
            "txId": ["c"],
            "time_step": [3],
            "feat_1": [0.4],
        }
    )
    edges = pd.DataFrame(
        {
            "txId1": ["a", "a", "a", "b"],
            "txId2": ["b", "c", "c", "c"],
        }
    )

    features = build_features(train, test, edges)

    assert features["in_degree"].tolist() == [0, 1, 2]
    assert get_model_feature_columns(features, include_in_degree=True) == ["feat_1", "in_degree"]
    assert get_model_feature_columns(features, include_in_degree=False) == ["feat_1"]
    assert "time_step" not in get_model_feature_columns(features)
    assert "out_degree" not in features.columns


def test_experiment_features_add_row_summaries_and_deduplicated_degrees():
    train = pd.DataFrame(
        {
            "txId": ["a", "b"],
            "time_step": [1, 2],
            "feat_1": [0.0, -2.0],
            "feat_2": [2.0, 2.0],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "txId": ["c"],
            "time_step": [3],
            "feat_1": [4.0],
            "feat_2": [0.0],
        }
    )
    edges = pd.DataFrame(
        {
            "txId1": ["a", "a", "b"],
            "txId2": ["b", "b", "c"],
        }
    )

    features = build_experiment_features(train, test, edges)

    assert features["in_degree"].tolist() == [0, 1, 1]
    assert features["out_degree"].tolist() == [1, 1, 0]
    assert np.allclose(features["log_in_degree"], [0.0, np.log(2), np.log(2)])
    assert features["feature_mean"].tolist() == [1.0, 0.0, 2.0]
    assert features["feature_std"].tolist() == [1.0, 2.0, 2.0]
    assert features["feature_zero_fraction"].tolist() == [0.5, 0.0, 0.5]

    variants = get_experiment_feature_sets(features, include_out_degree=True)
    assert "time_step" not in variants["supplied"]
    assert "in_degree" in variants["in_degree"]
    assert "out_degree" in variants["out_degree_diagnostic"]
    assert "feature_mean" in variants["row_summaries"]