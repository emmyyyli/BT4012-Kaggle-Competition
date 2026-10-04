import pandas as pd

from src.data import load_data
from src.features import build_features


def test_loader_accepts_train_without_index_and_preserves_test_index(tmp_path):
    train = pd.DataFrame(
        {
            "txId": ["train-a", "train-b"],
            "time_step": [1, 2],
            "feat_1": [0.1, 0.2],
            "label": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "index": [17],
            "txId": ["test-a"],
            "time_step": [3],
            "feat_1": [0.3],
        }
    )
    edges = pd.DataFrame({"txId1": ["train-a"], "txId2": ["test-a"]})
    sample_submission = pd.DataFrame({"index": [17], "target": [0.5]})

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    edges.to_csv(tmp_path / "txs_edgelist.csv", index=False)
    sample_submission.to_csv(tmp_path / "sample_submission.csv", index=False)

    loaded_train, loaded_test, loaded_edges, _ = load_data(tmp_path)
    feature_frame = build_features(loaded_train, loaded_test, loaded_edges)

    assert "index" not in loaded_train.columns
    assert loaded_test["index"].tolist() == [17]
    assert feature_frame["in_degree"].tolist() == [0, 0, 1]