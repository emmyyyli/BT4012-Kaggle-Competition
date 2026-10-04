from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_PATTERN = re.compile(r"feat_(\d+)$")


def get_feature_columns(columns):
    features = [column for column in columns if FEATURE_PATTERN.fullmatch(column)]
    return sorted(features, key=lambda column: int(FEATURE_PATTERN.fullmatch(column).group(1)))


def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"txId": "string"})
    test = pd.read_csv(data_dir / "test.csv", dtype={"txId": "string"})
    edges = pd.read_csv(
        data_dir / "txs_edgelist.csv",
        dtype={"txId1": "string", "txId2": "string"},
        usecols=["txId1", "txId2"],
    )
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    required_train = {"txId", "time_step", "label"}
    required_test = {"index", "txId", "time_step"}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    if missing := required_test.difference(test.columns):
        raise ValueError(f"test.csv is missing required columns: {sorted(missing)}")

    feature_columns = get_feature_columns(train.columns)
    if not feature_columns:
        raise ValueError("No feat_1 through feat_N feature columns were found")
    if get_feature_columns(test.columns) != feature_columns:
        raise ValueError("train.csv and test.csv must contain the same feat_ columns")

    normalized_labels = train["label"].astype("string").str.strip().str.lower()
    numeric_labels = pd.to_numeric(normalized_labels, errors="coerce")
    valid_labels = normalized_labels.isin(["none"]) | numeric_labels.isin([0, 1]) | normalized_labels.isna()
    if not valid_labels.all():
        unexpected = sorted(normalized_labels.loc[~valid_labels].unique().tolist())
        raise ValueError(f"Unexpected label values: {unexpected}")
    train["label"] = numeric_labels.astype("Int8")

    for frame_name, frame in (("train", train), ("test", test)):
        if frame["txId"].isna().any() or frame["txId"].duplicated().any():
            raise ValueError(f"{frame_name}.csv must have unique, non-null txId values")
        frame["time_step"] = pd.to_numeric(frame["time_step"], errors="raise")
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("txs_edgelist.csv must contain txId1 and txId2")
    if not {"index", "target"}.issubset(sample_submission.columns):
        raise ValueError("sample_submission.csv must contain index and target")

    return train, test, edges, sample_submission