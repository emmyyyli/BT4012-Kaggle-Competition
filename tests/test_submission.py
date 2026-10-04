import numpy as np
import pandas as pd
import pytest

from src.submission import make_submission, write_submission


class ProbabilityModel:
    def predict_proba(self, features):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_submission_uses_test_index_and_sample_order(tmp_path):
    test = pd.DataFrame({"index": [8, 3], "txId": ["a", "b"]})
    submission = make_submission(test, pd.DataFrame(index=range(2)), ProbabilityModel())
    sample = pd.DataFrame({"index": [8, 3], "target": [0.5, 0.5]})

    output = write_submission(
        submission,
        tmp_path / "submission.csv",
        sample_submission=sample,
    )

    actual = pd.read_csv(output)
    assert actual.columns.tolist() == ["index", "target"]
    assert actual["index"].tolist() == [8, 3]
    assert actual["target"].tolist() == pytest.approx([0.2, 0.9])


def test_submission_rejects_misaligned_sample_order(tmp_path):
    submission = pd.DataFrame({"index": [8, 3], "target": [0.2, 0.9]})
    sample = pd.DataFrame({"index": [3, 8], "target": [0.5, 0.5]})

    with pytest.raises(ValueError, match="sample_submission order"):
        write_submission(submission, tmp_path / "submission.csv", sample_submission=sample)