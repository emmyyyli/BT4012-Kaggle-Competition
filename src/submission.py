from pathlib import Path

import numpy as np
import pandas as pd


def make_submission(test, test_features, model):
    probabilities = model.predict_proba(test_features)[:, 1]
    submission = pd.DataFrame(
        {"index": test["index"].to_numpy(), "target": probabilities}
    )
    if submission["index"].duplicated().any():
        raise ValueError("Submission index values must be unique")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be finite probabilities between 0 and 1")
    return submission


def write_submission(submission, output_path, sample_submission=None):
    if list(submission.columns) != ["index", "target"]:
        raise ValueError("Submission columns must be exactly index,target")
    if sample_submission is not None:
        expected = sample_submission["index"].to_numpy()
        actual = submission["index"].to_numpy()
        if len(actual) != len(expected) or not np.array_equal(actual, expected):
            raise ValueError("Submission indices must match sample_submission order")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path