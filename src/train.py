import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_DIR, PROJECT_ROOT, load_data
from src.features import build_features, get_model_feature_columns
from src.modeling import evaluate_holdout, evaluate_temporal_cv, fit_final_model
from src.submission import make_submission, write_submission


def run_pipeline(data_dir=DATA_DIR, output_dir=None, n_splits=4, include_in_degree=False):
    output_dir = PROJECT_ROOT if output_dir is None else Path(output_dir)
    train, test, edges, sample_submission = load_data(data_dir)
    feature_frame = build_features(train, test, edges, include_in_degree=include_in_degree)
    feature_columns = get_model_feature_columns(feature_frame, include_in_degree=include_in_degree)
    feature_matrix = feature_frame[feature_columns]

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    validation_results = evaluate_temporal_cv(
        train,
        feature_matrix.iloc[: len(train)].reset_index(drop=True),
        n_splits=n_splits,
    )
    validation_path = artifacts_dir / "validation_results.csv"
    validation_results.to_csv(validation_path, index=False)

    holdout_results = evaluate_holdout(
        train,
        feature_matrix.iloc[: len(train)].reset_index(drop=True),
        train_end=21,
        valid_end=35,
    )
    holdout_path = artifacts_dir / "holdout_results.csv"
    pd.DataFrame([holdout_results]).to_csv(holdout_path, index=False)

    model = fit_final_model(
        train,
        feature_matrix.iloc[: len(train)].reset_index(drop=True),
    )
    test_features = feature_matrix.iloc[len(train) :].reset_index(drop=True)
    submission = make_submission(test, test_features, model)
    submission_path = write_submission(
        submission,
        output_dir / "submissions" / "submission.csv",
        sample_submission=sample_submission,
    )

    model_path = output_dir / "models" / "final_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    return validation_results, holdout_results, submission, model_path, submission_path


if __name__ == "__main__":
    cv_results, holdout_results, submission, model_path, submission_path = run_pipeline()
    print("=== Expanding Temporal CV ===")
    print(cv_results.to_string(index=False))
    print("\n=== Long Horizon Holdout (Steps 1-21 -> 22-35) ===")
    for k, v in holdout_results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved submission: {submission_path} ({len(submission)} rows)")