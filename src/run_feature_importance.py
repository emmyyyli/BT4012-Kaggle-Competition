import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.features import build_features, get_model_feature_columns
from src.modeling import make_model
from src.validation import holdout_split


def run_permutation_importance(output_dir=Path("artifacts"), top_n=30, n_repeats=3):
    train, test, edges, _ = load_data()
    feature_frame = build_features(train, test, edges, include_in_degree=False)
    feature_columns = get_model_feature_columns(feature_frame)
    labeled_mask = train["label"].notna().to_numpy()
    labeled = train.loc[labeled_mask, ["time_step", "label"]].reset_index(drop=True)
    labeled_positions = labeled_mask.nonzero()[0]
    train_positions, valid_positions = holdout_split(
        labeled, train_end=21, valid_end=35
    )
    train_rows = labeled_positions[train_positions]
    valid_rows = labeled_positions[valid_positions]

    model = make_model()
    model.fit(
        feature_frame.iloc[train_rows][feature_columns],
        train["label"].iloc[train_rows].astype(int),
    )
    valid_features = feature_frame.iloc[valid_rows][feature_columns]
    valid_labels = train["label"].iloc[valid_rows].astype(int)
    baseline_auc = float(
        roc_auc_score(valid_labels, model.predict_proba(valid_features)[:, 1])
    )

    gains = model.booster_.feature_importance(importance_type="gain")
    gain_ranking = pd.DataFrame(
        {"feature": feature_columns, "gain": gains}
    ).sort_values("gain", ascending=False)
    selected_features = gain_ranking.head(top_n)["feature"].tolist()
    rng = np.random.default_rng(42)
    permutation_drops = []
    for feature in selected_features:
        drops = []
        for _ in range(n_repeats):
            permuted_features = valid_features.copy()
            permuted_features[feature] = rng.permutation(
                permuted_features[feature].to_numpy()
            )
            permuted_auc = roc_auc_score(
                valid_labels, model.predict_proba(permuted_features)[:, 1]
            )
            drops.append(baseline_auc - permuted_auc)
        permutation_drops.append(drops)

    result = pd.DataFrame(
        {
            "feature": selected_features,
            "permutation_auc_drop_mean": np.mean(permutation_drops, axis=1),
            "permutation_auc_drop_std": np.std(permutation_drops, axis=1),
        }
    )
    result = result.merge(gain_ranking, on="feature", how="left")
    result["baseline_auc"] = baseline_auc
    result["valid_start_time_step"] = 22
    result["valid_end_time_step"] = 35
    result = result.sort_values(
        "permutation_auc_drop_mean", ascending=False, ignore_index=True
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "permutation_feature_importance.csv"
    result.to_csv(output_path, index=False)
    print(f"Long-holdout ROC-AUC: {baseline_auc:.6f}")
    print(result.head(20).to_string(index=False))
    print(f"Saved permutation importance: {output_path}")


if __name__ == "__main__":
    run_permutation_importance()
