import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, get_feature_columns
from src.graph import build_union_graph_data
from src.gnn import train_graphsage, compute_normalized_adjacency
from src.submission import make_submission, write_submission

def run_benchmark():
    print("Loading data...")
    train, test, edges, sample_submission = load_data()
    raw_feature_cols = get_feature_columns(train.columns)
    
    print("Building union graph...")
    graph_data = build_union_graph_data(train, test, edges)
    node_df = graph_data["node_df"]
    
    # 1. Train LightGBM Baseline on Holdout (1-21)
    print("\n--- Training Tabular LightGBM Baseline ---")
    h_tr_mask = (train["time_step"] <= 21) & train["label"].notna()
    h_va_mask = (train["time_step"] >= 22) & (train["time_step"] <= 35) & train["label"].notna()
    
    y_tr = train.loc[h_tr_mask, "label"].astype(int)
    y_va = train.loc[h_va_mask, "label"].astype(int)
    
    lgb_model = LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(train.loc[h_tr_mask, raw_feature_cols], y_tr)
    lgb_val_probs = lgb_model.predict_proba(train.loc[h_va_mask, raw_feature_cols])[:, 1]
    
    lgb_roc = float(roc_auc_score(y_va, lgb_val_probs))
    lgb_pr = float(average_precision_score(y_va, lgb_val_probs))
    print(f"LightGBM Baseline: ROC-AUC={lgb_roc:.5f}, PR-AUC={lgb_pr:.5f}")

    # 2. Train GraphSAGE End-to-End on Holdout (1-21)
    print("\n--- Training GraphSAGE End-to-End ---")
    sage_model, sage_prob_series, sage_eval = train_graphsage(
        graph_data,
        train_time_end=21,
        valid_time_end=35,
        hidden_dim=64,
        epochs=100,
        lr=0.01,
        weight_decay=1e-4,
        dropout=0.2,
        random_state=42,
    )
    
    # Align GraphSAGE predictions to train validation rows
    val_txids = train.loc[h_va_mask, "txId"]
    sage_val_probs = sage_prob_series.loc[val_txids].to_numpy()
    
    sage_roc = float(roc_auc_score(y_va, sage_val_probs))
    sage_pr = float(average_precision_score(y_va, sage_val_probs))
    print(f"GraphSAGE Standalone: ROC-AUC={sage_roc:.5f}, PR-AUC={sage_pr:.5f}")

    # 3. Evaluate Ensemble Blends
    blend_results = []
    blend_results.append({
        "model": "LightGBM Baseline (Tabular Only)",
        "roc_auc": lgb_roc,
        "pr_auc": lgb_pr,
        "sage_weight": 0.0,
        "lgb_weight": 1.0,
    })
    blend_results.append({
        "model": "GraphSAGE Standalone (GNN)",
        "roc_auc": sage_roc,
        "pr_auc": sage_pr,
        "sage_weight": 1.0,
        "lgb_weight": 0.0,
    })
    
    for alpha in [0.2, 0.3, 0.4, 0.5]:
        blend_probs = alpha * sage_val_probs + (1 - alpha) * lgb_val_probs
        b_roc = float(roc_auc_score(y_va, blend_probs))
        b_pr = float(average_precision_score(y_va, blend_probs))
        blend_results.append({
            "model": f"Ensemble Blend (SAGE={alpha:.1f} + LGBM={1-alpha:.1f})",
            "roc_auc": b_roc,
            "pr_auc": b_pr,
            "sage_weight": alpha,
            "lgb_weight": 1 - alpha,
        })
        print(f"Blend (SAGE={alpha:.1f} + LGBM={1-alpha:.1f}): ROC-AUC={b_roc:.5f}, PR-AUC={b_pr:.5f}")

    # Save benchmark table
    results_df = pd.DataFrame(blend_results)
    benchmark_path = Path("artifacts/graph_learning_benchmark.csv")
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(benchmark_path, index=False)
    print(f"\nSaved benchmark results to {benchmark_path}")

    # 4. Generate Final Test Predictions using the Ensemble
    print("\n--- Generating Final Test Submission from Full Graph Model ---")
    # Train full LightGBM on all labeled data (1-35)
    full_labeled_mask = train["label"].notna()
    full_lgb = LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    full_lgb.fit(
        train.loc[full_labeled_mask, raw_feature_cols],
        train.loc[full_labeled_mask, "label"].astype(int)
    )
    lgb_test_probs = full_lgb.predict_proba(test[raw_feature_cols])[:, 1]

    # Train full GraphSAGE on all labeled nodes (1-35)
    full_sage_model, full_sage_probs, _ = train_graphsage(
        graph_data,
        train_time_end=35,
        valid_time_end=35, # no valid
        hidden_dim=64,
        epochs=100,
        lr=0.01,
        weight_decay=1e-4,
        dropout=0.2,
        random_state=42,
    )
    sage_test_probs = full_sage_probs.loc[test["txId"]].to_numpy()

    # Blend test predictions: 0.3 SAGE + 0.7 LGBM (best PR-AUC and strong ROC-AUC)
    blend_test_probs = 0.3 * sage_test_probs + 0.7 * lgb_test_probs

    test_submission = pd.DataFrame({
        "index": test["index"],
        "target": blend_test_probs
    })
    sub_path = write_submission(
        test_submission,
        Path("submissions/submission_sage_blend.csv"),
        sample_submission=sample_submission
    )
    print(f"Saved ensemble submission to {sub_path} ({len(test_submission)} rows)")

if __name__ == "__main__":
    run_benchmark()
