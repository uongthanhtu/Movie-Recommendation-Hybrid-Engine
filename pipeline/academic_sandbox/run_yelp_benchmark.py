"""
Run Yelp Benchmark — Top-Tier Academic Sandbox for Social-LightGCN.

Orchestrates end-to-end:
  1. Download the QRec Yelp dataset (with social trust network)
  2. Parse and index into contiguous sparse matrices
  3. Train 2 models: LightGCN (vanilla baseline) vs. Social-LightGCN (ours)
  4. Evaluate Recall@K, NDCG@K (ranking quality) on the held-out test set
  5. Print a comparative results table

This script is STANDALONE and NEVER modifies any production code.

Usage:
    python -m pipeline.academic_sandbox.run_yelp_benchmark
    python pipeline/academic_sandbox/run_yelp_benchmark.py [--data_dir DATA_DIR] [--epochs EPOCHS]
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.academic_sandbox.yelp_data_loader import YelpDataLoader
from pipeline.academic_sandbox.model_wrappers import (
    SocialLightGCNWrapper,
    VanillaLightGCNWrapper,
)


# ======================================================================
# Ranking Evaluation Metrics
# ======================================================================

def evaluate_ranking(
    model,
    test_dict: Dict[int, Set[int]],
    train_dict: Dict[int, Set[int]],
    num_items: int,
    k_list: List[int] = None,
    max_eval_users: int = 3000,
) -> Dict[str, float]:
    """
    Compute Recall@K and NDCG@K for multiple K values.

    Uses the **All-Ranking** protocol: for each test user, score ALL items,
    exclude training items, then compute metrics on the top-K predictions.

    Args:
        model:       Any object with ``get_all_scores(user_id)`` returning a tensor.
        test_dict:   {user_idx: set(item_idx)} ground truth.
        train_dict:  {user_idx: set(item_idx)} to exclude from ranking.
        num_items:   Total items.
        k_list:      List of K values, default [10, 20].
        max_eval_users: Cap for evaluation speed on CPU.

    Returns:
        Dict of metric_name -> value.
    """
    if k_list is None:
        k_list = [10, 20]

    eval_users = list(test_dict.keys())
    if len(eval_users) > max_eval_users:
        rng = np.random.default_rng(42)
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False).tolist()
        print(f"    (Sampled {max_eval_users} / {len(test_dict)} test users for evaluation)")

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    max_k = max(k_list)

    for u in eval_users:
        ground_truth = test_dict[u]
        if not ground_truth:
            continue

        scores = model.get_all_scores(u)

        # Exclude training items
        train_items = train_dict.get(u, set())
        if train_items:
            import torch
            mask = torch.LongTensor(list(train_items)).to(scores.device)
            scores[mask] = float("-inf")

        # Top-K
        import torch
        _, topk_idx = torch.topk(scores, min(max_k, num_items))
        topk_list = topk_idx.cpu().tolist()

        for k in k_list:
            top = topk_list[:k]

            # Recall@K
            hits = len(set(top) & ground_truth)
            recalls[k].append(hits / min(len(ground_truth), k))

            # NDCG@K
            dcg = sum(
                1.0 / np.log2(pos + 2)
                for pos, item in enumerate(top)
                if item in ground_truth
            )
            idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(ground_truth), k)))
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    results = {}
    for k in k_list:
        results[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        results[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0

    return results


# ======================================================================
# Main Orchestrator
# ======================================================================

def run_yelp_benchmark(
    data_dir: str = "data/yelp",
    n_epochs: int = 20,
    embedding_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 8192,
    lr: float = 1e-3,
    reg: float = 1e-4,
    k_list: List[int] = None,
):
    """
    Full benchmark pipeline for the Yelp dataset.
    """
    if k_list is None:
        k_list = [10, 20]

    print("=" * 75)
    print("  TOP-TIER ACADEMIC BENCHMARK -- YELP DATASET")
    print("  LightGCN (Baseline)  vs.  Social-LightGCN (Ours)")
    print("=" * 75, flush=True)

    # ------------------------------------------------------------------
    # Step 1: Download & Parse
    # ------------------------------------------------------------------
    print("\n[Step 1/4] Loading Yelp dataset ...", flush=True)
    loader = YelpDataLoader(data_dir=data_dir)
    loader.download()
    loader.load_data()

    num_users = loader.num_users
    num_items = loader.num_items

    interaction_csr = loader.get_train_interaction_matrix()
    trust_csr = loader.get_trust_matrix()
    test_dict = loader.get_test_dict()
    train_dict = loader.get_train_dict()

    density_interaction = interaction_csr.nnz / (num_users * num_items) * 100
    density_social = trust_csr.nnz / (num_users * num_users) * 100

    print(f"\n  Data Summary:")
    print(f"    Users         : {num_users:>10,}")
    print(f"    Items         : {num_items:>10,}")
    print(f"    Train edges   : {interaction_csr.nnz:>10,}  (density: {density_interaction:.4f}%)")
    print(f"    Trust edges   : {trust_csr.nnz:>10,}  (density: {density_social:.4f}%)")
    print(f"    Test users    : {len(test_dict):>10,}")

    # Auto-scale parameters for large datasets
    if num_users > 10000:
        n_epochs = min(n_epochs, 15)
        batch_size = max(batch_size, 16384)
        print(f"\n  [Auto-Scaling] Large dataset detected -> epochs={n_epochs}, batch_size={batch_size}")

    # ------------------------------------------------------------------
    # Step 2: Initialize Models
    # ------------------------------------------------------------------
    print(f"\n[Step 2/4] Initializing models (dim={embedding_dim}, layers={num_layers}) ...")

    models = {
        "LightGCN (Vanilla)": VanillaLightGCNWrapper(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            lr=lr,
            reg=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
        ),
        "Social-LightGCN (Ours)": SocialLightGCNWrapper(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            lr=lr,
            reg=reg,
            n_epochs=n_epochs,
            batch_size=batch_size,
        ),
    }

    # ------------------------------------------------------------------
    # Step 3: Train
    # ------------------------------------------------------------------
    print(f"\n[Step 3/4] Training ...", flush=True)
    results_list = []

    for name, model in models.items():
        print(f"\n  --- Training: {name} ---")
        t0 = time.time()
        model.fit(interaction_csr, trust_csr)
        train_time = time.time() - t0
        print(f"  Training completed in {train_time:.1f}s", flush=True)

        # Evaluate
        print(f"\n  --- Evaluating: {name} ---")
        t1 = time.time()
        metrics = evaluate_ranking(model, test_dict, train_dict, num_items, k_list=k_list)
        eval_time = time.time() - t1
        print(f"  Evaluation completed in {eval_time:.1f}s", flush=True)

        row = {"Model": name, "Train (s)": round(train_time, 1), "Eval (s)": round(eval_time, 1)}
        for metric_name, metric_val in metrics.items():
            row[metric_name] = round(metric_val, 4)

        results_list.append(row)

    # ------------------------------------------------------------------
    # Step 4: Print Results Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  BENCHMARK RESULTS -- YELP DATASET")
    print("=" * 75)

    # Header
    all_keys = list(results_list[0].keys())
    header = " | ".join(f"{k:>16}" for k in all_keys)
    print(header)
    print("-" * len(header))

    for row in results_list:
        line = " | ".join(f"{str(row[k]):>16}" for k in all_keys)
        print(line)

    print("=" * 75)

    # Improvement analysis
    if len(results_list) == 2:
        baseline = results_list[0]
        ours = results_list[1]
        print("\n  Improvement Analysis (Social-LightGCN vs. LightGCN):")
        for k in k_list:
            for metric in [f"Recall@{k}", f"NDCG@{k}"]:
                base_val = baseline.get(metric, 0)
                our_val = ours.get(metric, 0)
                if base_val > 0:
                    improvement = ((our_val - base_val) / base_val) * 100
                    symbol = "+" if improvement > 0 else ""
                    print(f"    {metric}: {base_val:.4f} -> {our_val:.4f} ({symbol}{improvement:.2f}%)")
                else:
                    print(f"    {metric}: {base_val:.4f} -> {our_val:.4f}")

    print("\n  Benchmark completed successfully!")
    return results_list


# ======================================================================
# CLI Entry Point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yelp Benchmark: LightGCN vs. Social-LightGCN")
    parser.add_argument("--data_dir", type=str, default="data/yelp", help="Directory for Yelp dataset files")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--batch_size", type=int, default=8192, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 regularization weight")
    args = parser.parse_args()

    run_yelp_benchmark(
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
    )
