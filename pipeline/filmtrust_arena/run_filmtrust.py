"""
Run FilmTrust Benchmark — Social Arena for Social-Aware Recommendation Models.

Orchestrates end-to-end:
  1. Download the FilmTrust dataset (explicit ratings + explicit trust network)
  2. Parse and index into contiguous sparse matrices
  3. Train 3 engines: LightGCN (no-social baseline), TrustSVD, Social-LightGCN
  4. Evaluate Recall@K, NDCG@K, and inference latency on the held-out test set
  5. Print a comparative results table

This script is STANDALONE and never modifies production code or the Classic Arena.

Usage:
    python -m pipeline.filmtrust_arena.run_filmtrust
    python -m pipeline.filmtrust_arena.run_filmtrust --epochs 30 --dim 64 -k 10
"""
import os
import sys
import time
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.trust_svd_engine import TrustSVDEngine
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine
from pipeline.engines.benchmark_arena import recall_at_k, ndcg_at_k


def run_social_arena(
    data_dir: str = "data/filmtrust",
    n_epochs: int = 30,
    embedding_dim: int = 64,
    num_layers: int = 3,
    k: int = 10,
    max_eval_users: int = 500,
) -> pd.DataFrame:
    """Full Social Arena benchmark: load FilmTrust, train 3 engines, evaluate, compare."""
    print("=" * 70)
    print("  SOCIAL ARENA -- FILMTRUST DATASET (Explicit Trust Network)")
    print("  LightGCN (No-Social Baseline)  vs.  TrustSVD  vs.  Social-LightGCN")
    print("=" * 70, flush=True)

    print("\n[Step 1/4] Loading FilmTrust dataset ...", flush=True)
    loader = FilmTrustLoader(data_dir=data_dir)
    loader.download()
    loader.load_data()

    num_users = loader.num_users
    num_items = loader.num_items

    interaction_csr = loader.get_train_interaction_matrix()
    sym_adj_mat = loader.get_sym_adj_mat()
    trust_csr = loader.get_trust_matrix()
    test_dict = loader.get_test_dict()

    print(f"\n  Data Summary:")
    print(f"    Users        : {num_users:>8,}")
    print(f"    Items        : {num_items:>8,}")
    print(f"    Train ratings: {interaction_csr.nnz:>8,}")
    print(f"    Trust edges  : {trust_csr.nnz:>8,} (symmetrized)")
    print(f"    Test users   : {len(test_dict):>8,}")

    print(f"\n[Step 2/4] Initializing engines (dim={embedding_dim}, layers={num_layers}) ...")
    engines: Dict[str, Any] = {
        "LightGCN (No-Social)": LightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, num_layers=num_layers, n_epochs=n_epochs,
        ),
        "TrustSVD": TrustSVDEngine(n_factors=embedding_dim, n_epochs=n_epochs),
        "Social-LightGCN": SocialLightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, num_layers=num_layers, n_epochs=n_epochs,
        ),
    }

    print("\n[Step 3/4] Training & evaluating ...", flush=True)
    results: List[Dict[str, Any]] = []
    eval_users = list(test_dict.keys())[:max_eval_users]

    for name, engine in engines.items():
        print(f"\n{'=' * 50}")
        print(f"  Training: {name}")
        print(f"{'=' * 50}")

        t0 = time.time()
        if name == "LightGCN (No-Social)":
            engine.fit({"sym_adj_mat": sym_adj_mat, "interaction_matrix": interaction_csr})
        else:
            engine.fit({"interaction_matrix": interaction_csr, "trust_matrix": trust_csr})
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        print(f"  Evaluating {name} ...")
        recalls: List[float] = []
        ndcgs: List[float] = []
        latencies: List[float] = []
        for uid in eval_users:
            ground_truth = test_dict[uid]

            t1 = time.perf_counter()
            top_n = engine.recommend_top_n(uid, top_n=k)
            lat = (time.perf_counter() - t1) * 1000

            recalls.append(recall_at_k(top_n, ground_truth, k))
            ndcgs.append(ndcg_at_k(top_n, ground_truth, k))
            latencies.append(lat)

        result = {
            "Engine": name,
            "Train Time (s)": round(train_time, 1),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            "Latency (ms)": round(float(np.mean(latencies)), 2),
            "P95 Latency (ms)": round(float(np.percentile(latencies, 95)), 2),
        }
        results.append(result)
        print(f"    Recall@{k}: {result[f'Recall@{k}']:.4f}")
        print(f"    NDCG@{k}:   {result[f'NDCG@{k}']:.4f}")
        print(f"    Latency:   {result['Latency (ms)']:.2f}ms (P95: {result['P95 Latency (ms)']:.2f}ms)")

    print("\n" + "=" * 70)
    print("[Step 4/4] BENCHMARK RESULTS -- FILMTRUST SOCIAL ARENA")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    csv_path = "models/filmtrust_arena_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FilmTrust Social Arena: LightGCN vs. TrustSVD vs. Social-LightGCN"
    )
    parser.add_argument("--data_dir", type=str, default="data/filmtrust", help="Directory for FilmTrust dataset files")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("-k", type=int, default=10, help="Top-K for Recall@K / NDCG@K")
    args = parser.parse_args()

    run_social_arena(
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        k=args.k,
    )
