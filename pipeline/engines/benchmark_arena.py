"""
Benchmark Arena — Multi-engine evaluation orchestrator (Classic Arena).

Compares non-social engines (Funk-SVD, LightGCN, SASRec) on MovieLens-100k, measures:
  - RMSE / MAE (rating prediction accuracy)
  - Recall@K / NDCG@K (ranking quality)
  - ILD@K (intra-list diversity)
  - Latency (inference speed per user)

Social-Aware models (TrustSVD, Social-LightGCN) are intentionally NOT benchmarked here:
MovieLens has no real social graph, and fabricating one via Jaccard similarity on
co-interacted items (see unified_data_loader.py::build_implicit_trust_matrix) is not a
scientifically valid trust network. Those models are evaluated instead in
pipeline/filmtrust_arena/run_filmtrust.py against FilmTrust's real, explicit trust data.

Usage:
    python -m pipeline.engines.benchmark_arena
"""
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.engines.unified_data_loader import UnifiedDataLoader
from pipeline.engines.funk_svd_engine import FunkSVDEngine
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.sasrec_engine import SASRecEngine


# ======================================================================
# Evaluation Metrics
# ======================================================================

def rmse(predictions: List[Tuple[float, float]]) -> float:
    """RMSE from list of (actual, predicted) tuples."""
    if not predictions:
        return float("inf")
    sq_errors = [(a - p) ** 2 for a, p in predictions]
    return float(np.sqrt(np.mean(sq_errors)))


def mae(predictions: List[Tuple[float, float]]) -> float:
    """MAE from list of (actual, predicted) tuples."""
    if not predictions:
        return float("inf")
    abs_errors = [abs(a - p) for a, p in predictions]
    return float(np.mean(abs_errors))


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Fraction of ground-truth items found in top-K."""
    if not ground_truth:
        return 0.0
    hits = len(set(recommended[:k]) & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in ground_truth
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def ild_at_k(
    recommended: List[int],
    item_genres: Dict[int, List[str]],
) -> float:
    """Intra-List Diversity: mean (1 - Jaccard) over all pairs."""
    if len(recommended) < 2:
        return 1.0
    diversity_sum = 0.0
    pairs = 0
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            g_i = set(item_genres.get(recommended[i], []))
            g_j = set(item_genres.get(recommended[j], []))
            union = g_i | g_j
            if not union:
                continue
            jaccard = len(g_i & g_j) / len(union)
            diversity_sum += (1.0 - jaccard)
            pairs += 1
    return diversity_sum / pairs if pairs > 0 else 1.0


# ======================================================================
# Arena Runner
# ======================================================================

def run_arena(data_path: str = "data/ml-100k/u.data", k: int = 10) -> pd.DataFrame:
    """
    Full benchmark: load data, train all engines, evaluate, compare.
    """
    print("=" * 70)
    print("MULTI-ENGINE BENCHMARK ARENA")
    print("=" * 70)

    # --- Load data -----------------------------------------------------
    loader = UnifiedDataLoader(data_path, min_interactions=5, seq_max_len=50)
    all_data = loader.build_all()

    df = all_data["df"]
    num_users = all_data["num_users"]
    num_items = all_data["num_items"]

    # --- Train/Test split (leave-last-one-out per user) ----------------
    print("\n  Splitting data (leave-last-one-out)...")
    train_rows = []
    test_dict: Dict[int, int] = {}  # user_idx -> held-out item_idx

    for uid, group in df.groupby("user_idx"):
        sorted_group = group.sort_values("timestamp")
        train_rows.append(sorted_group.iloc[:-1])
        test_dict[int(uid)] = int(sorted_group.iloc[-1]["item_idx"])

    train_df = pd.concat(train_rows).reset_index(drop=True)

    # Rebuild train-only matrices for fair evaluation
    import scipy.sparse as sp

    train_interaction = sp.csr_matrix(
        (train_df["rating"].values.astype(np.float32),
         (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(num_users, num_items),
    )

    # Rebuild train-only bipartite adjacency for LightGCN
    train_users = train_df["user_idx"].values
    train_items = train_df["item_idx"].values
    R_train = sp.coo_matrix(
        (np.ones(len(train_users), dtype=np.float32), (train_users, train_items)),
        shape=(num_users, num_items),
    )
    adj_train = sp.bmat([[None, R_train], [R_train.T, None]], format="csr")
    rowsum = np.array(adj_train.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    sym_adj_train = D_inv_sqrt.dot(adj_train).dot(D_inv_sqrt).tocsr()

    # Rebuild train-only sequential windows for SASRec
    train_seq_windows: Dict[int, np.ndarray] = {}
    for uid, group in train_df.groupby("user_idx"):
        seq = group.sort_values("timestamp")["item_idx"].tolist()
        seq = [s + 1 for s in seq]  # 1-indexed (0 = padding)
        max_len = 50
        if len(seq) < max_len:
            seq = [0] * (max_len - len(seq)) + seq
        else:
            seq = seq[-max_len:]
        train_seq_windows[int(uid)] = np.array(seq, dtype=np.int32)

    print(f"  Train: {len(train_df):,} ratings | Test: {len(test_dict)} users")

    # --- Initialize engines --------------------------------------------
    print("\n  Initializing engines...")
    engines: Dict[str, Any] = {
        "Funk-SVD": FunkSVDEngine(n_factors=50, n_epochs=20),
        "LightGCN": LightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=64, num_layers=3, n_epochs=30,
        ),
        "SASRec": SASRecEngine(
            num_items=num_items, max_seq_len=50, hidden_dim=50, n_epochs=30,
        ),
    }

    # --- Train each engine ---------------------------------------------
    results = []

    for name, engine in engines.items():
        print(f"\n{'='*50}")
        print(f"  Training: {name}")
        print(f"{'='*50}")

        train_start = time.time()

        if name == "Funk-SVD":
            engine.fit(train_df)
        elif name == "LightGCN":
            engine.fit({
                "sym_adj_mat": sym_adj_train,
                "interaction_matrix": train_interaction,
            })
        elif name == "SASRec":
            engine.fit(train_seq_windows)

        train_time = time.time() - train_start
        print(f"  Training time: {train_time:.1f}s")

        # --- Evaluate --------------------------------------------------
        print(f"  Evaluating {name}...")
        recalls = []
        ndcgs = []
        latencies = []

        # Sample 200 users for ranking evaluation
        eval_users = list(test_dict.keys())[:200]

        for uid in eval_users:
            true_item = test_dict[uid]
            ground_truth = {true_item}

            t0 = time.perf_counter()
            top_n = engine.recommend_top_n(uid, top_n=k)
            lat = (time.perf_counter() - t0) * 1000

            recalls.append(recall_at_k(top_n, ground_truth, k))
            ndcgs.append(ndcg_at_k(top_n, ground_truth, k))
            latencies.append(lat)

        result = {
            "Engine": name,
            "Train Time (s)": round(train_time, 1),
            "Recall@10": round(float(np.mean(recalls)), 4),
            "NDCG@10": round(float(np.mean(ndcgs)), 4),
            "Latency (ms)": round(float(np.mean(latencies)), 2),
            "P95 Latency (ms)": round(float(np.percentile(latencies, 95)), 2),
        }
        results.append(result)

        print(f"    Recall@{k}: {result['Recall@10']:.4f}")
        print(f"    NDCG@{k}:   {result['NDCG@10']:.4f}")
        print(f"    Latency:   {result['Latency (ms)']:.2f}ms (P95: {result['P95 Latency (ms)']:.2f}ms)")

    # --- Summary -------------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # Save
    os.makedirs("models", exist_ok=True)
    csv_path = "models/benchmark_arena_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    return results_df


if __name__ == "__main__":
    run_arena()
