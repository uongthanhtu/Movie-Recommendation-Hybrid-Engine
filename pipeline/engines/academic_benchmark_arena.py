"""
Academic Benchmark Arena — Standalone sandbox evaluator.
Compares TrustSVD (Baseline) and Social-LightGCN (Ours) on Epinions/Ciao.
"""
import os
import sys
import time
from typing import Tuple, Dict, Any, List, Set

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Insert parent dir to import pipeline components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.engines.academic_data_loader import AcademicDataLoader
from pipeline.engines.trust_svd_engine import TrustSVDEngine
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine


def generate_dummy_epinions(ratings_path: str, trust_path: str):
    """Generate mock ratings and trust data for Epinions when files do not exist."""
    os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
    rng = np.random.default_rng(42)

    # 100 users, 200 items, 1500 ratings
    users = rng.integers(1, 101, size=1500)
    items = rng.integers(1, 201, size=1500)
    ratings = rng.integers(1, 6, size=1500)

    df_ratings = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "rating": ratings
    }).drop_duplicates(subset=["user_id", "item_id"])

    df_ratings.to_csv(ratings_path, sep="\t", index=False, header=False)

    # 600 trust links
    sources = rng.integers(1, 101, size=600)
    targets = rng.integers(1, 101, size=600)
    trust_vals = rng.uniform(0.1, 1.0, size=600)

    df_trust = pd.DataFrame({
        "source_id": sources,
        "target_id": targets,
        "trust_val": trust_vals
    }).drop_duplicates(subset=["source_id", "target_id"])

    # Remove self loops
    df_trust = df_trust[df_trust["source_id"] != df_trust["target_id"]]

    df_trust.to_csv(trust_path, sep="\t", index=False, header=False)
    print(f"  Generated dummy Epinions dataset at:\n   - {ratings_path}\n   - {trust_path}")


def evaluate_ranking(engine: Any, test_ratings: pd.DataFrame, num_items: int, k: int = 10) -> Tuple[float, float]:
    """Calculate average Recall@K and NDCG@K on test data."""
    test_dict: Dict[int, Set[int]] = {}
    for _, row in test_ratings.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        if u not in test_dict:
            test_dict[u] = set()
        test_dict[u].add(i)

    recalls = []
    ndcgs = []

    for u, ground_truth in test_dict.items():
        if not ground_truth:
            continue

        # Get top-N items
        recommended = engine.recommend_top_n(u, top_n=k)

        # Recall@K
        hits = len(set(recommended[:k]) & ground_truth)
        recalls.append(hits / len(ground_truth))

        # NDCG@K
        dcg = sum(
            1.0 / np.log2(p + 2)
            for p, item in enumerate(recommended[:k])
            if item in ground_truth
        )
        idcg = sum(1.0 / np.log2(p + 2) for p in range(min(len(ground_truth), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def evaluate_ratings(engine: Any, test_ratings: pd.DataFrame) -> Tuple[float, float]:
    """Calculate RMSE and MAE on test ratings."""
    predictions = []
    for _, row in test_ratings.iterrows():
        u = int(row["user_idx"])
        i = int(row["item_idx"])
        r = float(row["rating"])

        pred = engine.predict_rating(u, i)
        predictions.append((r, pred))

    if not predictions:
        return 0.0, 0.0

    sq_errors = [(a - p) ** 2 for a, p in predictions]
    abs_errors = [abs(a - p) for a, p in predictions]

    rmse = float(np.sqrt(np.mean(sq_errors)))
    mae = float(np.mean(abs_errors))
    return rmse, mae


def run_academic_benchmark(
    ratings_path: str = "data/epinions/ratings.txt",
    trust_path: str = "data/epinions/trust.txt"
):
    """Orchestrate training and evaluations."""
    print("=" * 70)
    print("ACADEMIC BENCHMARK SANDBOX - EPINIONS/CIAO")
    print("=" * 70)

    # Auto generate mock data if files are missing
    if not os.path.exists(ratings_path) or not os.path.exists(trust_path):
        print("  Dataset files not found. Auto generating mock Epinions data...")
        generate_dummy_epinions(ratings_path, trust_path)

    # 1. Load Data
    loader = AcademicDataLoader(ratings_path, trust_path, threshold=3.0)
    df_ratings, df_trust = loader.load_data()

    num_users = loader.num_users
    num_items = loader.num_items

    # 2. Split ratings (80/20 split)
    df_train, df_test = loader.split_data(df_ratings, ratio=0.8, seed=42)
    print(f"  Splitting ratings: Train={len(df_train):,} | Test={len(df_test):,}")

    # Build CSR matrices
    train_interaction = sp.csr_matrix(
        (df_train["rating"].values.astype(np.float32),
         (df_train["user_idx"].values, df_train["item_idx"].values)),
        shape=(num_users, num_items),
    )

    trust_matrix = loader.build_social_matrix(df_trust)

    # 3. Initialize Engines
    engines = {
        "TrustSVD (Baseline)": TrustSVDEngine(
            n_factors=32, 
            n_epochs=20, 
            lr=0.005, 
            reg=0.02
        ),
        "Social-LightGCN (Ours)": SocialLightGCNEngine(
            num_users=num_users, 
            num_items=num_items, 
            embedding_dim=32, 
            num_layers=3, 
            n_epochs=20, 
            batch_size=256
        )
    }

    # 4. Train & Evaluate Loop
    results = []

    for name, engine in engines.items():
        print(f"\nTraining {name}...")
        t_start = time.time()

        # Fit model
        engine.fit({
            "interaction_matrix": train_interaction,
            "trust_matrix": trust_matrix
        })

        train_time = time.time() - t_start
        print(f"  Training time: {train_time:.2f}s")

        # Evaluate ranking (Recall@10 & NDCG@10)
        print(f"  Evaluating ranking metrics for {name}...")
        recall, ndcg = evaluate_ranking(engine, df_test, num_items, k=10)

        # Evaluate ratings (RMSE & MAE)
        print(f"  Evaluating error metrics for {name}...")
        rmse_val, mae_val = evaluate_ratings(engine, df_test)

        results.append({
            "Engine": name,
            "Train Time (s)": round(train_time, 2),
            "Recall@10": round(recall, 4),
            "NDCG@10": round(ndcg, 4),
            "RMSE": round(rmse_val, 4),
            "MAE": round(mae_val, 4),
        })

    # 5. Output Summary Results Table
    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SANDBOX BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    # Allow overriding paths via command line args
    r_path = sys.argv[1] if len(sys.argv) > 1 else "data/epinions/ratings.txt"
    t_path = sys.argv[2] if len(sys.argv) > 2 else "data/epinions/trust.txt"
    run_academic_benchmark(r_path, t_path)
