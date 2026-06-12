"""
Experiment: Latency — Cache vs No-Cache performance comparison.

This experiment answers: "How much faster is Redis cache vs real-time SVD inference?"
Output: Comparison table + chart for the thesis report.

Usage:
    python -m evaluation.experiment_latency
"""
import os
import sys
import time
import json
import pickle
import statistics
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_ratings
from pipeline.train_svd import build_surprise_data, train_svd


def measure_realtime_inference(algo, trainset, user_ids: list, n: int = 10) -> list:
    """
    Measure latency of real-time SVD inference (NO cache).

    For each user: predict all unrated items → sort → take top-N.
    """
    latencies = []
    anti_testset = trainset.build_anti_testset()

    # Group anti-testset by user for per-user timing
    from collections import defaultdict
    user_items = defaultdict(list)
    for uid, iid, _ in anti_testset:
        user_items[uid].append(iid)

    for uid in user_ids:
        items = user_items.get(uid, [])
        if not items:
            continue

        start = time.perf_counter()
        predictions = [(iid, algo.predict(uid, iid).est) for iid in items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        elapsed = (time.perf_counter() - start) * 1000  # ms

        latencies.append(elapsed)

    return latencies


def measure_cache_latency(user_ids: list, redis_host: str = "localhost", redis_port: int = 6379) -> list:
    """
    Measure latency of Redis cache lookup.
    """
    import redis as redis_lib
    try:
        r = redis_lib.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        r.ping()
    except Exception:
        print("  ⚠️  Redis not available. Skipping cache latency test.")
        return []

    latencies = []
    for uid in user_ids:
        start = time.perf_counter()
        cached = r.get(f"reco:{uid}")
        if cached:
            data = json.loads(cached)
            _ = data["movies"][:10]
        elapsed = (time.perf_counter() - start) * 1000

        latencies.append(elapsed)

    return latencies


def run_latency_experiment(
    data_dir: str = "data/ml-100k",
    model_path: str = "models/svd_model.pkl",
    output_dir: str = "models",
    n_test_users: int = 50,
):
    """Run latency comparison experiment."""
    print("=" * 70)
    print("⏱️  EXPERIMENT 2: Latency — Cache vs Real-time Inference")
    print("=" * 70)

    # Load data and model
    ratings = load_ratings(data_dir)
    data = build_surprise_data(ratings)

    if os.path.exists(model_path):
        print(f"  Loading saved model: {model_path}")
        with open(model_path, "rb") as f:
            algo = pickle.load(f)
        trainset = data.build_full_trainset()
    else:
        print("  No saved model found. Training fresh SVD...")
        algo, trainset, _ = train_svd(data, verbose=False)

    # Select test users
    all_users = list(set(ratings["userId"].values))
    test_users = all_users[:n_test_users]
    print(f"  Testing with {len(test_users)} users...")

    # 1. Real-time inference latency
    print(f"\n  📐 Measuring real-time inference latency...")
    rt_latencies = measure_realtime_inference(algo, trainset, test_users)

    # 2. Redis cache latency
    print(f"  ⚡ Measuring Redis cache latency...")
    cache_latencies = measure_cache_latency(test_users)

    # Results
    results = {}

    if rt_latencies:
        results["realtime"] = {
            "mean_ms": round(statistics.mean(rt_latencies), 2),
            "median_ms": round(statistics.median(rt_latencies), 2),
            "p95_ms": round(sorted(rt_latencies)[int(len(rt_latencies) * 0.95)], 2),
            "p99_ms": round(sorted(rt_latencies)[int(len(rt_latencies) * 0.99)], 2),
            "min_ms": round(min(rt_latencies), 2),
            "max_ms": round(max(rt_latencies), 2),
        }

    if cache_latencies:
        results["cache"] = {
            "mean_ms": round(statistics.mean(cache_latencies), 2),
            "median_ms": round(statistics.median(cache_latencies), 2),
            "p95_ms": round(sorted(cache_latencies)[int(len(cache_latencies) * 0.95)], 2),
            "p99_ms": round(sorted(cache_latencies)[int(len(cache_latencies) * 0.99)], 2),
            "min_ms": round(min(cache_latencies), 2),
            "max_ms": round(max(cache_latencies), 2),
        }

    # Print comparison table
    print("\n" + "=" * 70)
    print("📋 LATENCY COMPARISON TABLE")
    print("=" * 70)

    if results.get("realtime") and results.get("cache"):
        rt = results["realtime"]
        ca = results["cache"]
        speedup = rt["mean_ms"] / max(ca["mean_ms"], 0.01)

        table_data = {
            "Metric": ["Mean", "Median", "P95", "P99", "Min", "Max"],
            "Real-time SVD (ms)": [rt["mean_ms"], rt["median_ms"], rt["p95_ms"], rt["p99_ms"], rt["min_ms"], rt["max_ms"]],
            "Redis Cache (ms)": [ca["mean_ms"], ca["median_ms"], ca["p95_ms"], ca["p99_ms"], ca["min_ms"], ca["max_ms"]],
        }
        df = pd.DataFrame(table_data)
        df["Speedup"] = (df["Real-time SVD (ms)"] / df["Redis Cache (ms)"].clip(lower=0.01)).round(1).astype(str) + "x"
        print(df.to_string(index=False))
        print(f"\n  🚀 Average Speedup: {speedup:.0f}x faster with Redis Cache!")
    elif results.get("realtime"):
        print("  Real-time inference measured, but Redis not available for comparison.")
        print(f"  Mean latency: {results['realtime']['mean_ms']:.2f}ms")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_latency_results.csv")

    if results.get("realtime") and results.get("cache"):
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved: {csv_path}")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot comparison
        ax1.boxplot([rt_latencies, cache_latencies],
                    labels=["Real-time SVD", "Redis Cache"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.7))
        ax1.set_ylabel("Latency (ms)", fontsize=12)
        ax1.set_title("Response Time Distribution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Bar chart
        metrics = ["Mean", "Median", "P95", "P99"]
        rt_vals = [rt["mean_ms"], rt["median_ms"], rt["p95_ms"], rt["p99_ms"]]
        ca_vals = [ca["mean_ms"], ca["median_ms"], ca["p95_ms"], ca["p99_ms"]]
        x = range(len(metrics))
        width = 0.35

        bars1 = ax2.bar([i - width/2 for i in x], rt_vals, width, label="Real-time SVD", color="#e74c3c", alpha=0.8)
        bars2 = ax2.bar([i + width/2 for i in x], ca_vals, width, label="Redis Cache", color="#2ecc71", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.set_ylabel("Latency (ms)", fontsize=12)
        ax2.set_title("Latency Comparison", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "experiment_latency_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"📈 Chart saved: {chart_path}")
        plt.close()

    return results


if __name__ == "__main__":
    run_latency_experiment()
