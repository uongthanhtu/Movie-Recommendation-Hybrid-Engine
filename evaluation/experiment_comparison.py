"""
Experiment: Algorithm Comparison — SVD vs KNN vs NMF full benchmark.

This experiment answers: "Why choose SVD over KNN/NMF?"
Output: Comparison table + chart for thesis Chapter 4.

Usage:
    python -m evaluation.experiment_comparison
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, SVDpp, KNNBaseline, KNNWithMeans, KNNBasic, NMF, BaselineOnly
from surprise.model_selection import cross_validate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_ratings
from pipeline.train_svd import build_surprise_data


def run_comparison_experiment(data_dir: str = "data/ml-100k", output_dir: str = "models"):
    """Run comprehensive algorithm comparison."""
    print("=" * 70)
    print("📊 EXPERIMENT 3: Algorithm Comparison (SVD vs KNN vs NMF)")
    print("=" * 70)

    ratings = load_ratings(data_dir)
    data = build_surprise_data(ratings)

    algorithms = {
        "BaselineOnly": BaselineOnly(verbose=False),
        "KNNBasic (user)": KNNBasic(k=40, sim_options={"name": "cosine", "user_based": True}, verbose=False),
        "KNNWithMeans (user)": KNNWithMeans(k=40, sim_options={"name": "cosine", "user_based": True}, verbose=False),
        "KNNBaseline (item)": KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False}, verbose=False),
        "NMF (K=100)": NMF(n_factors=100, n_epochs=50, verbose=False),
        "SVD (K=50)": SVD(n_factors=50, n_epochs=20, verbose=False),
        "SVD (K=100)": SVD(n_factors=100, n_epochs=20, verbose=False),
        "SVD (K=150)": SVD(n_factors=150, n_epochs=30, verbose=False),
    }

    results = []
    for name, algo in algorithms.items():
        print(f"\n  🔄 Testing: {name}...")
        start = time.time()
        cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
        elapsed = time.time() - start

        result = {
            "Algorithm": name,
            "RMSE": round(cv["test_rmse"].mean(), 4),
            "RMSE_std": round(cv["test_rmse"].std(), 4),
            "MAE": round(cv["test_mae"].mean(), 4),
            "Fit Time (s)": round(np.mean(cv["fit_time"]), 2),
            "Test Time (s)": round(np.mean(cv["test_time"]), 2),
            "Total CV Time (s)": round(elapsed, 1),
        }
        results.append(result)
        print(f"     RMSE: {result['RMSE']:.4f} ± {result['RMSE_std']:.4f} | "
              f"MAE: {result['MAE']:.4f} | Time: {elapsed:.1f}s")

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("📋 COMPLETE COMPARISON TABLE (sorted by RMSE)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved: {csv_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE comparison with error bars
    colors = []
    for name in df["Algorithm"]:
        if "SVD" in name:
            colors.append("#e74c3c")  # Red for SVD
        elif "KNN" in name:
            colors.append("#3498db")  # Blue for KNN
        elif "NMF" in name:
            colors.append("#f39c12")  # Orange for NMF
        else:
            colors.append("#95a5a6")  # Gray for baseline

    ax1.barh(df["Algorithm"], df["RMSE"], xerr=df["RMSE_std"],
             color=colors, alpha=0.8, capsize=4)
    ax1.set_xlabel("RMSE (lower is better)", fontsize=12)
    ax1.set_title("Algorithm Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Training time comparison
    ax2.barh(df["Algorithm"], df["Total CV Time (s)"], color=colors, alpha=0.8)
    ax2.set_xlabel("5-fold CV Time (seconds)", fontsize=12)
    ax2.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "experiment_comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"📈 Chart saved: {chart_path}")
    plt.close()

    return df


if __name__ == "__main__":
    run_comparison_experiment()
