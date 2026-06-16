"""
Experiment: RMSE vs n_factors — measure SVD accuracy with different latent factor counts.

This experiment answers: "What is the optimal number of latent factors for SVD?"
Output: Table + matplotlib chart for the thesis report.

Usage:
    python -m evaluation.experiment_rmse
"""
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_ratings
from pipeline.train_svd import build_surprise_data, train_svd


def run_rmse_experiment(data_dir: str = "data/ml-100k", output_dir: str = "models"):
    """Run RMSE experiment with varying n_factors."""
    print("=" * 70)
    print(" EXPERIMENT 1: RMSE vs n_factors")
    print("=" * 70)

    ratings = load_ratings(data_dir)
    data = build_surprise_data(ratings)

    # Parameter configurations to test
    configs = [
        {"n_factors": 10,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 20,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 50,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 75,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 150, "n_epochs": 30, "lr_all": 0.005, "reg_all": 0.02},
        {"n_factors": 200, "n_epochs": 30, "lr_all": 0.005, "reg_all": 0.02},
    ]

    results = []
    for cfg in configs:
        _, _, metrics = train_svd(data, **cfg)
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" RESULTS TABLE")
    print("=" * 70)
    print(df[["n_factors", "n_epochs", "rmse_mean", "rmse_std", "mae_mean", "cv_time_seconds"]].to_string(index=False))

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_rmse_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n Results saved: {csv_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE vs n_factors
    ax1.errorbar(df["n_factors"], df["rmse_mean"], yerr=df["rmse_std"],
                 marker="o", capsize=5, linewidth=2, color="#e74c3c", markersize=8)
    ax1.set_xlabel("Number of Latent Factors (K)", fontsize=12)
    ax1.set_ylabel("RMSE (5-fold CV)", fontsize=12)
    ax1.set_title("RMSE vs Number of Latent Factors", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Highlight best
    best_idx = df["rmse_mean"].idxmin()
    ax1.scatter(df.loc[best_idx, "n_factors"], df.loc[best_idx, "rmse_mean"],
                color="green", s=200, zorder=5, label=f"Best: K={int(df.loc[best_idx, 'n_factors'])}")
    ax1.legend(fontsize=11)

    # Training time vs n_factors
    ax2.bar(df["n_factors"].astype(str), df["cv_time_seconds"], color="#3498db", alpha=0.8)
    ax2.set_xlabel("Number of Latent Factors (K)", fontsize=12)
    ax2.set_ylabel("Cross-Validation Time (s)", fontsize=12)
    ax2.set_title("Training Time vs Latent Factors", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "experiment_rmse_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f" Chart saved: {chart_path}")
    plt.close()

    return df


if __name__ == "__main__":
    run_rmse_experiment()
