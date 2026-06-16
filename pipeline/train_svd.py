"""
SVD & KNN Trainer — Train recommendation models using Surprise library.

Supports:
  - SVD (Matrix Factorization) — primary algorithm
  - KNNBaseline — benchmark comparison
  - NMF (Non-negative MF) — additional comparison
  - GridSearchCV for hyperparameter tuning
  - 5-fold Cross Validation
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from surprise import SVD, SVDpp, KNNBaseline, KNNWithMeans, NMF, BaselineOnly
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV


def build_surprise_data(ratings: pd.DataFrame) -> Dataset:
    """Convert pandas DataFrame to Surprise Dataset."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]], reader
    )
    return data


def train_svd(
    data: Dataset,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    verbose: bool = True,
) -> tuple:
    """
    Train SVD model with cross-validation.

    Returns:
        algo: trained SVD algorithm
        trainset: full training set
        cv_results: cross-validation results dict
    """
    algo = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        verbose=False,
    )

    if verbose:
        print(f"\n Training SVD (n_factors={n_factors}, epochs={n_epochs}, lr={lr_all}, reg={reg_all})...")

    # Cross-validate
    start = time.time()
    cv_results = cross_validate(
        algo, data, measures=["RMSE", "MAE"], cv=5, verbose=verbose
    )
    cv_time = time.time() - start

    mean_rmse = cv_results["test_rmse"].mean()
    std_rmse = cv_results["test_rmse"].std()
    mean_mae = cv_results["test_mae"].mean()

    if verbose:
        print(f"   RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"   MAE:  {mean_mae:.4f}")
        print(f"   Time: {cv_time:.1f}s")

    # Train on full dataset
    trainset = data.build_full_trainset()
    start = time.time()
    algo.fit(trainset)
    train_time = time.time() - start

    if verbose:
        print(f"   Full train time: {train_time:.1f}s")

    return algo, trainset, {
        "algorithm": "SVD",
        "n_factors": n_factors,
        "n_epochs": n_epochs,
        "lr_all": lr_all,
        "reg_all": reg_all,
        "rmse_mean": round(mean_rmse, 4),
        "rmse_std": round(std_rmse, 4),
        "mae_mean": round(mean_mae, 4),
        "cv_time_seconds": round(cv_time, 1),
        "train_time_seconds": round(train_time, 1),
    }


def train_knn_baseline(data: Dataset, verbose: bool = True) -> tuple:
    """
    Train KNNBaseline (benchmark comparison).

    Uses item-based similarity with Pearson correlation baseline.
    """
    sim_options = {
        "name": "pearson_baseline",
        "user_based": False,  # item-based
    }
    algo = KNNBaseline(k=40, sim_options=sim_options, verbose=False)

    if verbose:
        print(f"\n Training KNNBaseline (k=40, item-based, pearson_baseline)...")

    start = time.time()
    cv_results = cross_validate(
        algo, data, measures=["RMSE", "MAE"], cv=5, verbose=verbose
    )
    cv_time = time.time() - start

    mean_rmse = cv_results["test_rmse"].mean()
    std_rmse = cv_results["test_rmse"].std()
    mean_mae = cv_results["test_mae"].mean()

    if verbose:
        print(f"   RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"   MAE:  {mean_mae:.4f}")
        print(f"   Time: {cv_time:.1f}s")

    # Train on full dataset
    trainset = data.build_full_trainset()
    start = time.time()
    algo.fit(trainset)
    train_time = time.time() - start

    return algo, trainset, {
        "algorithm": "KNNBaseline",
        "k": 40,
        "similarity": "pearson_baseline",
        "user_based": False,
        "rmse_mean": round(mean_rmse, 4),
        "rmse_std": round(std_rmse, 4),
        "mae_mean": round(mean_mae, 4),
        "cv_time_seconds": round(cv_time, 1),
        "train_time_seconds": round(train_time, 1),
    }


def train_nmf(data: Dataset, n_factors: int = 100, verbose: bool = True) -> tuple:
    """Train NMF model (additional benchmark)."""
    algo = NMF(n_factors=n_factors, n_epochs=50, verbose=False)

    if verbose:
        print(f"\n Training NMF (n_factors={n_factors})...")

    start = time.time()
    cv_results = cross_validate(
        algo, data, measures=["RMSE", "MAE"], cv=5, verbose=verbose
    )
    cv_time = time.time() - start

    mean_rmse = cv_results["test_rmse"].mean()
    mean_mae = cv_results["test_mae"].mean()

    if verbose:
        print(f"   RMSE: {mean_rmse:.4f}")
        print(f"   MAE:  {mean_mae:.4f}")
        print(f"   Time: {cv_time:.1f}s")

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    return algo, trainset, {
        "algorithm": "NMF",
        "n_factors": n_factors,
        "rmse_mean": round(mean_rmse, 4),
        "mae_mean": round(mean_mae, 4),
        "cv_time_seconds": round(cv_time, 1),
    }


def grid_search_svd(data: Dataset, verbose: bool = True) -> tuple:
    """
    GridSearchCV for SVD hyperparameter optimization.

    Returns best estimator and results DataFrame.
    """
    param_grid = {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.05],
    }

    if verbose:
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        print(f"\n GridSearch SVD ({total_combos} combinations × 5 folds)...")

    start = time.time()
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=-1)
    gs.fit(data)
    search_time = time.time() - start

    best_rmse = gs.best_score["rmse"]
    best_params = gs.best_params["rmse"]

    if verbose:
        print(f"   Best RMSE:   {best_rmse:.4f}")
        print(f"   Best Params: {best_params}")
        print(f"   Search Time: {search_time:.1f}s")

    # Get all results as DataFrame for reporting
    results_df = pd.DataFrame.from_dict(gs.cv_results)

    return gs.best_estimator["rmse"], best_rmse, best_params, results_df, search_time


def benchmark_all_algorithms(data: Dataset) -> pd.DataFrame:
    """
    Run all algorithms and return comparison DataFrame.

    For the thesis: SVD vs KNN vs NMF benchmark table.
    """
    print("\n" + "=" * 70)
    print(" BENCHMARK: SVD vs KNNBaseline vs NMF")
    print("=" * 70)

    results = []

    # 1. SVD
    _, _, svd_info = train_svd(data, n_factors=100)
    results.append(svd_info)

    # 2. KNNBaseline
    _, _, knn_info = train_knn_baseline(data)
    results.append(knn_info)

    # 3. NMF
    _, _, nmf_info = train_nmf(data, n_factors=100)
    results.append(nmf_info)

    # 4. BaselineOnly (simple baseline)
    print(f"\n Training BaselineOnly...")
    algo_base = BaselineOnly(verbose=False)
    start = time.time()
    cv_base = cross_validate(algo_base, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    base_time = time.time() - start
    results.append({
        "algorithm": "BaselineOnly",
        "rmse_mean": round(cv_base["test_rmse"].mean(), 4),
        "mae_mean": round(cv_base["test_mae"].mean(), 4),
        "cv_time_seconds": round(base_time, 1),
    })

    # 5. KNNWithMeans (user-based)
    print(f"\n Training KNNWithMeans (user-based)...")
    algo_knn2 = KNNWithMeans(k=40, sim_options={"name": "cosine", "user_based": True}, verbose=False)
    start = time.time()
    cv_knn2 = cross_validate(algo_knn2, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    knn2_time = time.time() - start
    results.append({
        "algorithm": "KNNWithMeans (user-based)",
        "rmse_mean": round(cv_knn2["test_rmse"].mean(), 4),
        "mae_mean": round(cv_knn2["test_mae"].mean(), 4),
        "cv_time_seconds": round(knn2_time, 1),
    })

    df = pd.DataFrame(results)
    df = df.sort_values("rmse_mean").reset_index(drop=True)

    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS (sorted by RMSE)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return df


def save_model(algo, filepath: str = "models/svd_model.pkl"):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(algo, f)
    print(f"   Model saved: {filepath}")


def load_model(filepath: str = "models/svd_model.pkl"):
    """Load trained model from disk."""
    with open(filepath, "rb") as f:
        algo = pickle.load(f)
    return algo
