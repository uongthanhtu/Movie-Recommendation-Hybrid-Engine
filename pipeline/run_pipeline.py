"""
Master Pipeline Orchestrator — Run the complete training pipeline.

Steps:
  1. Load data (from MovieLens files OR production database)
  2. Train SVD model (with GridSearch optimization)
  3. Benchmark SVD vs KNN vs NMF
  4. Generate Top-N recommendations per user
  5. Push results to Redis
  6. Save trained model to disk

Usage:
  # Demo mode (MovieLens files)
  python -m pipeline.run_pipeline                    # Full pipeline
  python -m pipeline.run_pipeline --skip-benchmark   # Skip algorithm comparison

  # Production mode (from database)
  python -m pipeline.run_pipeline --data-source database --db-path data/movies.db
  python -m pipeline.run_pipeline --data-source database --pg-url postgresql://user:pass@localhost/movie_db
"""
import os
import sys
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_all
from pipeline.etl_from_db import extract_from_sqlite, extract_from_postgres
from pipeline.train_svd import (
    build_surprise_data,
    train_svd,
    grid_search_svd,
    benchmark_all_algorithms,
    save_model,
)
from pipeline.generate_recommendations import generate_top_n, generate_popular_movies
from pipeline.push_to_redis import (
    get_redis_client,
    push_recommendations,
    push_popular,
    push_model_metadata,
    flush_old_recommendations,
)


def run_pipeline(
    data_dir: str = "data/ml-100k",
    model_dir: str = "models",
    db_path: str = "data/movies.db",
    top_n: int = 10,
    skip_benchmark: bool = False,
    skip_gridsearch: bool = False,
    no_redis: bool = False,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    cache_ttl: int = 86400,
    data_source: str = "movielens",  # "movielens" or "database"
    pg_url: str = None,
):
    """Execute the full training pipeline."""
    pipeline_start = time.time()

    print("=" * 70)
    print("MOVIE RECOMMENDATION PIPELINE")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 1: Loading Data")
    print("─" * 70)

    if data_source == "database":
        # Production mode: read from database
        if pg_url:
            print("  Source: PostgreSQL (production)")
            ratings, movies, stats = extract_from_postgres(pg_url)
        else:
            print(f"  Source: SQLite ({db_path})")
            ratings, movies, stats = extract_from_sqlite(db_path)
    else:
        # Demo mode: read from MovieLens files
        print(f"  Source: MovieLens files ({data_dir})")
        ratings, movies, stats = load_all(data_dir, save_db=True, db_path=db_path)

    data = build_surprise_data(ratings)

    # =========================================================================
    # STEP 2: Benchmark (optional)
    # =========================================================================
    benchmark_df = None
    if not skip_benchmark:
        print("\n" + "─" * 70)
        print("STEP 2: Algorithm Benchmark (SVD vs KNN vs NMF)")
        print("─" * 70)
        benchmark_df = benchmark_all_algorithms(data)

        # Save benchmark results
        os.makedirs(model_dir, exist_ok=True)
        benchmark_path = os.path.join(model_dir, "benchmark_results.csv")
        benchmark_df.to_csv(benchmark_path, index=False)
        print(f"\n  Benchmark saved: {benchmark_path}")
    else:
        print("\nSkipping benchmark (--skip-benchmark)")

    # =========================================================================
    # STEP 3: Train Best SVD Model
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 3: Training SVD Model")
    print("─" * 70)

    if not skip_gridsearch:
        print("  Running GridSearchCV for hyperparameter optimization...")
        best_algo, best_rmse, best_params, gs_results, search_time = grid_search_svd(data)

        # Save GridSearch results
        gs_path = os.path.join(model_dir, "gridsearch_results.csv")
        gs_results.to_csv(gs_path, index=False)
        print(f"  GridSearch results saved: {gs_path}")

        # Re-train on full data with best params
        algo, trainset, metrics = train_svd(
            data,
            n_factors=best_params["n_factors"],
            n_epochs=best_params["n_epochs"],
            lr_all=best_params["lr_all"],
            reg_all=best_params["reg_all"],
        )
    else:
        print("  Using default SVD params (--skip-gridsearch)")
        algo, trainset, metrics = train_svd(data)

    # Save model
    model_path = os.path.join(model_dir, "svd_model.pkl")
    save_model(algo, model_path)

    # Save model metadata locally
    import json
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"  Local model metadata saved: {metadata_path}")
    except Exception as e:
        print(f"  Failed to save model metadata: {e}")

    # =========================================================================
    # STEP 4: Generate Recommendations
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 4: Generating Recommendations")
    print("─" * 70)

    top_n_recs = generate_top_n(algo, trainset, n=top_n)
    popular = generate_popular_movies(ratings, movies, n=20)

    # =========================================================================
    # STEP 5: Push to Redis
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 5: Pushing to Redis")
    print("─" * 70)

    if no_redis:
        print("  Skipping Redis push (--no-redis)")
        n_pushed = 0
    else:
        r = get_redis_client(redis_host, redis_port, redis_db)
        if r:
            flush_old_recommendations(r)
            n_pushed = push_recommendations(r, top_n_recs, movies, ttl=cache_ttl)
            push_popular(r, popular, ttl=cache_ttl)
            push_model_metadata(r, metrics, n_pushed, ttl=cache_ttl)
        else:
            n_pushed = 0
            print("  Redis not available. Recommendations generated but not cached.")
            print("  Start Redis: docker-compose up redis -d")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    pipeline_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Dataset:        MovieLens-100k ({stats['n_ratings']:,} ratings)")
    print(f"   Algorithm:      SVD (n_factors={metrics.get('n_factors', '?')})")
    print(f"   RMSE:           {metrics['rmse_mean']}")
    print(f"   MAE:            {metrics['mae_mean']}")
    print(f"   Users served:   {len(top_n_recs):,}")
    print(f"   Redis cached:   {'Yes' if n_pushed > 0 else 'No'}")
    print(f"   Model saved:    {model_path}")
    print(f"   Total time:     {pipeline_time:.1f}s")
    print("=" * 70)

    return {
        "stats": stats,
        "metrics": metrics,
        "benchmark": benchmark_df,
        "top_n_count": len(top_n_recs),
        "redis_pushed": n_pushed,
        "pipeline_time": pipeline_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation Training Pipeline")

    # Data source options
    parser.add_argument("--data-source", choices=["movielens", "database"], default="movielens",
                        help="Data source: 'movielens' (CSV files) or 'database' (SQLite/PostgreSQL)")
    parser.add_argument("--data-dir", default="data/ml-100k", help="Path to MovieLens-100k directory")
    parser.add_argument("--db-path", default="data/movies.db", help="SQLite database path")
    parser.add_argument("--pg-url", default=None, help="PostgreSQL URL (for --data-source database)")

    # Pipeline options
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations per user")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip SVD vs KNN benchmark")
    parser.add_argument("--skip-gridsearch", action="store_true", help="Use default params instead of GridSearch")
    parser.add_argument("--no-redis", action="store_true", help="Skip Redis push")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--cache-ttl", type=int, default=86400, help="Redis cache TTL in seconds")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        db_path=args.db_path,
        top_n=args.top_n,
        skip_benchmark=args.skip_benchmark,
        skip_gridsearch=args.skip_gridsearch,
        no_redis=args.no_redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        cache_ttl=args.cache_ttl,
        data_source=args.data_source,
        pg_url=args.pg_url,
    )


if __name__ == "__main__":
    main()
