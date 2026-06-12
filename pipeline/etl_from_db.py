"""
ETL from PostgreSQL — Extract ratings from production DB for SVD training.

This module replaces the raw MovieLens file loader when running in production.
It reads from PostgreSQL (or SQLite) and outputs the same DataFrame format
that the SVD pipeline expects.

Usage:
    # From PostgreSQL
    python -m pipeline.etl_from_db --source postgres --pg-url postgresql://user:pass@localhost:5432/movie_db

    # From SQLite (local dev)
    python -m pipeline.etl_from_db --source sqlite --db-path data/movies.db
"""
import os
import sys
import argparse
import sqlite3
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_from_sqlite(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Extract ratings and movies from SQLite (production schema).

    Returns same format as data_loader.load_all():
        ratings (DataFrame), movies (DataFrame), stats (dict)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    # Extract ratings
    ratings = pd.read_sql(
        "SELECT user_id AS userId, movie_id AS movieId, rating, "
        "CAST(strftime('%s', created_at) AS INTEGER) AS timestamp "
        "FROM movie_ratings ORDER BY created_at",
        conn,
    )

    # Extract movies
    movies_raw = pd.read_sql(
        "SELECT id AS movieId, title, release_date, genres AS genre_str, "
        "poster_url, overview, tmdb_id FROM movies ORDER BY id",
        conn,
    )

    conn.close()

    # Build genres list (same format as data_loader)
    movies_raw["genres"] = movies_raw["genre_str"].apply(
        lambda x: x.split("|") if pd.notna(x) and x else []
    )

    movies = movies_raw[["movieId", "title", "release_date", "genres", "genre_str"]].copy()

    # Compute stats
    stats = _compute_stats(ratings, movies)

    _print_stats(stats, "SQLite", db_path)

    return ratings, movies, stats


def extract_from_postgres(pg_url: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Extract ratings and movies from PostgreSQL (production).

    Returns same format as data_loader.load_all().
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(pg_url)

    # Extract ratings
    ratings = pd.read_sql(
        """SELECT user_id AS "userId", movie_id AS "movieId", rating,
           EXTRACT(EPOCH FROM created_at)::int AS timestamp
           FROM movie_ratings ORDER BY created_at""",
        conn,
    )

    # Extract movies
    movies_raw = pd.read_sql(
        """SELECT id AS "movieId", title, release_date, genres AS genre_str,
           poster_url, overview, tmdb_id FROM movies ORDER BY id""",
        conn,
    )

    conn.close()

    movies_raw["genres"] = movies_raw["genre_str"].apply(
        lambda x: x.split("|") if pd.notna(x) and x else []
    )
    movies = movies_raw[["movieId", "title", "release_date", "genres", "genre_str"]].copy()

    stats = _compute_stats(ratings, movies)
    _print_stats(stats, "PostgreSQL", pg_url.split("@")[-1] if "@" in pg_url else pg_url)

    return ratings, movies, stats


def _compute_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Compute dataset statistics."""
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    total_cells = max(n_users * n_movies, 1)
    sparsity = 1 - (n_ratings / total_cells)

    # Count by source if available
    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "total_cells": total_cells,
        "sparsity": sparsity,
        "sparsity_pct": f"{sparsity:.1%}",
        "avg_rating": ratings["rating"].mean() if n_ratings > 0 else 0,
        "rating_std": ratings["rating"].std() if n_ratings > 0 else 0,
        "avg_ratings_per_user": n_ratings / max(n_users, 1),
        "avg_ratings_per_movie": n_ratings / max(n_movies, 1),
    }


def _print_stats(stats: dict, source_type: str, source_path: str):
    """Print dataset statistics."""
    print(f"\n📊 ETL from {source_type}: {source_path}")
    print(f"   ├── Users:     {stats['n_users']:,}")
    print(f"   ├── Movies:    {stats['n_movies']:,}")
    print(f"   ├── Ratings:   {stats['n_ratings']:,}")
    print(f"   ├── Sparsity:  {stats['sparsity_pct']}")
    if stats['n_ratings'] > 0:
        print(f"   ├── Avg Rating: {stats['avg_rating']:.2f} ± {stats['rating_std']:.2f}")
        print(f"   ├── Avg/User:   {stats['avg_ratings_per_user']:.1f} ratings")
        print(f"   └── Avg/Movie:  {stats['avg_ratings_per_movie']:.1f} ratings")


def main():
    parser = argparse.ArgumentParser(description="ETL: Extract ratings from production DB")
    parser.add_argument("--source", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--db-path", default="data/movies.db")
    parser.add_argument("--pg-url", default=None)
    args = parser.parse_args()

    if args.source == "sqlite":
        ratings, movies, stats = extract_from_sqlite(args.db_path)
    else:
        if not args.pg_url:
            print("ERROR: --pg-url required")
            sys.exit(1)
        ratings, movies, stats = extract_from_postgres(args.pg_url)

    print(f"\n✅ ETL complete!")
    print(f"   Ratings shape: {ratings.shape}")
    print(f"   Movies shape:  {movies.shape}")
    print(f"\n   Sample ratings:")
    print(ratings.head().to_string(index=False))


if __name__ == "__main__":
    main()
