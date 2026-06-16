"""
Data Loader — Load and preprocess MovieLens-100k dataset.

Supports both CSV (raw files) and SQLite storage for movie metadata.

Dataset structure:
  - u.data:  100,000 ratings (userId, movieId, rating, timestamp)
  - u.item:  1,682 movies (movieId, title, release_date, ..., 19 genre columns)
  - u.user:  943 users (userId, age, gender, occupation, zip_code)
"""
import os
import sqlite3
import pandas as pd
import numpy as np


# MovieLens-100k genre columns in order
GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

# Column names for u.item (pipe-separated, 24 columns)
ITEM_COLUMNS = [
    "movieId", "title", "release_date", "video_release_date", "imdb_url",
] + GENRE_COLUMNS


def load_ratings(data_dir: str = "data/ml-100k") -> pd.DataFrame:
    """
    Load ratings from u.data (tab-separated).

    Returns DataFrame with columns: userId, movieId, rating, timestamp
    """
    path = os.path.join(data_dir, "u.data")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Rating file not found: {path}\n"
            f"Please download MovieLens-100k from https://grouplens.org/datasets/movielens/100k/"
        )

    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    )
    return ratings


def load_movies(data_dir: str = "data/ml-100k") -> pd.DataFrame:
    """
    Load movie metadata from u.item (pipe-separated, latin-1 encoding).

    Returns DataFrame with columns: movieId, title, release_date, genres (list), genre_str
    """
    path = os.path.join(data_dir, "u.item")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Movie file not found: {path}\n"
            f"Please download MovieLens-100k from https://grouplens.org/datasets/movielens/100k/"
        )

    movies_raw = pd.read_csv(
        path,
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
        usecols=range(len(ITEM_COLUMNS)),
    )

    # Build genres list for each movie
    genre_cols = GENRE_COLUMNS
    movies_raw["genres"] = movies_raw[genre_cols].apply(
        lambda row: [g for g, val in zip(genre_cols, row) if val == 1], axis=1
    )
    movies_raw["genre_str"] = movies_raw["genres"].apply(lambda x: "|".join(x))

    # Keep only useful columns
    movies = movies_raw[["movieId", "title", "release_date", "genres", "genre_str"]].copy()
    return movies


def load_users(data_dir: str = "data/ml-100k") -> pd.DataFrame:
    """Load user demographics from u.user (pipe-separated)."""
    path = os.path.join(data_dir, "u.user")
    if not os.path.exists(path):
        return pd.DataFrame()

    users = pd.read_csv(
        path,
        sep="|",
        names=["userId", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )
    return users


def compute_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Compute dataset statistics for reporting."""
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    total_cells = n_users * n_movies
    sparsity = 1 - (n_ratings / total_cells)

    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "total_cells": total_cells,
        "sparsity": sparsity,
        "sparsity_pct": f"{sparsity:.1%}",
        "avg_rating": ratings["rating"].mean(),
        "rating_std": ratings["rating"].std(),
        "avg_ratings_per_user": n_ratings / n_users,
        "avg_ratings_per_movie": n_ratings / n_movies,
        "rating_distribution": ratings["rating"].value_counts().sort_index().to_dict(),
    }


def save_to_sqlite(
    ratings: pd.DataFrame, movies: pd.DataFrame, db_path: str = "data/movies.db"
):
    """
    Save ratings and movies to SQLite for persistent storage.

    Creates tables: movies, ratings
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Movies table (flatten genres to string)
    movies_db = movies[["movieId", "title", "release_date", "genre_str"]].copy()
    movies_db.columns = ["movie_id", "title", "release_date", "genres"]
    movies_db.to_sql("movies", conn, if_exists="replace", index=False)

    # Ratings table
    ratings_db = ratings.copy()
    ratings_db.columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_db.to_sql("ratings", conn, if_exists="replace", index=False)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_id ON movies(movie_id)")

    conn.commit()
    conn.close()
    print(f"   Saved to SQLite: {db_path}")


def load_all(data_dir: str = "data/ml-100k", save_db: bool = True, db_path: str = "data/movies.db"):
    """
    Master load function — loads all data, prints stats, optionally saves to SQLite.

    Returns:
        ratings (DataFrame), movies (DataFrame), stats (dict)
    """
    print(" Loading MovieLens-100k dataset...")
    print(f"   Directory: {os.path.abspath(data_dir)}")

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    stats = compute_stats(ratings, movies)

    print(f"\n   Dataset Statistics:")
    print(f"   ├── Users:          {stats['n_users']:,}")
    print(f"   ├── Movies:         {stats['n_movies']:,}")
    print(f"   ├── Ratings:        {stats['n_ratings']:,}")
    print(f"   ├── Sparsity:       {stats['sparsity_pct']}")
    print(f"   ├── Avg Rating:     {stats['avg_rating']:.2f} ± {stats['rating_std']:.2f}")
    print(f"   ├── Avg/User:       {stats['avg_ratings_per_user']:.1f} ratings")
    print(f"   └── Avg/Movie:      {stats['avg_ratings_per_movie']:.1f} ratings")
    print(f"\n   Rating Distribution: {stats['rating_distribution']}")

    if save_db:
        save_to_sqlite(ratings, movies, db_path)

    return ratings, movies, stats


if __name__ == "__main__":
    ratings, movies, stats = load_all()
    print(f"\n Data loaded successfully!")
    print(f"   Ratings shape: {ratings.shape}")
    print(f"   Movies shape:  {movies.shape}")
