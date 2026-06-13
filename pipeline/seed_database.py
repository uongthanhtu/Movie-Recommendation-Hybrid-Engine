"""
Data Adapter — Import MovieLens-100k into PostgreSQL schema for production use.

This script:
  1. Reads MovieLens-100k raw files (u.data, u.item, u.user)
  2. Creates proper PostgreSQL tables (movies, users, movie_ratings)
  3. Seeds the database with MovieLens data
  4. Can also export to SQLite for local dev

Usage:
    # Seed PostgreSQL (production)
    python -m pipeline.seed_database --target postgres --pg-url postgresql://user:pass@localhost:5432/movie_db

    # Seed SQLite (local dev / demo)
    python -m pipeline.seed_database --target sqlite --db-path data/movies.db
"""
import os
import sys
import argparse
import sqlite3
import time
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_ratings, load_movies, load_users, compute_stats


# =============================================================================
# Schema Definition
# =============================================================================

POSTGRES_SCHEMA = """
-- Movies catalog
CREATE TABLE IF NOT EXISTS movies (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    release_date    VARCHAR(50),
    genres          TEXT DEFAULT '',
    poster_url      VARCHAR(500),
    overview        TEXT,
    tmdb_id         INTEGER,
    source          VARCHAR(20) DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Users (maps to dự án cha's user table)
CREATE TABLE IF NOT EXISTS recommendation_users (
    id              SERIAL PRIMARY KEY,
    external_id     BIGINT UNIQUE,  -- maps to Spring Boot user id
    age             INTEGER,
    gender          VARCHAR(10),
    occupation      VARCHAR(50),
    source          VARCHAR(20) DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Explicit ratings (1-5 stars)
CREATE TABLE IF NOT EXISTS movie_ratings (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    movie_id        INTEGER NOT NULL,
    rating          DECIMAL(2,1) NOT NULL CHECK (rating >= 1 AND rating <= 5),
    source          VARCHAR(20) DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_movie_ratings_user ON movie_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_movie_ratings_movie ON movie_ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movies_source ON movies(source);
"""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    release_date    TEXT,
    genres          TEXT DEFAULT '',
    poster_url      TEXT,
    overview        TEXT,
    tmdb_id         INTEGER,
    source          TEXT DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendation_users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id     INTEGER UNIQUE,
    age             INTEGER,
    gender          TEXT,
    occupation      TEXT,
    source          TEXT DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movie_ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    movie_id        INTEGER NOT NULL,
    rating          REAL NOT NULL CHECK (rating >= 1 AND rating <= 5),
    source          TEXT DEFAULT 'movielens',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_movie_ratings_user ON movie_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_movie_ratings_movie ON movie_ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movies_source ON movies(source);
"""


# =============================================================================
# Seed Functions
# =============================================================================

def seed_sqlite(data_dir: str, db_path: str):
    """Seed SQLite database with MovieLens-100k data."""
    print("=" * 70)
    print("SEEDING SQLite DATABASE")
    print(f"   Source:  {os.path.abspath(data_dir)}")
    print(f"   Target:  {os.path.abspath(db_path)}")
    print("=" * 70)

    start = time.time()

    # Load raw data
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)
    stats = compute_stats(ratings, movies)

    # Create database
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Create schema
    for stmt in SQLITE_SCHEMA.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    # Clear old movielens data (keep user-generated data)
    conn.execute("DELETE FROM movie_ratings WHERE source = 'movielens'")
    conn.execute("DELETE FROM movies WHERE source = 'movielens'")
    conn.execute("DELETE FROM recommendation_users WHERE source = 'movielens'")

    # Insert movies
    print("\n  Inserting movies...")
    movies_data = []
    for _, row in movies.iterrows():
        movies_data.append((
            int(row["movieId"]),
            row["title"],
            row.get("release_date", None),
            row["genre_str"] if pd.notna(row.get("genre_str")) else "",
            None,  # poster_url
            None,  # overview
            None,  # tmdb_id
            "movielens",
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO movies (id, title, release_date, genres, poster_url, overview, tmdb_id, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        movies_data,
    )
    print(f"     Inserted {len(movies_data):,} movies")

    # Insert users
    if not users.empty:
        print("  Inserting users...")
        users_data = []
        for _, row in users.iterrows():
            users_data.append((
                int(row["userId"]),
                int(row["userId"]),  # external_id = userId for demo
                int(row["age"]) if pd.notna(row.get("age")) else None,
                row.get("gender", None),
                row.get("occupation", None),
                "movielens",
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO recommendation_users (id, external_id, age, gender, occupation, source) VALUES (?, ?, ?, ?, ?, ?)",
            users_data,
        )
        print(f"     Inserted {len(users_data):,} users")

    # Insert ratings
    print("  Inserting ratings...")
    ratings_data = []
    for _, row in ratings.iterrows():
        ts = datetime.fromtimestamp(row["timestamp"]).isoformat() if pd.notna(row.get("timestamp")) else None
        ratings_data.append((
            int(row["userId"]),
            int(row["movieId"]),
            float(row["rating"]),
            "movielens",
            ts,
            ts,
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO movie_ratings (user_id, movie_id, rating, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ratings_data,
    )
    print(f"     Inserted {len(ratings_data):,} ratings")

    conn.commit()

    # Verify
    n_movies = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    n_users = conn.execute("SELECT COUNT(*) FROM recommendation_users").fetchone()[0]
    n_ratings = conn.execute("SELECT COUNT(*) FROM movie_ratings").fetchone()[0]
    conn.close()

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print("SEED COMPLETE!")
    print(f"   Movies:   {n_movies:,}")
    print(f"   Users:    {n_users:,}")
    print(f"   Ratings:  {n_ratings:,}")
    print(f"   Sparsity: {stats['sparsity_pct']}")
    print(f"   DB Size:  {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")
    print(f"   Time:     {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return {"movies": n_movies, "users": n_users, "ratings": n_ratings}


def seed_postgres(data_dir: str, pg_url: str):
    """Seed PostgreSQL database with MovieLens-100k data."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    print("=" * 70)
    print("SEEDING PostgreSQL DATABASE")
    print(f"   Source:  {os.path.abspath(data_dir)}")
    print(f"   Target:  {pg_url.split('@')[-1] if '@' in pg_url else pg_url}")
    print("=" * 70)

    start = time.time()

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)
    stats = compute_stats(ratings, movies)

    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()

    # Create schema
    cur.execute(POSTGRES_SCHEMA)

    # Clear old movielens data
    cur.execute("DELETE FROM movie_ratings WHERE source = 'movielens'")
    cur.execute("DELETE FROM movies WHERE source = 'movielens'")
    cur.execute("DELETE FROM recommendation_users WHERE source = 'movielens'")

    # Insert movies
    print("\n  Inserting movies...")
    for _, row in movies.iterrows():
        cur.execute(
            """INSERT INTO movies (id, title, release_date, genres, source)
               VALUES (%s, %s, %s, %s, 'movielens')
               ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, genres=EXCLUDED.genres""",
            (int(row["movieId"]), row["title"],
             row.get("release_date", None),
             row["genre_str"] if pd.notna(row.get("genre_str")) else ""),
        )
    print(f"     Inserted {len(movies):,} movies")

    # Insert users
    if not users.empty:
        print("  Inserting users...")
        for _, row in users.iterrows():
            cur.execute(
                """INSERT INTO recommendation_users (id, external_id, age, gender, occupation, source)
                   VALUES (%s, %s, %s, %s, %s, 'movielens')
                   ON CONFLICT (id) DO NOTHING""",
                (int(row["userId"]), int(row["userId"]),
                 int(row["age"]) if pd.notna(row.get("age")) else None,
                 row.get("gender", None), row.get("occupation", None)),
            )
        print(f"     Inserted {len(users):,} users")

    # Insert ratings
    print("  Inserting ratings...")
    for _, row in ratings.iterrows():
        ts = datetime.fromtimestamp(row["timestamp"]) if pd.notna(row.get("timestamp")) else None
        cur.execute(
            """INSERT INTO movie_ratings (user_id, movie_id, rating, source, created_at, updated_at)
               VALUES (%s, %s, %s, 'movielens', %s, %s)
               ON CONFLICT (user_id, movie_id) DO UPDATE SET rating=EXCLUDED.rating""",
            (int(row["userId"]), int(row["movieId"]), float(row["rating"]), ts, ts),
        )
    print(f"     Inserted {len(ratings):,} ratings")

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM movies")
    n_movies = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM recommendation_users")
    n_users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM movie_ratings")
    n_ratings = cur.fetchone()[0]

    cur.close()
    conn.close()

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print("SEED COMPLETE!")
    print(f"   Movies:   {n_movies:,}")
    print(f"   Users:    {n_users:,}")
    print(f"   Ratings:  {n_ratings:,}")
    print(f"   Sparsity: {stats['sparsity_pct']}")
    print(f"   Time:     {elapsed:.1f}s")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Seed database with MovieLens-100k data")
    parser.add_argument("--target", choices=["sqlite", "postgres"], default="sqlite",
                        help="Target database type")
    parser.add_argument("--data-dir", default="data/ml-100k",
                        help="Path to MovieLens-100k directory")
    parser.add_argument("--db-path", default="data/movies.db",
                        help="SQLite database path (for sqlite target)")
    parser.add_argument("--pg-url", default=None,
                        help="PostgreSQL connection URL (for postgres target)")
    args = parser.parse_args()

    if args.target == "sqlite":
        seed_sqlite(args.data_dir, args.db_path)
    elif args.target == "postgres":
        if not args.pg_url:
            print("ERROR: --pg-url required for postgres target")
            print("Example: --pg-url postgresql://user:pass@localhost:5432/movie_db")
            sys.exit(1)
        seed_postgres(args.data_dir, args.pg_url)


if __name__ == "__main__":
    main()
