"""
Generate Pre-computed Recommendations — Top-N per user.

After training, this module generates the Top-N movie recommendations
for every user by predicting ratings on all unrated items, then sorting.
"""
import time
from collections import defaultdict
import pandas as pd


def generate_top_n(algo, trainset, n: int = 10, verbose: bool = True) -> dict:
    """
    Pre-compute Top-N recommendations for every user.

    Algorithm:
      1. Build anti-testset: all (user, item) pairs NOT in training set
      2. Predict ratings for all these pairs
      3. Group by user, sort by predicted rating descending
      4. Keep top-N per user

    Args:
        algo: trained Surprise algorithm
        trainset: Surprise trainset (from data.build_full_trainset())
        n: number of recommendations per user
        verbose: print progress

    Returns:
        dict: {raw_user_id: [(raw_movie_id, predicted_rating), ...]}
    """
    if verbose:
        print(f"\n📋 Generating Top-{n} recommendations per user...")

    start = time.time()

    # Anti-testset = all (user, item) pairs NOT rated
    anti_testset = trainset.build_anti_testset()
    if verbose:
        print(f"   Anti-testset size: {len(anti_testset):,} predictions to make")

    # Predict all
    predictions = algo.test(anti_testset)

    # Group by user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort and keep top-N for each user
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]

    elapsed = time.time() - start

    if verbose:
        print(f"   Users with recommendations: {len(top_n):,}")
        avg_pred = sum(len(v) for v in top_n.values()) / max(len(top_n), 1)
        print(f"   Avg recommendations/user:   {avg_pred:.0f}")
        print(f"   Generation time:            {elapsed:.1f}s")

        # Show sample for user 1
        if 1 in top_n or "1" in top_n:
            sample_uid = 1 if 1 in top_n else "1"
            print(f"\n   Sample (User {sample_uid}):")
            for movie_id, rating in top_n[sample_uid][:5]:
                print(f"     Movie {movie_id}: predicted {rating:.2f}")

    return dict(top_n)


def generate_popular_movies(ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 20) -> list:
    """
    Generate popular movies list (fallback for cold-start users).

    Criteria: movies with >= 50 ratings, sorted by average rating descending.
    """
    movie_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        count=("rating", "count"),
    ).reset_index()

    # Filter: at least 50 ratings
    popular = movie_stats[movie_stats["count"] >= 50].copy()
    popular = popular.sort_values("avg_rating", ascending=False).head(n)

    # Enrich with movie metadata
    popular = popular.merge(movies[["movieId", "title", "genre_str"]], on="movieId", how="left")

    result = []
    for _, row in popular.iterrows():
        result.append({
            "movie_id": int(row["movieId"]),
            "title": row["title"] if pd.notna(row["title"]) else f"Movie {row['movieId']}",
            "predicted_rating": round(row["avg_rating"], 2),
            "genres": row["genre_str"].split("|") if pd.notna(row.get("genre_str")) else [],
            "rating_count": int(row["count"]),
        })

    return result
