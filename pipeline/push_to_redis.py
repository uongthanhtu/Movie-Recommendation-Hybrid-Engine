"""
Push to Redis — Store pre-computed recommendations for O(1) serving.

Key schema:
  - reco:{user_id}     → JSON: {"movies": [...], "cached_at": "ISO timestamp"}
  - reco:popular       → JSON: [{"movie_id": ..., "title": ..., ...}]
  - model:metadata     → JSON: {rmse, params, trained_at, ...}
"""
import json
import time
from datetime import datetime
import redis
import pandas as pd


def get_redis_client(host: str = "localhost", port: int = 6379, db: int = 0) -> redis.Redis:
    """Create Redis client with connection test."""
    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    try:
        r.ping()
        print(f"   Redis connected: {host}:{port}/{db}")
    except redis.ConnectionError:
        print(f"  ️  Redis not available at {host}:{port}. Results will NOT be cached.")
        return None
    return r


def push_recommendations(
    r: redis.Redis,
    top_n: dict,
    movies_df: pd.DataFrame,
    ttl: int = 86400,
    verbose: bool = True,
) -> int:
    """
    Push pre-computed Top-N recommendations to Redis.

    Args:
        r: Redis client
        top_n: {user_id: [(movie_id, predicted_rating), ...]}
        movies_df: DataFrame with movie metadata
        ttl: TTL in seconds (default: 24h)
        verbose: print progress

    Returns:
        Number of users pushed
    """
    if r is None:
        print("  ️  Redis not available, skipping push.")
        return 0

    if verbose:
        print(f"\n Pushing recommendations to Redis...")

    now = datetime.now().isoformat()

    # Build movie lookup for fast enrichment
    movie_lookup = {}
    for _, row in movies_df.iterrows():
        mid = row["movieId"]
        movie_lookup[mid] = {
            "title": row["title"] if pd.notna(row["title"]) else f"Movie {mid}",
            "genres": row["genres"] if isinstance(row["genres"], list) else [],
        }

    pushed = 0
    start = time.time()

    # Use pipeline for batch writes (much faster)
    pipe = r.pipeline()

    for user_id, recs in top_n.items():
        movies_list = []
        for movie_id, pred_rating in recs:
            info = movie_lookup.get(movie_id, {"title": f"Movie {movie_id}", "genres": []})
            movies_list.append({
                "movie_id": int(movie_id) if not isinstance(movie_id, int) else movie_id,
                "title": info["title"],
                "predicted_rating": round(pred_rating, 2),
                "genres": info["genres"],
            })

        key = f"reco:{int(user_id) if not isinstance(user_id, int) else user_id}"
        pipe.set(key, json.dumps({"movies": movies_list, "cached_at": now}), ex=ttl)
        pushed += 1

        # Execute pipeline batch every 100 users
        if pushed % 100 == 0:
            pipe.execute()
            pipe = r.pipeline()

    # Execute remaining
    pipe.execute()

    elapsed = time.time() - start

    if verbose:
        print(f"   Users pushed:  {pushed:,}")
        print(f"   TTL:           {ttl}s ({ttl // 3600}h)")
        print(f"   Push time:     {elapsed:.2f}s")

    return pushed


def push_popular(r: redis.Redis, popular_movies: list, ttl: int = 86400):
    """Push popular movies list to Redis (cold-start fallback)."""
    if r is None:
        return
    r.set("reco:popular", json.dumps(popular_movies), ex=ttl)
    print(f"   Popular movies pushed ({len(popular_movies)} movies)")


def push_model_metadata(
    r: redis.Redis,
    metrics: dict,
    n_users: int,
    ttl: int = 86400,
):
    """Push model metadata to Redis for monitoring."""
    if r is None:
        return

    metadata = {
        **metrics,
        "users_served": n_users,
        "dataset": "MovieLens-100k",
        "pushed_at": datetime.now().isoformat(),
    }
    r.set("model:metadata", json.dumps(metadata), ex=ttl)
    print(f"   Model metadata pushed")


def flush_old_recommendations(r: redis.Redis):
    """Remove all old recommendation keys."""
    if r is None:
        return
    keys = r.keys("reco:*")
    if keys:
        r.delete(*keys)
        print(f"  ️  Flushed {len(keys)} old recommendation keys")
