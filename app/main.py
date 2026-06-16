"""
FastAPI Main Application — Movie Recommendation Microservice (Hybrid).

Endpoints:
  GET  /api/v1/recommendations/{user_id}     — Hybrid Top-N recommendations
  GET  /api/v1/recommendations/{user_id}/explain/{movie_id} — Why this movie?
  GET  /api/v1/users/{user_id}/profile        — User genre profile & preferences
  GET  /api/v1/movies/popular                 — Popular movies (cold-start)
  GET  /api/v1/movies/{movie_id}              — Movie details
  GET  /api/v1/movies/{movie_id}/similar      — Similar movies
  GET  /api/v1/model/status                   — Model metadata & health
  POST /api/v1/pipeline/train                 — Trigger re-training (admin)
  GET  /health                                — Health check

Unhappy Cases Handled:
  - User has NO ratings → popular fallback by demographic/genre
  - User has FEW ratings (<5) → content-boosted recommendations
  - User ID not found in dataset → popular fallback with warning
  - SVD model not loaded → content-only or popular fallback
  - Redis down → local inference fallback
  - SQLite down → in-memory fallback
  - All services down → 503 with helpful error
"""
import json
import os
import pickle
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings


# =============================================================================
# Pydantic Schemas
# =============================================================================

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
    genres: list[str] = []
    hybrid_score: Optional[float] = None


class ScoreBreakdown(BaseModel):
    svd: float = 0.0
    content: float = 0.0
    recent: float = 0.0
    popularity: float = 0.0


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]
    source: str
    user_type: str  # "personalized" | "few_ratings" | "cold_start" | "unknown_user"
    cached_at: Optional[str] = None
    latency_ms: float


class UserProfileResponse(BaseModel):
    user_id: int
    total_rated: int
    avg_rating: float
    top_genres: list[dict]
    recent_genres: list[dict]
    user_type: str


class MovieSimilarResponse(BaseModel):
    movie_id: int
    title: str
    similar_movies: list[dict]


class ExplainResponse(BaseModel):
    user_id: int
    movie: dict
    scores: dict
    reasons: dict


class MovieDetail(BaseModel):
    movie_id: int
    title: str
    release_date: Optional[str] = None
    genres: str = ""


class ModelStatus(BaseModel):
    status: str
    model_info: Optional[dict] = None
    redis_connected: bool
    total_cached_users: int
    hybrid_enabled: bool = False


class HealthResponse(BaseModel):
    status: str
    redis: str
    sqlite: str
    hybrid_engine: str
    timestamp: str


class TrainResponse(BaseModel):
    status: str
    message: str


# =============================================================================
# Global State
# =============================================================================

redis_client: Optional[redis.Redis] = None
lightgcn_model = None
trust_svd_model = None
id_mappings = None
hybrid_engine = None  # HybridRecommender instance
ratings_df: Optional[pd.DataFrame] = None
movies_df: Optional[pd.DataFrame] = None


# =============================================================================
# App Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global redis_client, lightgcn_model, trust_svd_model, id_mappings, hybrid_engine, ratings_df, movies_df

    print("Starting Movie Recommendation API (Hybrid Mode)...")

    # 1. Connect to Redis
    try:
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        redis_client.ping()
        print(f"  Redis connected: {settings.redis_host}:{settings.redis_port}")
    except (redis.ConnectionError, redis.TimeoutError):
        redis_client = None
        print("  Redis not available — running in degraded mode (no cache)")

    # 2. Load ID Mappings
    id_mappings_path = os.path.join(settings.model_dir, "id_mappings.pkl")
    if os.path.exists(id_mappings_path):
        try:
            with open(id_mappings_path, "rb") as f:
                id_mappings = pickle.load(f)
            print(f"  ID mappings loaded: {id_mappings_path}")
        except Exception as e:
            id_mappings = None
            print(f"  Failed to load ID mappings: {e}")
    else:
        id_mappings = None
        print(f"  No ID mappings at {id_mappings_path}")

    # Load LightGCN model
    lightgcn_model = None
    lightgcn_path = os.path.join(settings.model_dir, "lightgcn_model.pth")
    if id_mappings is not None and os.path.exists(lightgcn_path):
        try:
            from pipeline.engines.lightgcn_engine import LightGCNEngine
            num_users = len(id_mappings["user_idx_to_raw"])
            num_items = len(id_mappings["item_idx_to_raw"])
            lightgcn_model = LightGCNEngine(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=64,
                num_layers=3,
            )
            lightgcn_model.load_model(lightgcn_path)
            print(f"  LightGCN model loaded: {lightgcn_path}")
        except Exception as e:
            lightgcn_model = None
            print(f"  Failed to load LightGCN model: {e}")
    else:
        print(f"  No LightGCN model loaded (path: {lightgcn_path})")

    # Load TrustSVD model
    trust_svd_model = None
    trust_svd_path = os.path.join(settings.model_dir, "trust_svd_model.pth")
    if id_mappings is not None and os.path.exists(trust_svd_path):
        try:
            from pipeline.engines.trust_svd_engine import TrustSVDEngine
            trust_svd_model = TrustSVDEngine(
                n_factors=50,
            )
            trust_svd_model.load_model(trust_svd_path)
            print(f"  TrustSVD model loaded: {trust_svd_path}")
        except Exception as e:
            trust_svd_model = None
            print(f"  Failed to load TrustSVD model: {e}")
    else:
        print(f"  No TrustSVD model loaded (path: {trust_svd_path})")

    # 3. Load data for hybrid engine
    try:
        from pipeline.data_loader import load_ratings, load_movies
        ratings_df = load_ratings(settings.data_dir)
        movies_df = load_movies(settings.data_dir)
        print(f"  Data loaded: {len(ratings_df):,} ratings, {len(movies_df):,} movies")
    except Exception as e:
        print(f"  Failed to load data: {e}")
        ratings_df = None
        movies_df = None

    # 4. Initialize Hybrid Engine
    if lightgcn_model is not None and trust_svd_model is not None and id_mappings is not None and ratings_df is not None and movies_df is not None:
        try:
            from pipeline.hybrid_recommender import HybridRecommender
            hybrid_engine = HybridRecommender(
                lightgcn_model, trust_svd_model, id_mappings, ratings_df, movies_df, verbose=True,
            )
            print("  Hybrid engine initialized (5 signals: LightGCN + TrustSVD core)")
        except Exception as e:
            hybrid_engine = None
            print(f"  Hybrid engine failed: {e}")
    else:
        print("  Hybrid engine not initialized (missing model or data)")

    yield

    # Shutdown
    if redis_client:
        redis_client.close()
    print("API shutdown complete.")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "Hybrid movie recommendation microservice.\n\n"
        "**5 Signals:** SVD (50%) + Content-Based (25%) + Recent Taste (10%) "
        "+ Popularity (10%) + Diversity (5%)\n\n"
        "**Fallback Chain:** Hybrid → Content-Only → Popular → 503\n\n"
        "**Dataset:** MovieLens-100k (943 users × 1,682 movies)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

MIN_RATINGS_FOR_HYBRID = 5  # Minimum ratings for full hybrid to work well


def classify_user(user_id: int) -> tuple[str, int]:
    """
    Classify user type based on their rating history.

    Returns: (user_type, rating_count)
      - "personalized": ≥5 ratings → full hybrid
      - "few_ratings":  1-4 ratings → content-boosted hybrid
      - "cold_start":   0 ratings but user exists → popular by demographics
      - "unknown_user": user not in dataset → global popular
    """
    if ratings_df is None:
        return "unknown_user", 0

    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    count = len(user_ratings)

    if count >= MIN_RATINGS_FOR_HYBRID:
        return "personalized", count
    elif count > 0:
        return "few_ratings", count
    elif user_id > 0:
        # Check if user exists but has no ratings
        return "cold_start", 0
    else:
        return "unknown_user", 0


def get_movie_from_sqlite(movie_id: int) -> Optional[dict]:
    """Fetch movie details from SQLite database."""
    db_path = settings.db_path
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT movie_id, title, release_date, genres FROM movies WHERE movie_id = ?",
            (movie_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "movie_id": row[0], "title": row[1],
                "release_date": row[2], "genres": row[3] or "",
            }
    except Exception:
        pass
    return None


def get_popular_movies_from_sqlite(top_n: int = 10) -> list[dict]:
    """Fetch globally popular movies from SQLite."""
    db_path = settings.db_path
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT m.movie_id, m.title, m.genres, AVG(r.rating) as avg_rating, COUNT(r.rating) as cnt
            FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
            GROUP BY m.movie_id HAVING cnt >= 50
            ORDER BY avg_rating DESC LIMIT ?
        """
        rows = conn.execute(query, (top_n,)).fetchall()
        conn.close()
        return [
            {
                "movie_id": r[0], "title": r[1],
                "predicted_rating": round(r[3], 2),
                "genres": r[2].split("|") if r[2] else [],
            }
            for r in rows
        ]
    except Exception:
        return []


def get_popular_by_genre(genres: list[str], top_n: int = 10) -> list[dict]:
    """
    Get popular movies filtered by specific genres.
    Used for cold-start users who selected genre preferences.
    """
    if ratings_df is None or movies_df is None:
        return get_popular_movies_from_sqlite(top_n)

    genre_col = "genre_str" if "genre_str" in movies_df.columns else "genres"

    # Filter movies that match any of the target genres
    matching_movies = []
    for _, row in movies_df.iterrows():
        movie_genres = row.get(genre_col, "")
        if isinstance(movie_genres, list):
            movie_genres = "|".join(movie_genres)
        if not movie_genres:
            continue
        movie_genre_set = set(movie_genres.split("|"))
        if movie_genre_set & set(genres):
            matching_movies.append(int(row["movieId"]))

    if not matching_movies:
        return get_popular_movies_from_sqlite(top_n)

    # Get ratings for matching movies
    filtered = ratings_df[ratings_df["movieId"].isin(matching_movies)]
    stats = filtered.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        count=("rating", "count"),
    ).reset_index()
    stats = stats[stats["count"] >= 20].sort_values("avg_rating", ascending=False).head(top_n)

    results = []
    for _, row in stats.iterrows():
        mid = int(row["movieId"])
        movie_row = movies_df[movies_df["movieId"] == mid]
        title = movie_row["title"].values[0] if len(movie_row) > 0 else f"Movie {mid}"
        genre_val = movie_row[genre_col].values[0] if len(movie_row) > 0 else ""
        if isinstance(genre_val, list):
            genre_list = genre_val
        else:
            genre_list = genre_val.split("|") if genre_val else []

        results.append({
            "movie_id": mid,
            "title": title,
            "predicted_rating": round(float(row["avg_rating"]), 2),
            "genres": genre_list,
        })

    return results


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — verify all components."""
    redis_status = "disconnected"
    if redis_client:
        try:
            redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "error"

    sqlite_status = "available" if os.path.exists(settings.db_path) else "not_found"
    hybrid_status = "active" if hybrid_engine else "inactive"

    overall = "healthy" if hybrid_engine else "degraded"
    return HealthResponse(
        status=overall,
        redis=redis_status,
        sqlite=sqlite_status,
        hybrid_engine=hybrid_status,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/api/v1/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get personalized movie recommendations (Hybrid)",
)
async def get_recommendations(
    user_id: int,
    top_n: int = Query(default=10, le=50),
    prefer_genres: Optional[str] = Query(
        default=None,
        description="Comma-separated genres for cold-start (e.g. 'Action,Drama')",
    ),
):
    """
    Retrieve Top-N movie recommendations using the Hybrid engine.

    **Adaptive behavior based on user data:**
    - **≥5 ratings:** Full hybrid (SVD + Content + Recent + Popularity + Diversity)
    - **1-4 ratings:** Content-boosted (SVD weight reduced, content weight increased)
    - **0 ratings (cold start):** Popular movies filtered by preferred genres
    - **Unknown user:** Global popular fallback

    **Parameters:**
    - `top_n`: Number of recommendations (1-50)
    - `prefer_genres`: Genres for cold-start users (e.g. "Action,Drama,Sci-Fi")
    """
    start = time.time()

    # Classify user
    user_type, rating_count = classify_user(user_id)

    # ─────────────────────────────────────────────────────────
    # CASE 1: Personalized user (≥5 ratings) → Full Hybrid
    # ─────────────────────────────────────────────────────────
    if user_type == "personalized" and hybrid_engine:
        try:
            # Try Redis cache first
            if redis_client:
                try:
                    cached = redis_client.get(f"reco:hybrid:{user_id}")
                    if cached:
                        data = json.loads(cached)
                        movies = [MovieRecommendation(**m) for m in data["movies"][:top_n]]
                        latency = (time.time() - start) * 1000
                        return RecommendationResponse(
                            user_id=user_id,
                            recommendations=movies,
                            source="hybrid_cache",
                            user_type="personalized",
                            cached_at=data.get("cached_at"),
                            latency_ms=round(latency, 2),
                        )
                except Exception:
                    pass

            # Real-time hybrid inference
            recs = hybrid_engine.recommend(user_id, top_n=top_n, explain=False)
            movies = [
                MovieRecommendation(
                    movie_id=r["movie_id"],
                    title=r["title"],
                    predicted_rating=r.get("predicted_rating", r.get("svd_pred", 0.0)),
                    genres=r["genres"],
                    hybrid_score=r["hybrid_score"],
                )
                for r in recs
            ]
            latency = (time.time() - start) * 1000
            return RecommendationResponse(
                user_id=user_id,
                recommendations=movies,
                source="hybrid_realtime",
                user_type="personalized",
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            print(f"Warning: Hybrid failed for user {user_id}: {e}")
            # Fall through to next case

    # ─────────────────────────────────────────────────────────
    # CASE 2: Few ratings (1-4) → Content-boosted hybrid
    # ─────────────────────────────────────────────────────────
    if user_type == "few_ratings" and hybrid_engine:
        try:
            # Use hybrid with boosted content weights
            from pipeline.hybrid_recommender import HybridRecommender
            boosted_weights = {
                "lightgcn": 0.25,
                "trust_svd": 0.15,
                "content": 0.45,
                "recent": 0.10,
                "diversity": 0.05,
            }
            # Temporarily override weights
            original_weights = hybrid_engine.weights.copy()
            hybrid_engine.weights = boosted_weights

            recs = hybrid_engine.recommend(user_id, top_n=top_n, explain=False)

            # Restore weights
            hybrid_engine.weights = original_weights

            movies = [
                MovieRecommendation(
                    movie_id=r["movie_id"],
                    title=r["title"],
                    predicted_rating=r.get("predicted_rating", r.get("svd_pred", 0.0)),
                    genres=r["genres"],
                    hybrid_score=r["hybrid_score"],
                )
                for r in recs
            ]
            latency = (time.time() - start) * 1000
            return RecommendationResponse(
                user_id=user_id,
                recommendations=movies,
                source="hybrid_content_boosted",
                user_type="few_ratings",
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            print(f"Warning: Content-boosted failed for user {user_id}: {e}")

    # ─────────────────────────────────────────────────────────
    # CASE 3: Cold start (0 ratings) → Popular by genre or global
    # ─────────────────────────────────────────────────────────
    if user_type in ("cold_start", "unknown_user"):
        # If user specified genre preferences
        if prefer_genres:
            genres = [g.strip() for g in prefer_genres.split(",")]
            popular = get_popular_by_genre(genres, top_n)
            source = "popular_by_genre"
        else:
            popular = get_popular_movies_from_sqlite(top_n)
            source = "global_popular"

        if popular:
            movies = [MovieRecommendation(**m) for m in popular]
            latency = (time.time() - start) * 1000
            return RecommendationResponse(
                user_id=user_id,
                recommendations=movies,
                source=source,
                user_type=user_type,
                latency_ms=round(latency, 2),
            )

    # ─────────────────────────────────────────────────────────
    # CASE 4: LightGCN-only fallback (hybrid engine not loaded)
    # ─────────────────────────────────────────────────────────
    if lightgcn_model and id_mappings is not None and ratings_df is not None:
        try:
            user_idx = id_mappings["user_raw_to_idx"].get(user_id)
            if user_idx is not None:
                # Recommend directly using LightGCN model top-n
                top_idxs = lightgcn_model.recommend_top_n(user_idx, top_n=top_n)
                item_idx_to_raw = id_mappings["item_idx_to_raw"]
                movies = []
                genre_col = "genre_str" if "genre_str" in movies_df.columns else "genres"
                for idx in top_idxs:
                    mid = item_idx_to_raw.get(idx)
                    if mid is not None:
                        row = movies_df[movies_df["movieId"] == mid]
                        title = row["title"].values[0] if len(row) > 0 else f"Movie {mid}"
                        g = row[genre_col].values[0] if len(row) > 0 else ""
                        if isinstance(g, list):
                            genre_list = g
                        else:
                            genre_list = g.split("|") if g else []
                        movies.append(
                            MovieRecommendation(
                                movie_id=mid,
                                title=title,
                                predicted_rating=0.0,
                                genres=genre_list,
                                hybrid_score=None,
                            )
                        )
                latency = (time.time() - start) * 1000
                return RecommendationResponse(
                    user_id=user_id,
                    recommendations=movies,
                    source="lightgcn_only_fallback",
                    user_type=user_type,
                    latency_ms=round(latency, 2),
                )
        except Exception as e:
            print(f"Warning: LightGCN fallback failed: {e}")

    # ─────────────────────────────────────────────────────────
    # CASE 5: Everything failed → global popular from SQLite
    # ─────────────────────────────────────────────────────────
    popular = get_popular_movies_from_sqlite(top_n)
    if popular:
        movies = [MovieRecommendation(**m) for m in popular]
        latency = (time.time() - start) * 1000
        return RecommendationResponse(
            user_id=user_id,
            recommendations=movies,
            source="emergency_popular_fallback",
            user_type=user_type,
            latency_ms=round(latency, 2),
        )

    # ─────────────────────────────────────────────────────────
    # CASE 6: Nothing works at all → 503
    # ─────────────────────────────────────────────────────────
    latency = (time.time() - start) * 1000
    raise HTTPException(
        status_code=503,
        detail={
            "error": "No recommendations available",
            "user_type": user_type,
            "hint": "Run the training pipeline first: python setup.py",
            "latency_ms": round(latency, 2),
        },
    )


@app.get(
    "/api/v1/recommendations/{user_id}/explain/{movie_id}",
    response_model=ExplainResponse,
    tags=["Recommendations"],
    summary="Explain why a movie is recommended",
)
async def explain_recommendation(user_id: int, movie_id: int):
    """
    Explain WHY a specific movie is recommended for this user.

    Returns detailed breakdown of each signal.
    """
    if not hybrid_engine:
        raise HTTPException(status_code=503, detail="Hybrid engine not initialized")

    user_type, count = classify_user(user_id)
    if user_type == "unknown_user":
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} has no rating history. Cannot explain recommendations.",
        )

    try:
        explanation = hybrid_engine.explain_recommendation(user_id, movie_id)
        return ExplainResponse(
            user_id=user_id,
            movie=explanation["movie"],
            scores=explanation["scores"],
            reasons=explanation["reasons"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {str(e)}")


@app.get(
    "/api/v1/users/{user_id}/profile",
    response_model=UserProfileResponse,
    tags=["Users"],
    summary="Get user genre profile & preferences",
)
async def get_user_profile(user_id: int):
    """
    Get user's genre preference profile.

    Shows: top genres (all-time), recent genre trends, rating stats.
    """
    if not hybrid_engine:
        raise HTTPException(status_code=503, detail="Hybrid engine not initialized")

    user_type, count = classify_user(user_id)

    if user_type == "unknown_user" or count == 0:
        return UserProfileResponse(
            user_id=user_id,
            total_rated=0,
            avg_rating=0.0,
            top_genres=[],
            recent_genres=[],
            user_type=user_type,
        )

    try:
        profile = hybrid_engine.get_user_profile(user_id)
        top_genres = [
            {"genre": g, "score": s}
            for g, s in profile["overall"]["top_genres"]
        ]
        recent_genres = [
            {"genre": g, "count": c}
            for g, c in profile["recent"]["recent_genres"]
        ]

        return UserProfileResponse(
            user_id=user_id,
            total_rated=profile["overall"]["total_rated"],
            avg_rating=profile["overall"]["avg_rating"],
            top_genres=top_genres,
            recent_genres=recent_genres,
            user_type=user_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile failed: {str(e)}")


@app.get(
    "/api/v1/movies/{movie_id}/similar",
    response_model=MovieSimilarResponse,
    tags=["Movies"],
    summary="Get similar movies",
)
async def get_similar_movies(movie_id: int, top_n: int = Query(default=10, le=50)):
    """Find movies similar to a given movie based on genre similarity."""
    if not hybrid_engine:
        raise HTTPException(status_code=503, detail="Hybrid engine not initialized")

    from pipeline.content_based import find_similar_movies

    info = hybrid_engine.movie_info.get(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    similar = find_similar_movies(
        movie_id, hybrid_engine.genre_matrix,
        hybrid_engine.movie_id_to_idx,
        hybrid_engine.idx_to_movie_id,
        top_n=top_n,
    )

    similar_list = []
    for sim_id, sim_score in similar:
        sim_info = hybrid_engine.movie_info.get(sim_id, {})
        similar_list.append({
            "movie_id": sim_id,
            "title": sim_info.get("title", f"Movie {sim_id}"),
            "genres": sim_info.get("genres", []),
            "similarity": sim_score,
        })

    return MovieSimilarResponse(
        movie_id=movie_id,
        title=info.get("title", f"Movie {movie_id}"),
        similar_movies=similar_list,
    )


@app.get(
    "/api/v1/movies/popular",
    response_model=list[MovieRecommendation],
    tags=["Movies"],
    summary="Get popular movies (cold-start fallback)",
)
async def get_popular_movies(
    top_n: int = Query(default=10, le=50),
    genres: Optional[str] = Query(
        default=None,
        description="Filter by genres (comma-separated, e.g. 'Action,Drama')",
    ),
):
    """
    Return globally popular movies.

    Optionally filter by genres for cold-start users who have genre preferences.
    """
    if genres:
        genre_list = [g.strip() for g in genres.split(",")]
        popular = get_popular_by_genre(genre_list, top_n)
    else:
        # Try Redis first
        if redis_client:
            try:
                cached = redis_client.get("reco:popular")
                if cached:
                    return [MovieRecommendation(**m) for m in json.loads(cached)[:top_n]]
            except Exception:
                pass
        popular = get_popular_movies_from_sqlite(top_n)

    if not popular:
        raise HTTPException(status_code=503, detail="Popular movies not available. Run pipeline first.")

    return [MovieRecommendation(**m) for m in popular]


@app.get(
    "/api/v1/movies/{movie_id}",
    response_model=MovieDetail,
    tags=["Movies"],
    summary="Get movie details",
)
async def get_movie_detail(movie_id: int):
    """Fetch movie metadata."""
    movie = get_movie_from_sqlite(movie_id)
    if not movie:
        # Try from in-memory data
        if hybrid_engine and movie_id in hybrid_engine.movie_info:
            info = hybrid_engine.movie_info[movie_id]
            return MovieDetail(
                movie_id=movie_id,
                title=info.get("title", f"Movie {movie_id}"),
                genres="|".join(info.get("genres", [])),
            )
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return MovieDetail(**movie)


@app.get(
    "/api/v1/model/status",
    response_model=ModelStatus,
    tags=["System"],
    summary="Get model status and metrics",
)
async def get_model_status():
    """Return current model metadata."""
    model_info = None
    redis_connected = False
    total_cached = 0

    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
            meta = redis_client.get("model:metadata")
            if meta:
                model_info = json.loads(meta)
            keys = redis_client.keys("reco:*")
            total_cached = len([k for k in keys if k != "reco:popular"])
        except Exception:
            pass

    if not model_info:
        metadata_path = os.path.join(settings.model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    model_info = json.load(f)
            except Exception:
                pass

    if model_info is None and lightgcn_model:
        model_info = {"algorithm": "LightGCN + TrustSVD", "serving_mode": "online_inference"}

    if model_info:
        model_info["hybrid_enabled"] = hybrid_engine is not None
        if hybrid_engine:
            model_info["hybrid_weights"] = hybrid_engine.weights
            model_info["signals"] = [
                "LightGCN (Graph Collaborative Filtering - 50%)",
                "TrustSVD (Social Collaborative Filtering - 15%)",
                "Content-Based (Genre Matching - 25%)",
                "Recent Taste (Time-Weighted - 10%)",
                "Diversity (MMR Re-ranking - 5%)",
            ]

    status = "ready" if hybrid_engine else ("degraded" if lightgcn_model else "no_model")

    return ModelStatus(
        status=status,
        model_info=model_info,
        redis_connected=redis_connected,
        total_cached_users=total_cached,
        hybrid_enabled=hybrid_engine is not None,
    )


@app.post(
    "/api/v1/pipeline/train",
    response_model=TrainResponse,
    tags=["Admin"],
    summary="Trigger model re-training",
)
async def trigger_training(
    skip_benchmark: bool = Query(default=True),
    skip_gridsearch: bool = Query(default=False),
):
    """
    Trigger the full training pipeline (admin endpoint).

    WARNING: Blocking operation that takes 30-60 seconds.
    """
    try:
        from pipeline.run_pipeline import run_pipeline

        result = run_pipeline(
            data_dir=settings.data_dir,
            model_dir=settings.model_dir,
            db_path=settings.db_path,
            top_n=settings.top_n,
            skip_benchmark=skip_benchmark,
            skip_gridsearch=skip_gridsearch,
            no_redis=not bool(redis_client),
            redis_host=settings.redis_host,
            redis_port=settings.redis_port,
            redis_db=settings.redis_db,
            cache_ttl=settings.cache_ttl_seconds,
        )

        return TrainResponse(
            status="success",
            message=(
                f"Pipeline completed in {result['pipeline_time']:.1f}s. "
                f"Users: {result['top_n_count']}. "
                f"Restart server to reload hybrid engine."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# =============================================================================
# Run with: uvicorn app.main:app --reload
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
