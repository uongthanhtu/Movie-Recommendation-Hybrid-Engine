"""
Content-Based Filtering & User Profiling Engine.

Tín hiệu bổ sung ngoài SVD (Collaborative Filtering):
  1. Genre Preference Profile — phân tích thể loại user hay xem/thích
  2. Recent History Weighting — thể loại xem gần đây quan trọng hơn
  3. Movie Content Similarity — phim tương tự dựa trên genres
  4. Demographic Hints — gợi ý theo nhóm tuổi/giới tính

Kết hợp với SVD → Hybrid Score.
"""
import os
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. GENRE VECTOR — Biểu diễn mỗi phim thành vector 19 chiều
# =============================================================================

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "unknown",
]

GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}


def movie_to_genre_vector(genre_str: str) -> np.ndarray:
    """Convert genre string 'Action|Drama|Thriller' → binary vector [1,0,0,...,1,...,1,...]."""
    vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
    if genre_str:
        for g in genre_str.split("|"):
            g = g.strip()
            if g in GENRE_TO_IDX:
                vec[GENRE_TO_IDX[g]] = 1.0
    return vec


def build_movie_genre_matrix(movies_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Build genre matrix for all movies.

    Returns:
        genre_matrix: (n_movies, 19) binary matrix
        movie_id_to_idx: {movie_id: matrix_row_index}
    """
    movie_ids = movies_df["movieId"].tolist()
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    genre_col = "genre_str" if "genre_str" in movies_df.columns else "genres"
    genre_matrix = np.zeros((len(movie_ids), len(ALL_GENRES)), dtype=np.float32)

    for i, (_, row) in enumerate(movies_df.iterrows()):
        genre_str = row.get(genre_col, "")
        if isinstance(genre_str, list):
            genre_str = "|".join(genre_str)
        genre_matrix[i] = movie_to_genre_vector(genre_str if pd.notna(genre_str) else "")

    return genre_matrix, movie_id_to_idx


# =============================================================================
# 2. USER GENRE PROFILE — Phân tích sở thích thể loại của user
# =============================================================================

def build_user_genre_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_matrix: np.ndarray,
    movie_id_to_idx: dict,
    time_decay: bool = True,
    decay_factor: float = 0.95,
) -> dict:
    """
    Phân tích sở thích thể loại của 1 user.

    Tính weighted average of genre vectors, trọng số = rating.
    Nếu time_decay=True, ratings gần đây có trọng số cao hơn.

    Returns:
        {
            "genre_vector": np.array([0.8, 0.1, ...]),  # normalized preference
            "top_genres": [("Drama", 0.82), ("Action", 0.65), ...],
            "genre_distribution": {"Drama": 15, "Action": 8, ...},
            "total_rated": 42,
            "avg_rating": 3.8,
        }
    """
    user_ratings = ratings_df[ratings_df["userId"] == user_id].copy()

    if user_ratings.empty:
        return {
            "genre_vector": np.zeros(len(ALL_GENRES), dtype=np.float32),
            "top_genres": [],
            "genre_distribution": {},
            "total_rated": 0,
            "avg_rating": 0.0,
        }

    # Sort by timestamp (recent first) for time decay
    if "timestamp" in user_ratings.columns:
        user_ratings = user_ratings.sort_values("timestamp", ascending=False)

    # Weighted genre vector
    weighted_genre_vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
    genre_count = defaultdict(int)
    total_weight = 0

    for rank, (_, row) in enumerate(user_ratings.iterrows()):
        movie_id = int(row["movieId"])
        rating = float(row["rating"])

        idx = movie_id_to_idx.get(movie_id)
        if idx is None:
            continue

        # Time decay: recent ratings matter more
        weight = rating
        if time_decay:
            weight *= (decay_factor ** rank)

        weighted_genre_vec += genre_matrix[idx] * weight
        total_weight += weight

        # Count genres
        for g_idx, val in enumerate(genre_matrix[idx]):
            if val > 0:
                genre_count[ALL_GENRES[g_idx]] += 1

    # Normalize
    if total_weight > 0:
        weighted_genre_vec /= total_weight

    # Top genres sorted by preference score
    genre_scores = [(ALL_GENRES[i], float(weighted_genre_vec[i])) for i in range(len(ALL_GENRES))]
    genre_scores.sort(key=lambda x: x[1], reverse=True)
    top_genres = [(g, round(s, 3)) for g, s in genre_scores if s > 0.01]

    return {
        "genre_vector": weighted_genre_vec,
        "top_genres": top_genres,
        "genre_distribution": dict(genre_count),
        "total_rated": len(user_ratings),
        "avg_rating": round(float(user_ratings["rating"].mean()), 2),
    }


def build_recent_genre_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_matrix: np.ndarray,
    movie_id_to_idx: dict,
    recent_n: int = 20,
) -> dict:
    """
    Phân tích sở thích thể loại GẦN ĐÂY (last N ratings).

    Giúp phát hiện thay đổi sở thích:
    - Trước thích Action, gần đây chuyển sang Drama
    - Mùa hè thích Comedy, mùa đông thích Romance
    """
    user_ratings = ratings_df[ratings_df["userId"] == user_id].copy()
    if "timestamp" in user_ratings.columns:
        user_ratings = user_ratings.sort_values("timestamp", ascending=False)

    recent = user_ratings.head(recent_n)

    if recent.empty:
        return {"recent_genres": [], "recent_avg_rating": 0.0, "shift_detected": False}

    genre_count = defaultdict(int)
    for _, row in recent.iterrows():
        idx = movie_id_to_idx.get(int(row["movieId"]))
        if idx is None:
            continue
        for g_idx, val in enumerate(genre_matrix[idx]):
            if val > 0:
                genre_count[ALL_GENRES[g_idx]] += 1

    sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)

    return {
        "recent_genres": sorted_genres[:5],
        "recent_avg_rating": round(float(recent["rating"].mean()), 2),
        "recent_count": len(recent),
    }


# =============================================================================
# 3. CONTENT-BASED SCORE — Đánh giá phim dựa trên nội dung
# =============================================================================

def content_based_score(
    user_profile: np.ndarray,
    movie_genre_vec: np.ndarray,
) -> float:
    """
    Cosine similarity giữa user genre profile và movie genre vector.

    Score 0.0 → 1.0 (càng cao = càng phù hợp sở thích).
    """
    if np.linalg.norm(user_profile) == 0 or np.linalg.norm(movie_genre_vec) == 0:
        return 0.0

    score = np.dot(user_profile, movie_genre_vec) / (
        np.linalg.norm(user_profile) * np.linalg.norm(movie_genre_vec)
    )
    return float(max(0.0, score))


def batch_content_scores(
    user_profile: np.ndarray,
    genre_matrix: np.ndarray,
) -> np.ndarray:
    """Compute content scores for ALL movies at once (vectorized)."""
    user_norm = np.linalg.norm(user_profile)
    if user_norm == 0:
        return np.zeros(genre_matrix.shape[0])

    movie_norms = np.linalg.norm(genre_matrix, axis=1)
    movie_norms[movie_norms == 0] = 1e-8  # avoid division by zero

    scores = genre_matrix @ user_profile / (movie_norms * user_norm)
    return np.clip(scores, 0, 1)


# =============================================================================
# 4. MOVIE SIMILARITY — Phim tương tự dựa trên genres
# =============================================================================

def find_similar_movies(
    movie_id: int,
    genre_matrix: np.ndarray,
    movie_id_to_idx: dict,
    idx_to_movie_id: dict = None,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """
    Tìm phim tương tự dựa trên genre cosine similarity.

    Returns: [(movie_id, similarity_score), ...]
    """
    idx = movie_id_to_idx.get(movie_id)
    if idx is None:
        return []

    if idx_to_movie_id is None:
        idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}

    movie_vec = genre_matrix[idx].reshape(1, -1)
    similarities = cosine_similarity(movie_vec, genre_matrix)[0]

    # Sort by similarity, exclude self
    ranked = np.argsort(similarities)[::-1]
    results = []
    for r_idx in ranked:
        if r_idx == idx:
            continue
        mid = idx_to_movie_id.get(r_idx)
        if mid is not None and similarities[r_idx] > 0:
            results.append((mid, round(float(similarities[r_idx]), 3)))
        if len(results) >= top_n:
            break

    return results


# =============================================================================
# 5. POPULARITY & TRENDING — Phim phổ biến gần đây
# =============================================================================

def compute_popularity_scores(
    ratings_df: pd.DataFrame,
    recency_days: int = 90,
) -> pd.DataFrame:
    """
    Tính popularity score cho mỗi phim.

    Score = weighted combination of:
      - Total rating count (normalized)
      - Average rating (normalized)
      - Recent rating count (last N days)
    """
    movie_stats = ratings_df.groupby("movieId").agg(
        total_count=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Normalize
    max_count = movie_stats["total_count"].max()
    movie_stats["count_norm"] = movie_stats["total_count"] / max(max_count, 1)
    movie_stats["rating_norm"] = (movie_stats["avg_rating"] - 1) / 4  # scale 1-5 → 0-1

    # Recent ratings (if timestamp available)
    if "timestamp" in ratings_df.columns:
        max_ts = ratings_df["timestamp"].max()
        if pd.notna(max_ts) and max_ts > 0:
            cutoff = max_ts - (recency_days * 86400)
            recent = ratings_df[ratings_df["timestamp"] >= cutoff]
            recent_counts = recent.groupby("movieId").size().reset_index(name="recent_count")
            movie_stats = movie_stats.merge(recent_counts, on="movieId", how="left")
            movie_stats["recent_count"] = movie_stats["recent_count"].fillna(0)
            max_recent = movie_stats["recent_count"].max()
            movie_stats["recent_norm"] = movie_stats["recent_count"] / max(max_recent, 1)
        else:
            movie_stats["recent_norm"] = 0.0
    else:
        movie_stats["recent_norm"] = 0.0

    # Weighted popularity score
    movie_stats["popularity_score"] = (
        0.3 * movie_stats["count_norm"]
        + 0.4 * movie_stats["rating_norm"]
        + 0.3 * movie_stats["recent_norm"]
    )

    return movie_stats[["movieId", "total_count", "avg_rating", "popularity_score"]]


# =============================================================================
# 6. DEMOGRAPHIC HINTS — Gợi ý theo nhóm tuổi/giới tính
# =============================================================================

def build_demographic_preferences(
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> dict:
    """
    Build average genre preferences per demographic group.

    Returns: {
        "age_group": {"18-25": {"top_genres": [...], "avg_rating": 3.5}},
        "gender": {"M": {...}, "F": {...}},
    }
    """
    if users_df.empty or "age" not in users_df.columns:
        return {}

    # Merge ratings with user demographics
    merged = ratings_df.merge(
        users_df[["userId", "age", "gender"]],
        on="userId",
        how="inner",
    )

    if merged.empty:
        return {}

    # Age groups
    def age_group(age):
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        return "55+"

    merged["age_group"] = merged["age"].apply(age_group)

    result = {"age_group": {}, "gender": {}}

    for group in merged["age_group"].unique():
        subset = merged[merged["age_group"] == group]
        result["age_group"][group] = {
            "avg_rating": round(float(subset["rating"].mean()), 2),
            "total_ratings": len(subset),
        }

    for gender in merged["gender"].unique():
        subset = merged[merged["gender"] == gender]
        result["gender"][gender] = {
            "avg_rating": round(float(subset["rating"].mean()), 2),
            "total_ratings": len(subset),
        }

    return result


# =============================================================================
# 7. MASTER: Build All Content Features
# =============================================================================

def build_content_features(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame = None,
    verbose: bool = True,
) -> dict:
    """
    Build all content-based features for the recommendation engine.

    Returns dict containing all pre-computed features needed for hybrid scoring.
    """
    if verbose:
        print("\n🎯 Building Content-Based Features...")

    # Genre matrix
    genre_matrix, movie_id_to_idx = build_movie_genre_matrix(movies_df)
    idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}
    if verbose:
        print(f"   ├── Genre matrix: {genre_matrix.shape} ({len(ALL_GENRES)} genres)")

    # Popularity scores
    popularity = compute_popularity_scores(ratings_df)
    if verbose:
        print(f"   ├── Popularity scores: {len(popularity)} movies")

    # Demographic preferences
    demo_prefs = {}
    if users_df is not None and not users_df.empty:
        demo_prefs = build_demographic_preferences(ratings_df, users_df)
        if verbose:
            print(f"   ├── Demographics: {len(demo_prefs.get('age_group', {}))} age groups, "
                  f"{len(demo_prefs.get('gender', {}))} genders")

    if verbose:
        print(f"   └── ✅ Content features ready")

    return {
        "genre_matrix": genre_matrix,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
        "popularity": popularity,
        "demographic_prefs": demo_prefs,
    }
