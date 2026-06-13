"""
Hybrid Recommender — Kết hợp nhiều tín hiệu để recommend.

Combines:
  1. SVD Score (Collaborative Filtering) — "users giống bạn thích phim này"
  2. Content Score (Genre matching) — "phim này cùng thể loại bạn hay xem"
  3. Recent Taste Score — "gần đây bạn đang chuyển sang thích thể loại này"
  4. Popularity Score — "phim này đang được nhiều người xem"
  5. Diversity Bonus — đảm bảo kết quả đa dạng, không chỉ 1 thể loại

Final Score = w1*SVD + w2*Content + w3*Recent + w4*Popularity + diversity_adjustment
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from pipeline.content_based import (
    build_movie_genre_matrix,
    build_user_genre_profile,
    build_recent_genre_profile,
    batch_content_scores,
    content_based_score,
    compute_popularity_scores,
    find_similar_movies,
    ALL_GENRES,
)


# =============================================================================
# Hybrid Weights Configuration
# =============================================================================

DEFAULT_WEIGHTS = {
    "svd": 0.50,        # Collaborative filtering (core)
    "content": 0.25,    # Genre matching with user profile
    "recent": 0.10,     # Recent genre preference shift
    "popularity": 0.10, # Trending / popular bonus
    "diversity": 0.05,  # Diversity bonus
}


# =============================================================================
# Hybrid Recommender
# =============================================================================

class HybridRecommender:
    """
    Multi-signal recommendation engine.

    Kết hợp SVD + Content-Based + Popularity + Diversity.
    """

    def __init__(
        self,
        svd_model,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        weights: dict = None,
        verbose: bool = True,
    ):
        self.svd_model = svd_model
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.verbose = verbose

        # Pre-compute content features
        if verbose:
            print("\nInitializing Hybrid Recommender...")

        self.genre_matrix, self.movie_id_to_idx = build_movie_genre_matrix(movies_df)
        self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}

        self.popularity_df = compute_popularity_scores(ratings_df)
        self.popularity_map = dict(
            zip(self.popularity_df["movieId"], self.popularity_df["popularity_score"])
        )

        # Movie titles lookup
        title_col = "title"
        genre_col = "genre_str" if "genre_str" in movies_df.columns else "genres"
        self.movie_info = {}
        for _, row in movies_df.iterrows():
            mid = row["movieId"]
            genres = row.get(genre_col, "")
            if isinstance(genres, list):
                genres = "|".join(genres)
            self.movie_info[mid] = {
                "title": row.get(title_col, f"Movie {mid}"),
                "genres": genres.split("|") if pd.notna(genres) and genres else [],
            }

        if verbose:
            print(f"   ├── SVD model loaded")
            print(f"   ├── Genre matrix: {self.genre_matrix.shape}")
            print(f"   ├── Popularity: {len(self.popularity_map)} movies")
            print(f"   ├── Weights: {self.weights}")
            print(f"   └── Ready")

    def get_user_profile(self, user_id: int) -> dict:
        """Get comprehensive user profile including genre preferences."""
        profile = build_user_genre_profile(
            user_id, self.ratings_df, self.movies_df,
            self.genre_matrix, self.movie_id_to_idx,
            time_decay=True,
        )

        recent = build_recent_genre_profile(
            user_id, self.ratings_df, self.movies_df,
            self.genre_matrix, self.movie_id_to_idx,
            recent_n=20,
        )

        # Recent genre vector (from last 20 ratings, no time decay)
        recent_profile = build_user_genre_profile(
            user_id, self.ratings_df, self.movies_df,
            self.genre_matrix, self.movie_id_to_idx,
            time_decay=False,
        )

        return {
            "overall": profile,
            "recent": recent,
            "recent_genre_vector": recent_profile["genre_vector"],
        }

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        explain: bool = False,
    ) -> list[dict]:
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: target user
            top_n: number of recommendations
            explain: if True, include score breakdown per movie

        Returns:
            List of recommendation dicts with hybrid scores.
        """
        start = time.time()

        # 1. Get user's rated movies (to exclude)
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        rated_movies = set(user_ratings["movieId"].tolist())

        # 2. Build user profile
        profile = self.get_user_profile(user_id)
        user_genre_vec = profile["overall"]["genre_vector"]
        recent_genre_vec = profile["recent_genre_vector"]

        # 3. Get all candidate movies (unrated)
        all_movie_ids = [mid for mid in self.movie_info.keys() if mid not in rated_movies]

        if not all_movie_ids:
            return []

        # 4. Score each candidate
        candidates = []

        # Batch content scores
        content_scores_all = batch_content_scores(user_genre_vec, self.genre_matrix)
        recent_scores_all = batch_content_scores(recent_genre_vec, self.genre_matrix)

        for movie_id in all_movie_ids:
            idx = self.movie_id_to_idx.get(movie_id)
            if idx is None:
                continue

            # SVD score (normalize to 0-1 from 1-5 scale)
            svd_pred = self.svd_model.predict(user_id, movie_id).est
            svd_norm = (svd_pred - 1.0) / 4.0  # 1-5 → 0-1

            # Content score (genre match)
            content_score = float(content_scores_all[idx])

            # Recent taste score
            recent_score = float(recent_scores_all[idx])

            # Popularity score
            pop_score = self.popularity_map.get(movie_id, 0.0)

            # Hybrid score
            hybrid_score = (
                self.weights["svd"] * svd_norm
                + self.weights["content"] * content_score
                + self.weights["recent"] * recent_score
                + self.weights["popularity"] * pop_score
            )

            candidate = {
                "movie_id": movie_id,
                "hybrid_score": hybrid_score,
                "svd_pred": round(svd_pred, 2),
            }

            if explain:
                candidate["breakdown"] = {
                    "svd": round(svd_norm, 3),
                    "content": round(content_score, 3),
                    "recent": round(recent_score, 3),
                    "popularity": round(pop_score, 3),
                }

            candidates.append(candidate)

        # 5. Sort by hybrid score
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 6. Apply diversity re-ranking (MMR-like)
        final = self._diversify(candidates, top_n)

        # 7. Enrich with movie info
        for item in final:
            info = self.movie_info.get(item["movie_id"], {})
            item["title"] = info.get("title", f"Movie {item['movie_id']}")
            item["genres"] = info.get("genres", [])
            item["hybrid_score"] = round(item["hybrid_score"], 4)

        elapsed = (time.time() - start) * 1000
        if self.verbose:
            print(f"   Hybrid recommend for user {user_id}: {len(final)} movies in {elapsed:.1f}ms")

        return final

    def _diversify(
        self,
        candidates: list[dict],
        top_n: int,
        lambda_div: float = 0.3,
    ) -> list[dict]:
        """
        MMR-style diversification.

        Tránh recommend toàn phim cùng 1 thể loại.
        Chọn phim vừa score cao VÀ khác biệt với những phim đã chọn.
        """
        if len(candidates) <= top_n:
            return candidates

        selected = [candidates[0]]  # Best score first
        remaining = candidates[1:]

        while len(selected) < top_n and remaining:
            best_score = -1
            best_idx = 0

            selected_genres = set()
            for s in selected:
                genres = self.movie_info.get(s["movie_id"], {}).get("genres", [])
                selected_genres.update(genres)

            for i, cand in enumerate(remaining):
                cand_genres = set(self.movie_info.get(cand["movie_id"], {}).get("genres", []))

                # Novelty = how different from already selected
                if selected_genres:
                    overlap = len(cand_genres & selected_genres) / max(len(cand_genres | selected_genres), 1)
                    novelty = 1.0 - overlap
                else:
                    novelty = 1.0

                # MMR score = (1-λ)*relevance + λ*novelty
                mmr = (1 - lambda_div) * cand["hybrid_score"] + lambda_div * novelty

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def explain_recommendation(self, user_id: int, movie_id: int) -> dict:
        """
        Giải thích TẠI SAO recommend phim này cho user.

        Trả về breakdown chi tiết từng yếu tố.
        """
        profile = self.get_user_profile(user_id)
        user_genre_vec = profile["overall"]["genre_vector"]
        recent_genre_vec = profile["recent_genre_vector"]

        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return {"error": f"Movie {movie_id} not found"}

        movie_genre_vec = self.genre_matrix[idx]

        # Scores
        svd_pred = self.svd_model.predict(user_id, movie_id).est
        content = content_based_score(user_genre_vec, movie_genre_vec)
        recent = content_based_score(recent_genre_vec, movie_genre_vec)
        pop = self.popularity_map.get(movie_id, 0.0)

        # Genre overlap explanation
        info = self.movie_info.get(movie_id, {})
        movie_genres = set(info.get("genres", []))
        user_top = [g for g, _ in profile["overall"]["top_genres"][:5]]
        matching_genres = list(movie_genres & set(user_top))

        # Similar movies the user has liked
        similar = find_similar_movies(
            movie_id, self.genre_matrix, self.movie_id_to_idx,
            self.idx_to_movie_id, top_n=5,
        )
        user_rated = set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"])
        similar_liked = []
        for sim_id, sim_score in similar:
            if sim_id in user_rated:
                sim_info = self.movie_info.get(sim_id, {})
                similar_liked.append({
                    "movie_id": sim_id,
                    "title": sim_info.get("title", ""),
                    "similarity": sim_score,
                })

        return {
            "movie": {
                "id": movie_id,
                "title": info.get("title", ""),
                "genres": list(movie_genres),
            },
            "scores": {
                "svd_prediction": round(svd_pred, 2),
                "content_match": round(content, 3),
                "recent_taste_match": round(recent, 3),
                "popularity": round(pop, 3),
            },
            "reasons": {
                "matching_genres": matching_genres,
                "user_top_genres": user_top,
                "similar_movies_user_liked": similar_liked[:3],
                "recent_genre_trend": profile["recent"]["recent_genres"][:3],
            },
        }


# =============================================================================
# Batch Generation (for pipeline)
# =============================================================================

def generate_hybrid_top_n(
    svd_model,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    weights: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Generate hybrid Top-N for ALL users (batch mode for pipeline).

    Returns: {user_id: [{"movie_id": ..., "hybrid_score": ..., ...}, ...]}
    """
    if verbose:
        print(f"\nGenerating Hybrid Top-{n} recommendations for all users...")

    start = time.time()

    recommender = HybridRecommender(
        svd_model, ratings_df, movies_df,
        weights=weights, verbose=verbose,
    )

    all_users = ratings_df["userId"].unique()
    results = {}
    total = len(all_users)

    for i, user_id in enumerate(all_users):
        recs = recommender.recommend(int(user_id), top_n=n, explain=False)
        results[int(user_id)] = recs

        if verbose and (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{total} users ({(i+1)/total*100:.0f}%)")

    elapsed = time.time() - start

    if verbose:
        print(f"\n   Hybrid recommendations generated")
        print(f"   Users: {len(results)}")
        print(f"   Time: {elapsed:.1f}s")
        if 1 in results:
            print(f"\n   Sample (User 1):")
            for r in results[1][:5]:
                print(f"     {r['title']}: hybrid={r['hybrid_score']:.3f}, svd={r['svd_pred']}")

    return results
