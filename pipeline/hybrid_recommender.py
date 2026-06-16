"""
Hybrid Recommender — Kết hợp nhiều tín hiệu để recommend.

Combines:
  1. LightGCN Score (Graph Collaborative Filtering) — "users giống bạn thích phim này"
  2. TrustSVD Score (Social Collaborative Filtering) — "những người tin cậy thích phim này"
  3. Content Score (Genre matching) — "phim này cùng thể loại bạn hay xem"
  4. Recent Taste Score — "gần đây bạn đang chuyển sang thích thể loại này"
  5. Diversity Bonus — đảm bảo kết quả đa dạng, không chỉ 1 thể loại

Final Score = w1*LightGCN + w2*TrustSVD + w3*Content + w4*Recent + diversity_adjustment
"""
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

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
    "lightgcn": 0.50,    # Graph-based collaborative filtering (core)
    "trust_svd": 0.15,   # Social-regularized collaborative filtering (ratings)
    "content": 0.25,     # Genre matching with user profile
    "recent": 0.10,      # Recent genre preference shift
    "diversity": 0.05,   # Diversity bonus (MMR weight)
}


# =============================================================================
# Hybrid Recommender
# =============================================================================

class HybridRecommender:
    """
    Multi-signal recommendation engine.

    Kết hợp LightGCN + TrustSVD + Content-Based + Diversity.
    """

    def __init__(
        self,
        lightgcn_engine,
        trust_svd_engine,
        id_mappings: dict,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        weights: dict = None,
        verbose: bool = True,
    ):
        self.lightgcn_engine = lightgcn_engine
        self.trust_svd_engine = trust_svd_engine
        self.id_mappings = id_mappings
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
            if lightgcn_engine is not None:
                print("   |- LightGCN engine loaded")
            if trust_svd_engine is not None:
                print("   |- TrustSVD engine loaded")
            if id_mappings is not None:
                print("   |- ID mappings loaded")
            print(f"   |- Genre matrix: {self.genre_matrix.shape}")
            print(f"   |- Popularity: {len(self.popularity_map)} movies")
            print(f"   |- Weights: {self.weights}")
            print("   - Ready")

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

        if not all_movie_ids or not self.id_mappings:
            return []

        user_idx = self.id_mappings["user_raw_to_idx"].get(user_id)
        if user_idx is None:
            # Cold start fallback if user not in training interaction matrix
            return []

        item_raw_to_idx = self.id_mappings["item_raw_to_idx"]
        candidate_idxs = []
        valid_movie_ids = []
        for mid in all_movie_ids:
            idx = item_raw_to_idx.get(mid)
            if idx is not None:
                candidate_idxs.append(idx)
                valid_movie_ids.append(mid)

        if not valid_movie_ids:
            return []

        # 4. Batch prediction for LightGCN (Sigmoid normalized)
        all_gcn_sigmoid = None
        if self.lightgcn_engine and user_idx is not None:
            with torch.no_grad():
                user_vec = self.lightgcn_engine._user_emb[user_idx].to(self.lightgcn_engine.device)
                all_scores = torch.matmul(self.lightgcn_engine._item_emb, user_vec)
                all_gcn_sigmoid = torch.sigmoid(all_scores).cpu().numpy()

        # 5. Batch prediction for TrustSVD (Scale normalized)
        trust_ratings = None
        trust_scores = None
        if self.trust_svd_engine and user_idx is not None and candidate_idxs:
            trust_ratings = self.trust_svd_engine.predict_batch(user_idx, candidate_idxs)
            trust_scores = (trust_ratings - 1.0) / 4.0  # 1-5 scale -> 0-1

        # 6. Score each candidate
        candidates = []

        # Batch content scores
        content_scores_all = batch_content_scores(user_genre_vec, self.genre_matrix)
        recent_scores_all = batch_content_scores(recent_genre_vec, self.genre_matrix)

        for i, mid in enumerate(valid_movie_ids):
            item_idx = candidate_idxs[i]

            # LightGCN score (Sigmoid)
            gcn_score = float(all_gcn_sigmoid[item_idx]) if all_gcn_sigmoid is not None else 0.0

            # TrustSVD score (Scale)
            trust_score = float(trust_scores[i]) if trust_scores is not None else 0.0

            # Content score (genre match)
            idx = self.movie_id_to_idx.get(mid)
            content_score = float(content_scores_all[idx]) if idx is not None else 0.0

            # Recent taste score
            recent_score = float(recent_scores_all[idx]) if idx is not None else 0.0

            # Hybrid score
            hybrid_score = (
                self.weights["lightgcn"] * gcn_score
                + self.weights["trust_svd"] * trust_score
                + self.weights["content"] * content_score
                + self.weights["recent"] * recent_score
            )

            candidate_rating = float(trust_ratings[i]) if trust_ratings is not None else float(self.trust_svd_engine.mu) if self.trust_svd_engine else 3.0

            candidate = {
                "movie_id": mid,
                "hybrid_score": hybrid_score,
                "predicted_rating": round(candidate_rating, 2),
                "svd_pred": round(candidate_rating, 2),  # Backward compatibility
                "lightgcn_score": round(gcn_score, 4),
            }

            if explain:
                candidate["breakdown"] = {
                    "lightgcn": round(gcn_score, 3),
                    "trust_svd": round(trust_score, 3),
                    "content": round(content_score, 3),
                    "recent": round(recent_score, 3),
                }

            candidates.append(candidate)

        # 7. Sort by hybrid score
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 8. Apply diversity re-ranking (MMR-like)
        final = self._diversify(candidates, top_n)

        # 9. Enrich with movie info
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
        if idx is None or not self.id_mappings:
            return {"error": f"Movie {movie_id} not found"}

        movie_genre_vec = self.genre_matrix[idx]

        user_idx = self.id_mappings["user_raw_to_idx"].get(user_id)
        item_idx = self.id_mappings["item_raw_to_idx"].get(movie_id)

        # LightGCN score
        if self.lightgcn_engine and user_idx is not None and item_idx is not None:
            with torch.no_grad():
                u_emb = self.lightgcn_engine._user_emb[user_idx]
                i_emb = self.lightgcn_engine._item_emb[item_idx]
                dot_val = torch.dot(u_emb, i_emb).item()
                gcn_score = torch.sigmoid(torch.tensor(dot_val)).item()
        else:
            gcn_score = 0.0

        # TrustSVD rating prediction
        if self.trust_svd_engine and user_idx is not None and item_idx is not None:
            trust_pred = self.trust_svd_engine.predict_rating(user_idx, item_idx)
        else:
            trust_pred = float(self.trust_svd_engine.mu) if self.trust_svd_engine else 3.0

        content = content_based_score(user_genre_vec, movie_genre_vec)
        recent = content_based_score(recent_genre_vec, movie_genre_vec)

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
                "lightgcn_match": round(gcn_score, 3),
                "trust_svd_prediction": round(trust_pred, 2),
                "content_match": round(content, 3),
                "recent_taste_match": round(recent, 3),
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
    lightgcn_engine,
    trust_svd_engine,
    id_mappings: dict,
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
        lightgcn_engine, trust_svd_engine, id_mappings,
        ratings_df, movies_df,
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
        print("\n   Hybrid recommendations generated")
        print(f"   Users: {len(results)}")
        print(f"   Time: {elapsed:.1f}s")
        if 1 in results:
            print("\n   Sample (User 1):")
            for r in results[1][:5]:
                print(f"     {r['title']}: hybrid={r['hybrid_score']:.3f}, trust_svd={r['svd_pred']}, gcn={r['lightgcn_score']:.3f}")

    return results

