"""Test hybrid recommender with existing SVD model + MovieLens data."""
import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_loader import load_ratings, load_movies
from pipeline.hybrid_recommender import HybridRecommender


def main():
    print("=" * 70)
    print(" TESTING HYBRID RECOMMENDER")
    print("=" * 70)

    # Load data
    ratings = load_ratings("data/ml-100k")
    movies = load_movies("data/ml-100k")

    # Load trained SVD model
    with open("models/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    # Create hybrid recommender
    hybrid = HybridRecommender(svd_model, ratings, movies, verbose=True)

    # =====================================================
    # Test 1: User profile
    # =====================================================
    print("\n" + "=" * 70)
    print(" TEST 1: User 1 Genre Profile")
    print("=" * 70)

    profile = hybrid.get_user_profile(1)
    print(f"\n  Total rated: {profile['overall']['total_rated']}")
    print(f"  Avg rating:  {profile['overall']['avg_rating']}")
    print(f"\n  Top genres (all-time):")
    for g, s in profile["overall"]["top_genres"][:8]:
        bar = "█" * int(s * 30)
        print(f"    {g:15s} {s:.3f} {bar}")

    print(f"\n  Recent genres (last 20):")
    for g, count in profile["recent"]["recent_genres"]:
        print(f"    {g:15s} {count} phim")

    # =====================================================
    # Test 2: Hybrid recommendations
    # =====================================================
    print("\n" + "=" * 70)
    print(" TEST 2: Hybrid Recommendations for User 1")
    print("=" * 70)

    recs = hybrid.recommend(1, top_n=10, explain=True)
    print(f"\n  {'#':>2} {'Score':>6} {'SVD':>5} {'Title':<45} {'Genres'}")
    print("  " + "-" * 95)
    for i, r in enumerate(recs, 1):
        genres = ", ".join(r["genres"][:3])
        bd = r.get("breakdown", {})
        print(f"  {i:2d} {r['hybrid_score']:6.3f} {r['svd_pred']:5.2f} {r['title'][:44]:<45} {genres}")
        if bd:
            print(f"     └── svd={bd.get('svd',0):.2f}  content={bd.get('content',0):.2f}  "
                  f"recent={bd.get('recent',0):.2f}  pop={bd.get('popularity',0):.2f}")

    # =====================================================
    # Test 3: SVD-only vs Hybrid comparison
    # =====================================================
    print("\n" + "=" * 70)
    print(" TEST 3: SVD-only vs Hybrid (User 1)")
    print("=" * 70)

    # SVD-only top 10
    unrated = [m for m in movies["movieId"] if m not in set(ratings[ratings["userId"] == 1]["movieId"])]
    svd_preds = [(mid, svd_model.predict(1, mid).est) for mid in unrated]
    svd_preds.sort(key=lambda x: x[1], reverse=True)
    svd_top10 = svd_preds[:10]

    print(f"\n  {'SVD-only Top 10':<50} {'Hybrid Top 10'}")
    print("  " + "-" * 100)
    for i in range(10):
        svd_mid, svd_score = svd_top10[i]
        svd_title = movies[movies["movieId"] == svd_mid]["title"].values[0] if len(movies[movies["movieId"] == svd_mid]) > 0 else f"Movie {svd_mid}"

        hyb = recs[i] if i < len(recs) else {"title": "-", "hybrid_score": 0}

        marker = " " if hyb["title"] != svd_title[:44] else ""
        print(f"  {i+1:2d}. {svd_title[:35]:<36} ({svd_score:.2f})  │  {hyb['title'][:35]:<36} ({hyb['hybrid_score']:.3f}){marker}")

    # =====================================================
    # Test 4: Explain a specific recommendation
    # =====================================================
    if recs:
        top_movie = recs[0]["movie_id"]
        print(f"\n{'=' * 70}")
        print(f" TEST 4: Why recommend '{recs[0]['title']}' to User 1?")
        print(f"{'=' * 70}")

        explanation = hybrid.explain_recommendation(1, top_movie)

        print(f"\n  Scores:")
        for k, v in explanation["scores"].items():
            print(f"    {k:25s}: {v}")

        print(f"\n  Reasons:")
        print(f"    Matching genres: {explanation['reasons']['matching_genres']}")
        print(f"    User top genres: {explanation['reasons']['user_top_genres']}")
        print(f"    Recent trend:    {explanation['reasons']['recent_genre_trend']}")
        if explanation["reasons"]["similar_movies_user_liked"]:
            print(f"    Similar movies user liked:")
            for s in explanation["reasons"]["similar_movies_user_liked"]:
                print(f"      - {s['title']} (similarity: {s['similarity']})")


if __name__ == "__main__":
    main()
