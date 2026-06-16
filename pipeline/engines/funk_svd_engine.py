"""
Funk-SVD Engine — Baseline collaborative filtering using Surprise library.

Mathematical formulation (Koren 2009):
  r_hat(u,i) = mu + b_u + b_i + p_u^T * q_i

Optimization via SGD on the regularized squared-error loss:
  L = sum( (r_ui - r_hat)^2 ) + lambda * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)
"""
import pickle
from typing import List

import pandas as pd
from surprise import SVD, Dataset, Reader

from pipeline.engines.base_engine import BaseRecommenderEngine


class FunkSVDEngine(BaseRecommenderEngine):
    """Funk-SVD implementation via scikit-surprise."""

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ):
        self.n_factors = n_factors
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            verbose=False,
        )
        self.trainset = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the SVD model.

        Args:
            data: DataFrame with columns [user_idx, item_idx, rating].
        """
        reader = Reader(rating_scale=(1, 5))
        surp_data = Dataset.load_from_df(
            data[["user_idx", "item_idx", "rating"]], reader
        )
        self.trainset = surp_data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        pred = self.model.predict(user_id, item_id)
        return float(pred.est)

    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        all_items = set(self.trainset.all_items())
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            seen_items = {j for j, _ in self.trainset.ur[inner_uid]}
        except ValueError:
            seen_items = set()

        unseen = all_items - seen_items

        predictions = [
            (
                self.trainset.to_raw_iid(iid),
                self.model.predict(user_id, self.trainset.to_raw_iid(iid)).est,
            )
            for iid in unseen
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in predictions[:top_n]]

    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
