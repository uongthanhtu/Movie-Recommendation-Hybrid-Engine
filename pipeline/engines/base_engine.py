"""
Base Recommender Engine — Abstract interface for all recommendation algorithms.

All engines MUST implement:
  - fit(data)            : Train the model on provided data
  - predict_rating(u, i) : Predict a single rating score
  - recommend_top_n(u, n): Generate Top-N item list for a user
  - save_model(path)     : Persist model to disk
  - load_model(path)     : Restore model from disk
"""
import abc
from typing import Any, List


class BaseRecommenderEngine(abc.ABC):
    """Technical contract that every recommendation engine must fulfill."""

    @abc.abstractmethod
    def fit(self, data: Any) -> None:
        """Train the model. The `data` type varies per engine."""

    @abc.abstractmethod
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict a scalar relevance score for (user, item)."""

    @abc.abstractmethod
    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        """Return ordered list of top-N item IDs for a user."""

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """Serialize model artifacts to `path`."""

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """Deserialize model artifacts from `path`."""
