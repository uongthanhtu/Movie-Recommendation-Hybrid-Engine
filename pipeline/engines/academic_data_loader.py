"""
Academic Data Loader — Isolated data preprocessor for Ciao/Epinions benchmarks.
Handles parsing rating networks, trust networks, symmetrization of trust graphs,
and train/test splitting.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, Any


class AcademicDataLoader:
    """
    Data preprocessor tailored for Epinions and Ciao academic datasets.
    Isolates rating/trust network parsing to prevent production environment side-effects.
    """

    def __init__(self, ratings_path: str, trust_path: str, threshold: float = 3.0):
        self.ratings_path = ratings_path
        self.trust_path = trust_path
        self.threshold = threshold

        self.num_users = 0
        self.num_items = 0
        self.user_mapping: Dict[Any, int] = {}
        self.item_mapping: Dict[Any, int] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse ratings.txt and trust.txt raw dataset files.
        Builds contiguous index mappings for both users and items.
        """
        # Load ratings: user_id, item_id, rating
        try:
            df_ratings = pd.read_csv(
                self.ratings_path, 
                sep=None, 
                names=["user_id", "item_id", "rating"], 
                engine="python"
            )
        except Exception:
            # Fallback if timestamp column exists
            df_ratings = pd.read_csv(
                self.ratings_path, 
                sep=None, 
                names=["user_id", "item_id", "rating", "timestamp"], 
                usecols=[0, 1, 2], 
                engine="python"
            )

        # Lọc rating >= threshold (positive implicit feedback)
        df_ratings = df_ratings[df_ratings["rating"] >= self.threshold].copy()

        # Load trust graph: source_id, target_id, [trust_val]
        try:
            df_trust = pd.read_csv(
                self.trust_path, 
                sep=None, 
                names=["source_id", "target_id", "trust_val"], 
                engine="python"
            )
        except Exception:
            df_trust = pd.read_csv(
                self.trust_path, 
                sep=None, 
                names=["source_id", "target_id"], 
                engine="python"
            )
            df_trust["trust_val"] = 1.0

        # Collect unique users across both ratings and trust mappings
        all_users = pd.concat([df_ratings["user_id"], df_trust["source_id"], df_trust["target_id"]]).unique()
        all_items = df_ratings["item_id"].unique()

        self.user_mapping = {uid: idx for idx, uid in enumerate(all_users)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(all_items)}

        self.num_users = len(all_users)
        self.num_items = len(all_items)

        df_ratings["user_idx"] = df_ratings["user_id"].map(self.user_mapping)
        df_ratings["item_idx"] = df_ratings["item_id"].map(self.item_mapping)

        df_trust["source_idx"] = df_trust["source_id"].map(self.user_mapping)
        df_trust["target_idx"] = df_trust["target_id"].map(self.user_mapping)

        df_ratings = df_ratings.dropna(subset=["user_idx", "item_idx"])
        df_trust = df_trust.dropna(subset=["source_idx", "target_idx"])

        df_ratings["user_idx"] = df_ratings["user_idx"].astype(int)
        df_ratings["item_idx"] = df_ratings["item_idx"].astype(int)
        df_trust["source_idx"] = df_trust["source_idx"].astype(int)
        df_trust["target_idx"] = df_trust["target_idx"].astype(int)

        print(f"  AcademicDataLoader: loaded {len(df_ratings):,} implicit positive ratings "
              f"({self.num_users} users x {self.num_items} items) and {len(df_trust):,} trust links")
        return df_ratings, df_trust

    def build_social_matrix(self, df_trust: pd.DataFrame) -> sp.csr_matrix:
        """
        Convert the directed trust network into an undirected symmetric trust matrix.
        A_social = A_trust + A_trust.T
        """
        sources = df_trust["source_idx"].values
        targets = df_trust["target_idx"].values
        vals = df_trust["trust_val"].values

        A_trust = sp.coo_matrix(
            (vals, (sources, targets)),
            shape=(self.num_users, self.num_users)
        )

        # Symmetric undirected trust graph conversion
        A_social = A_trust + A_trust.T
        return A_social.tocsr()

    def split_data(
        self, 
        df_ratings: pd.DataFrame, 
        ratio: float = 0.8, 
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a global random split on ratings into Train Set (80%) and Test Set (20%).
        """
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(df_ratings))
        split_point = int(len(df_ratings) * ratio)

        train_indices = shuffled_indices[:split_point]
        test_indices = shuffled_indices[split_point:]

        df_train = df_ratings.iloc[train_indices].copy()
        df_test = df_ratings.iloc[test_indices].copy()

        return df_train, df_test
