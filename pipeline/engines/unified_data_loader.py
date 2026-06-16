"""
Unified Data Loader — Single ETL producing 4 data formats for 4 engines.

Outputs:
  1. Sparse CSR matrix       -> Funk-SVD, TrustSVD
  2. Bipartite graph adjacency -> LightGCN
  3. Implicit trust matrix    -> TrustSVD
  4. Sequential sliding windows -> SASRec

All outputs share the same user/item index mappings so that engine
outputs are directly comparable in the benchmark arena.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


class UnifiedDataLoader:
    """
    Transforms raw MovieLens-100k ratings into 4 mathematical formats
    required by the multi-engine recommendation platform.
    """

    def __init__(
        self,
        filepath: str,
        min_interactions: int = 5,
        seq_max_len: int = 50,
    ):
        self.filepath = filepath
        self.min_interactions = min_interactions
        self.seq_max_len = seq_max_len

        self.df: pd.DataFrame = pd.DataFrame()
        self.user_mapping: Dict[int, int] = {}   # idx -> raw_id
        self.item_mapping: Dict[int, int] = {}   # idx -> raw_id
        self.num_users: int = 0
        self.num_items: int = 0

    # ------------------------------------------------------------------
    # Step 1: Load & Clean
    # ------------------------------------------------------------------
    def load_and_clean(self) -> pd.DataFrame:
        """Load u.data, filter cold users/items, create contiguous indices."""
        columns = ["user_id", "item_id", "rating", "timestamp"]
        df = pd.read_csv(self.filepath, sep="\t", names=columns)

        # Filter users and items with fewer than min_interactions
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index

        df = df[
            df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)
        ].copy()

        # Create contiguous integer indices
        df["user_idx"] = df["user_id"].astype("category").cat.codes
        df["item_idx"] = df["item_id"].astype("category").cat.codes

        self.user_mapping = dict(
            enumerate(df["user_id"].astype("category").cat.categories)
        )
        self.item_mapping = dict(
            enumerate(df["item_id"].astype("category").cat.categories)
        )

        self.df = df.sort_values(by=["user_idx", "timestamp"]).reset_index(drop=True)
        self.num_users = self.df["user_idx"].nunique()
        self.num_items = self.df["item_idx"].nunique()

        print(f"  UnifiedDataLoader: {len(self.df):,} ratings "
              f"({self.num_users} users x {self.num_items} items)")
        return self.df

    # ------------------------------------------------------------------
    # Format 1: Sparse Interaction Matrix (for Funk-SVD & TrustSVD)
    # ------------------------------------------------------------------
    def build_sparse_matrix(self) -> sp.csr_matrix:
        """Build user-item rating matrix in CSR format."""
        ratings = self.df["rating"].values.astype(np.float32)
        interaction_matrix = sp.csr_matrix(
            (ratings, (self.df["user_idx"].values, self.df["item_idx"].values)),
            shape=(self.num_users, self.num_items),
        )
        return interaction_matrix

    # ------------------------------------------------------------------
    # Format 2: Bipartite Graph (for LightGCN)
    # ------------------------------------------------------------------
    def build_bipartite_graph(self) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        Build symmetric normalized adjacency matrix for the bipartite
        user-item graph.

        Returns:
            edge_index:   (2, E) array of (user_idx, item_idx) edges
            sym_adj_mat:  D^{-1/2} A D^{-1/2} sparse matrix
                          shape (num_users + num_items, num_users + num_items)
        """
        users = self.df["user_idx"].values
        items = self.df["item_idx"].values

        edge_index = np.vstack((users, items))

        # Build bipartite adjacency: [[0, R], [R^T, 0]]
        R = sp.coo_matrix(
            (np.ones(len(users), dtype=np.float32), (users, items)),
            shape=(self.num_users, self.num_items),
        )
        adj_mat = sp.bmat([[None, R], [R.T, None]], format="csr")

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        sym_adj_mat = D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt).tocsr()
        return edge_index, sym_adj_mat

    # ------------------------------------------------------------------
    # Format 3: Implicit Trust Matrix (for TrustSVD)
    # ------------------------------------------------------------------
    def build_implicit_trust_matrix(self, threshold: float = 0.3) -> sp.csr_matrix:
        """
        Construct an implicit trust network via Jaccard similarity on
        co-interacted items. Two users are "trusted neighbors" if their
        Jaccard coefficient exceeds `threshold`.

        Returns:
            trust_matrix: (num_users x num_users) sparse CSR
        """
        # Binarize interactions
        binary = self.build_sparse_matrix().copy()
        binary.data = np.ones_like(binary.data)

        # |I_u intersect I_v| = R * R^T
        intersection = binary.dot(binary.T)
        user_degrees = np.array(binary.sum(axis=1)).flatten()

        # Build trust matrix by Jaccard thresholding
        trust = sp.lil_matrix((self.num_users, self.num_users), dtype=np.float32)
        coo_inter = intersection.tocoo()

        for u, v, inter_val in zip(coo_inter.row, coo_inter.col, coo_inter.data):
            if u != v:
                union_val = user_degrees[u] + user_degrees[v] - inter_val
                jaccard = inter_val / union_val if union_val > 0 else 0
                if jaccard > threshold:
                    trust[u, v] = jaccard

        return trust.tocsr()

    # ------------------------------------------------------------------
    # Format 4: Sequential Sliding Windows (for SASRec)
    # ------------------------------------------------------------------
    def build_sequential_windows(self) -> Dict[int, np.ndarray]:
        """
        For each user, extract a time-ordered sequence of item interactions
        padded/truncated to `seq_max_len`.

        Returns:
            Dict[user_idx, np.ndarray of shape (seq_max_len,)]
        """
        sequences: Dict[int, np.ndarray] = {}
        grouped = self.df.groupby("user_idx")

        for user_id, group in grouped:
            seq = group["item_idx"].tolist()
            # SASRec uses 1-indexed items (0 = padding)
            seq = [s + 1 for s in seq]

            if len(seq) < self.seq_max_len:
                seq = [0] * (self.seq_max_len - len(seq)) + seq
            else:
                seq = seq[-self.seq_max_len :]

            sequences[int(user_id)] = np.array(seq, dtype=np.int32)

        return sequences

    # ------------------------------------------------------------------
    # Convenience: Build all formats at once
    # ------------------------------------------------------------------
    def build_all(self) -> dict:
        """
        Build all 4 data representations in one call.

        Returns:
            dict with keys:
              - "df":                  cleaned DataFrame
              - "interaction_matrix":  sp.csr_matrix
              - "edge_index":          np.ndarray
              - "sym_adj_mat":         sp.csr_matrix
              - "trust_matrix":        sp.csr_matrix
              - "sequential_windows":  Dict[int, np.ndarray]
              - "num_users":           int
              - "num_items":           int
        """
        if self.df.empty:
            self.load_and_clean()

        print("  Building sparse interaction matrix...")
        interaction_mat = self.build_sparse_matrix()

        print("  Building bipartite graph (LightGCN)...")
        edge_index, sym_adj_mat = self.build_bipartite_graph()

        print("  Building implicit trust matrix (TrustSVD)...")
        trust_mat = self.build_implicit_trust_matrix()

        print("  Building sequential windows (SASRec)...")
        seq_windows = self.build_sequential_windows()

        print(f"  All data formats ready.")
        return {
            "df": self.df,
            "interaction_matrix": interaction_mat,
            "edge_index": edge_index,
            "sym_adj_mat": sym_adj_mat,
            "trust_matrix": trust_mat,
            "sequential_windows": seq_windows,
            "num_users": self.num_users,
            "num_items": self.num_items,
        }
