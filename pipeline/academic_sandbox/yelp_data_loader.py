"""
Yelp Data Loader -- Isolated data downloader & parser for the QRec Yelp benchmark dataset.

Downloads the pre-processed Yelp dataset used in SEPT (KDD'21) from the author's
Dropbox mirror, then parses ratings.txt and trusts.txt into contiguous-index
scipy sparse matrices ready for Social-LightGCN training and evaluation.

Actual QRec Yelp format (from the Dropbox ZIP):
    ratings.txt:  user_id  item_id  click   (space-separated, implicit binary)
    trusts.txt:   user_a   user_b   follow  (space-separated, bidirectional)

Since the QRec Yelp dataset does NOT provide a pre-split train/test,
this loader performs a stratified 80/20 leave-N-out split per user.

Reference:
    Yu et al., "Socially-Aware Self-Supervised Tri-Training for Recommendation", KDD 2021.
    Repository: https://github.com/Coder-Yu/QRec
"""
import os
import io
import zipfile
import urllib.request
from typing import Dict, Tuple, Set, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Dropbox mirror link from the official QRec README
# (https://github.com/Coder-Yu/QRec - "Related Datasets" section)
# dl=1 forces a direct download instead of a preview page.
# ---------------------------------------------------------------------------
YELP_DROPBOX_URL = (
    "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1"
)


class YelpDataLoader:
    """
    End-to-end Yelp dataset handler for the Academic Benchmark Sandbox.

    Lifecycle:
        1. download()   -- fetch & extract raw files into *data_dir*
        2. load_data()  -- parse text files, build ID mappings, split train/test
        3. get_train_interaction_matrix() / get_trust_matrix() -- sparse matrices
        4. get_test_dict()  -- {user_idx: set(item_idx)} for ranking evaluation
    """

    def __init__(self, data_dir: str = "data/yelp", test_ratio: float = 0.2, seed: int = 42):
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.seed = seed

        # Paths (resolved after download / file-scan)
        self.ratings_path: str = ""
        self.trust_path: str = ""

        # ID mappings (original string -> contiguous int)
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.num_users: int = 0
        self.num_items: int = 0

        # Parsed DataFrames
        self._df_train: pd.DataFrame = pd.DataFrame()
        self._df_test: pd.DataFrame = pd.DataFrame()
        self._df_trust: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Download & Extract
    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> None:
        """Download the Yelp dataset ZIP from Dropbox and extract text files."""
        os.makedirs(self.data_dir, exist_ok=True)

        # Check if files already exist
        if not force and self._files_exist():
            print(f"  [YelpDataLoader] Dataset files already present in {self.data_dir}")
            self._resolve_paths()
            return

        zip_path = os.path.join(self.data_dir, "yelp_qrec.zip")
        print(f"  [YelpDataLoader] Downloading Yelp dataset from Dropbox ...")

        req = urllib.request.Request(
            YELP_DROPBOX_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()

        with open(zip_path, "wb") as f:
            f.write(data)
        print(f"  [YelpDataLoader] Downloaded {len(data) / 1024 / 1024:.1f} MB -> {zip_path}")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)
        print(f"  [YelpDataLoader] Extracted to {self.data_dir}")

        self._resolve_paths()

    def _files_exist(self) -> bool:
        """Check if ratings and trust files already exist somewhere under data_dir."""
        found_ratings = False
        found_trust = False
        for root, _, files in os.walk(self.data_dir):
            lower_files = {f.lower() for f in files}
            if "ratings.txt" in lower_files or "train.txt" in lower_files:
                found_ratings = True
            if "trusts.txt" in lower_files or "trust.txt" in lower_files:
                found_trust = True
        return found_ratings

    def _resolve_paths(self) -> None:
        """Walk data_dir to locate ratings.txt / trusts.txt (case-insensitive)."""
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                fl = f.lower()
                full = os.path.join(root, f)
                if fl in ("ratings.txt", "train.txt") and not self.ratings_path:
                    self.ratings_path = full
                elif fl in ("trusts.txt", "trust.txt", "trustnetwork.txt", "links.txt") and not self.trust_path:
                    self.trust_path = full

        if not self.ratings_path:
            raise FileNotFoundError(f"ratings.txt / train.txt not found under {self.data_dir}")

        print(f"  [YelpDataLoader] Resolved paths:")
        print(f"    ratings : {self.ratings_path}")
        print(f"    trust   : {self.trust_path or '(not found - will proceed without social data)'}")

    # ------------------------------------------------------------------
    # 2. Parse & Split
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        """Parse text files, build contiguous ID mappings, and split train/test."""
        if not self.ratings_path:
            self._resolve_paths()

        # Read all interactions
        df_all = self._read_interaction_file(self.ratings_path)
        print(f"  [YelpDataLoader] Loaded {len(df_all):,} interactions from {os.path.basename(self.ratings_path)}")

        # Parse trust data (if available)
        if self.trust_path and os.path.exists(self.trust_path):
            self._df_trust = self._read_trust_file(self.trust_path)
            print(f"  [YelpDataLoader] Loaded {len(self._df_trust):,} trust links from {os.path.basename(self.trust_path)}")
        else:
            self._df_trust = pd.DataFrame(columns=["src", "dst", "weight"])

        # Collect all unique user / item IDs
        all_users: Set[str] = set(df_all["user"].unique())
        all_items: Set[str] = set(df_all["item"].unique())
        if len(self._df_trust) > 0:
            all_users.update(self._df_trust["src"].unique())
            all_users.update(self._df_trust["dst"].unique())

        # Build contiguous ID mappings (numeric sort for reproducibility)
        sorted_users = sorted(all_users, key=lambda x: int(x) if x.isdigit() else x)
        sorted_items = sorted(all_items, key=lambda x: int(x) if x.isdigit() else x)
        self.user_map = {uid: idx for idx, uid in enumerate(sorted_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(sorted_items)}
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)

        # Map IDs to contiguous indices in the full DataFrame
        df_all["u_idx"] = df_all["user"].map(self.user_map)
        df_all["i_idx"] = df_all["item"].map(self.item_map)

        # Stratified train/test split: for each user, hold out test_ratio of items
        self._df_train, self._df_test = self._stratified_split(df_all, self.test_ratio, self.seed)

        # Map trust indices
        if len(self._df_trust) > 0:
            self._df_trust["s_idx"] = self._df_trust["src"].map(self.user_map)
            self._df_trust["d_idx"] = self._df_trust["dst"].map(self.user_map)
            self._df_trust = self._df_trust.dropna(subset=["s_idx", "d_idx"])
            self._df_trust["s_idx"] = self._df_trust["s_idx"].astype(int)
            self._df_trust["d_idx"] = self._df_trust["d_idx"].astype(int)

        print(f"\n  [YelpDataLoader] Dataset statistics:")
        print(f"    Users : {self.num_users:,}")
        print(f"    Items : {self.num_items:,}")
        print(f"    Train : {len(self._df_train):,} interactions")
        print(f"    Test  : {len(self._df_test):,} interactions")
        print(f"    Trust : {len(self._df_trust):,} social links")
        density = len(df_all) / (self.num_users * self.num_items) * 100
        print(f"    Density : {density:.4f}%")

    @staticmethod
    def _stratified_split(
        df: pd.DataFrame, test_ratio: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Leave-N-out split: for each user, randomly hold out test_ratio of items.
        Users with fewer than 2 interactions go entirely to training.
        """
        rng = np.random.default_rng(seed)
        train_indices: List[int] = []
        test_indices: List[int] = []

        grouped = df.groupby("u_idx")
        for _, group in grouped:
            indices = group.index.tolist()
            n_test = max(1, int(len(indices) * test_ratio))

            if len(indices) < 2:
                train_indices.extend(indices)
            else:
                rng.shuffle(indices)
                test_indices.extend(indices[:n_test])
                train_indices.extend(indices[n_test:])

        return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)

    @staticmethod
    def _read_interaction_file(path: str) -> pd.DataFrame:
        """Read a QRec-format interaction file (user item rating/click)."""
        rows: List[Tuple[str, str, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    rows.append((parts[0], parts[1], float(parts[2])))
                elif len(parts) == 2:
                    rows.append((parts[0], parts[1], 1.0))
        return pd.DataFrame(rows, columns=["user", "item", "rating"])

    @staticmethod
    def _read_trust_file(path: str) -> pd.DataFrame:
        """Read a QRec-format trust file (source target weight)."""
        rows: List[Tuple[str, str, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    rows.append((parts[0], parts[1], float(parts[2])))
                elif len(parts) == 2:
                    rows.append((parts[0], parts[1], 1.0))
        return pd.DataFrame(rows, columns=["src", "dst", "weight"])

    # ------------------------------------------------------------------
    # 3. Build Sparse Matrices
    # ------------------------------------------------------------------
    def get_train_interaction_matrix(self) -> sp.csr_matrix:
        """Build a (num_users x num_items) CSR interaction matrix from training data."""
        rows = self._df_train["u_idx"].values.astype(np.int64)
        cols = self._df_train["i_idx"].values.astype(np.int64)
        vals = np.ones(len(rows), dtype=np.float32)  # implicit feedback
        mat = sp.csr_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_items))
        # Binarize (in case of duplicate interactions)
        mat.data[:] = 1.0
        return mat

    def get_trust_matrix(self) -> sp.csr_matrix:
        """
        Build a symmetric undirected trust matrix (num_users x num_users).
        Per the README: "social relations are bidirectional in this dataset"
        We still symmetrize to be safe: A_social = A_trust + A_trust^T
        """
        if len(self._df_trust) == 0:
            return sp.csr_matrix((self.num_users, self.num_users), dtype=np.float32)

        rows = self._df_trust["s_idx"].values.astype(np.int64)
        cols = self._df_trust["d_idx"].values.astype(np.int64)
        vals = self._df_trust["weight"].values.astype(np.float32)

        A_trust = sp.coo_matrix(
            (vals, (rows, cols)), shape=(self.num_users, self.num_users)
        )
        A_social = (A_trust + A_trust.T).tocsr()
        A_social.data = np.minimum(A_social.data, 1.0)  # clip to binary
        return A_social

    # ------------------------------------------------------------------
    # 4. Test / Train Dictionaries for Ranking Evaluation
    # ------------------------------------------------------------------
    def get_test_dict(self) -> Dict[int, Set[int]]:
        """Return {user_idx: set(item_idx)} of ground-truth test interactions."""
        test_dict: Dict[int, Set[int]] = {}
        for _, row in self._df_test.iterrows():
            u = int(row["u_idx"])
            i = int(row["i_idx"])
            test_dict.setdefault(u, set()).add(i)
        return test_dict

    def get_train_dict(self) -> Dict[int, Set[int]]:
        """Return {user_idx: set(item_idx)} of training interactions (for exclusion during eval)."""
        train_dict: Dict[int, Set[int]] = {}
        for _, row in self._df_train.iterrows():
            u = int(row["u_idx"])
            i = int(row["i_idx"])
            train_dict.setdefault(u, set()).add(i)
        return train_dict
