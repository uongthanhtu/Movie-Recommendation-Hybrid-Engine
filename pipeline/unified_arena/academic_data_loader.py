"""
Academic Data Loader -- Ciao dataset preprocessing for the Unified Arena.

Implements the standard academic preprocessing protocol used in
SEPT (KDD'21), DRSoRec (AAAI'26), and other SOTA social recommendation papers:

  1. Download raw CiaoDVD data (movie-ratings.txt + trusts.txt)
  2. 5-core iterative filtering (users & items with >= 5 interactions)
  3. Contiguous ID re-mapping
  4. Symmetric undirected trust matrix: A_social = A_trust + A_trust^T
  5. Stratified 80/20 train/test split per user

Reference:
  Guo et al., "ETAF: An Extended Trust Antecedents Framework", ASONAM 2014.
  Yu et al., "Socially-Aware Self-Supervised Tri-Training", KDD 2021.
"""
from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Download URLs (multiple fallbacks)
# ---------------------------------------------------------------------------
_CIAO_URLS: List[str] = [
    # LibRec mirror (Guibing Guo)
    "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
    # daicoolb GitHub mirror
    "https://raw.githubusercontent.com/daicoolb/RecommenderSystem-DataSet/master/CiaoDVD/CiaoDVD.zip",
]

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Typed output container
# ---------------------------------------------------------------------------
@dataclass
class ArenaDataset:
    """Preprocessed dataset ready for the benchmark arena."""
    num_users: int
    num_items: int
    train_csr: sp.csr_matrix        # (num_users, num_items) binary implicit
    test_dict: Dict[int, Set[int]]  # {user_idx: set(item_idx)}
    train_dict: Dict[int, Set[int]] # {user_idx: set(item_idx)}
    social_csr: sp.csr_matrix       # (num_users, num_users) symmetric binary

    # Raw stats for reporting
    n_train_interactions: int = 0
    n_test_interactions: int = 0
    n_trust_links: int = 0
    n_raw_interactions: int = 0
    n_raw_users: int = 0
    n_raw_items: int = 0
    filtering_rounds: int = 0


# ---------------------------------------------------------------------------
# Main Loader Class
# ---------------------------------------------------------------------------
class AcademicDataLoader:
    """
    End-to-end CiaoDVD dataset handler with 5-core filtering.

    Usage:
        loader = AcademicDataLoader(data_dir="data/ciao")
        dataset = loader.load()  # downloads, preprocesses, splits
    """

    def __init__(
        self,
        data_dir: str = "data/ciao",
        k_core: int = 5,
        test_ratio: float = 0.2,
        seed: int = 42,
        rating_threshold: float = 3.0,
    ):
        self.data_dir = data_dir
        self.k_core = k_core
        self.test_ratio = test_ratio
        self.seed = seed
        self.rating_threshold = rating_threshold

        self._ratings_path = ""
        self._trust_path = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> ArenaDataset:
        """Full pipeline: download -> parse -> filter -> split -> matrices."""
        self._ensure_downloaded()
        self._resolve_paths()

        print(f"  [DataLoader] Parsing raw files ...", flush=True)
        df_ratings = self._parse_ratings()
        df_trust = self._parse_trust()

        n_raw = len(df_ratings)
        n_raw_users = df_ratings["user"].nunique()
        n_raw_items = df_ratings["item"].nunique()
        print(f"    Raw: {n_raw:,} interactions, {n_raw_users:,} users, {n_raw_items:,} items", flush=True)

        # Convert to implicit feedback (rating >= threshold)
        df_ratings = df_ratings[df_ratings["rating"] >= self.rating_threshold].copy()
        print(f"    After threshold >= {self.rating_threshold}: {len(df_ratings):,} interactions", flush=True)

        # 5-core filtering
        df_ratings, n_rounds = self._k_core_filter(df_ratings, self.k_core)

        # Build contiguous ID mappings
        user_map, item_map = self._build_id_maps(df_ratings)
        num_users = len(user_map)
        num_items = len(item_map)

        # Re-index ratings
        df_ratings["u_idx"] = df_ratings["user"].map(user_map)
        df_ratings["i_idx"] = df_ratings["item"].map(item_map)
        df_ratings = df_ratings.dropna(subset=["u_idx", "i_idx"])
        df_ratings["u_idx"] = df_ratings["u_idx"].astype(int)
        df_ratings["i_idx"] = df_ratings["i_idx"].astype(int)

        print(f"    After {self.k_core}-core: {len(df_ratings):,} interactions, "
              f"{num_users:,} users, {num_items:,} items ({n_rounds} rounds)", flush=True)

        # Train/Test split
        df_train, df_test = self._stratified_split(df_ratings)
        print(f"    Split: Train={len(df_train):,} | Test={len(df_test):,}", flush=True)

        # Build interaction matrices
        train_csr = self._build_interaction_matrix(df_train, num_users, num_items)
        train_dict = self._build_dict(df_train)
        test_dict = self._build_dict(df_test)

        # Build social matrix
        social_csr = self._build_social_matrix(df_trust, user_map, num_users)
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)

        return ArenaDataset(
            num_users=num_users,
            num_items=num_items,
            train_csr=train_csr,
            test_dict=test_dict,
            train_dict=train_dict,
            social_csr=social_csr,
            n_train_interactions=len(df_train),
            n_test_interactions=len(df_test),
            n_trust_links=social_csr.nnz,
            n_raw_interactions=n_raw,
            n_raw_users=n_raw_users,
            n_raw_items=n_raw_items,
            filtering_rounds=n_rounds,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def _ensure_downloaded(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        if self._find_files():
            print(f"  [DataLoader] Dataset files found in {self.data_dir}", flush=True)
            return

        zip_path = os.path.join(self.data_dir, "CiaoDVD.zip")
        for url in _CIAO_URLS:
            try:
                print(f"  [DataLoader] Downloading from {url} ...", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                with open(zip_path, "wb") as f:
                    f.write(data)
                print(f"  [DataLoader] Downloaded {len(data)/1024/1024:.1f} MB", flush=True)
                break
            except Exception as e:
                print(f"  [DataLoader] Failed: {e}", flush=True)
                continue
        else:
            raise RuntimeError(
                f"Could not download CiaoDVD from any source.\n"
                f"Please manually place movie-ratings.txt and trusts.txt in {self.data_dir}"
            )

        # Extract
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            print(f"  [DataLoader] Extracted to {self.data_dir}", flush=True)

    def _find_files(self) -> bool:
        """Check if data files exist anywhere under data_dir."""
        for root, _, files in os.walk(self.data_dir):
            fl = {f.lower() for f in files}
            if ("movie-ratings.txt" in fl or "ratings.txt" in fl) and "trusts.txt" in fl:
                return True
        return False

    def _resolve_paths(self) -> None:
        """Locate rating and trust files under data_dir."""
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                fl = f.lower()
                full = os.path.join(root, f)
                if fl in ("movie-ratings.txt", "ratings.txt") and not self._ratings_path:
                    self._ratings_path = full
                elif fl == "trusts.txt" and not self._trust_path:
                    self._trust_path = full

        if not self._ratings_path:
            raise FileNotFoundError(f"ratings file not found under {self.data_dir}")
        if not self._trust_path:
            raise FileNotFoundError(f"trusts.txt not found under {self.data_dir}")

        print(f"  [DataLoader] Resolved: ratings={self._ratings_path}", flush=True)
        print(f"  [DataLoader] Resolved: trust  ={self._trust_path}", flush=True)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_ratings(self) -> pd.DataFrame:
        """
        Parse CiaoDVD movie-ratings.txt.
        Format: userId, movieId, categoryId, reviewId, movieRating, date
        OR generic: user item rating (space/tab separated)
        """
        rows: List[Tuple[str, str, float]] = []
        with open(self._ratings_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                # Try comma-separated (CiaoDVD format)
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()

                if len(parts) >= 5:
                    # CiaoDVD: userId, movieId, categoryId, reviewId, movieRating, ...
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[4].strip())))
                elif len(parts) >= 3:
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
                elif len(parts) == 2:
                    rows.append((parts[0].strip(), parts[1].strip(), 1.0))

        return pd.DataFrame(rows, columns=["user", "item", "rating"])

    def _parse_trust(self) -> pd.DataFrame:
        """
        Parse CiaoDVD trusts.txt.
        Format: trustorId, trusteeId, trustValue
        OR: user_a user_b weight
        """
        rows: List[Tuple[str, str, float]] = []
        with open(self._trust_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()

                if len(parts) >= 3:
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
                elif len(parts) == 2:
                    rows.append((parts[0].strip(), parts[1].strip(), 1.0))

        return pd.DataFrame(rows, columns=["src", "dst", "weight"])

    # ------------------------------------------------------------------
    # 5-Core Iterative Filtering
    # ------------------------------------------------------------------
    @staticmethod
    def _k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]:
        """
        Iteratively remove users and items with fewer than k interactions
        until convergence (no more removals).
        """
        n_rounds = 0
        while True:
            n_before = len(df)

            # Filter users
            user_counts = df["user"].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df["user"].isin(valid_users)]

            # Filter items
            item_counts = df["item"].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df["item"].isin(valid_items)]

            n_rounds += 1
            if len(df) == n_before:
                break

        return df.reset_index(drop=True), n_rounds

    # ------------------------------------------------------------------
    # ID Mapping & Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _build_id_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build contiguous 0-indexed ID maps, sorted numerically where possible."""
        def _sort_key(x: str):
            try:
                return (0, int(x))
            except ValueError:
                return (1, x)

        users = sorted(df["user"].unique(), key=_sort_key)
        items = sorted(df["item"].unique(), key=_sort_key)
        return {u: i for i, u in enumerate(users)}, {it: i for i, it in enumerate(items)}

    def _stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Per-user leave-N-out: hold out test_ratio of each user's items."""
        rng = np.random.default_rng(self.seed)
        train_idx: List[int] = []
        test_idx: List[int] = []

        for _, group in df.groupby("u_idx"):
            indices = group.index.tolist()
            n_test = max(1, int(len(indices) * self.test_ratio))
            if len(indices) < 2:
                train_idx.extend(indices)
            else:
                rng.shuffle(indices)
                test_idx.extend(indices[:n_test])
                train_idx.extend(indices[n_test:])

        return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sparse Matrix Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
        rows = df["u_idx"].values.astype(np.int64)
        cols = df["i_idx"].values.astype(np.int64)
        vals = np.ones(len(rows), dtype=np.float32)
        mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        mat.data[:] = 1.0  # binarize duplicates
        return mat

    @staticmethod
    def _build_social_matrix(
        df_trust: pd.DataFrame, user_map: Dict[str, int], n_users: int
    ) -> sp.csr_matrix:
        """Build symmetric undirected trust matrix: A = A_raw + A_raw^T."""
        df = df_trust.copy()
        df["s_idx"] = df["src"].map(user_map)
        df["d_idx"] = df["dst"].map(user_map)
        df = df.dropna(subset=["s_idx", "d_idx"])
        df["s_idx"] = df["s_idx"].astype(int)
        df["d_idx"] = df["d_idx"].astype(int)

        if len(df) == 0:
            return sp.csr_matrix((n_users, n_users), dtype=np.float32)

        rows = df["s_idx"].values
        cols = df["d_idx"].values
        vals = np.ones(len(rows), dtype=np.float32)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_users))
        A_sym = (A + A.T).tocsr()
        A_sym.data = np.minimum(A_sym.data, 1.0)
        return A_sym

    @staticmethod
    def _build_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
        result: Dict[int, Set[int]] = {}
        for _, row in df.iterrows():
            result.setdefault(int(row["u_idx"]), set()).add(int(row["i_idx"]))
        return result
