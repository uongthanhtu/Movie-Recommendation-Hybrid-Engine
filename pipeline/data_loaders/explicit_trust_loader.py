"""
Explicit Trust Loader -- Generic, config-driven loader for any dataset with real,
explicit trust/social data (Ciao, Yelp, FilmTrust now; more datasets once a future
sub-project registers their configs).

Consolidates pipeline/unified_arena/academic_data_loader.py::AcademicDataLoader,
pipeline/academic_sandbox/yelp_data_loader.py::YelpDataLoader, and
pipeline/filmtrust_arena/filmtrust_loader.py::FilmTrustLoader's parsing logic into one
class parameterized by DatasetConfig. Those three existing loaders and their CLI
runners are left untouched and continue to work -- this is a parallel implementation,
not a replacement (see docs/superpowers/specs/2026-06-24-dataset-factory-design.md
for why retirement is deferred to a later sub-project).

Canonical user universe: this loader unions ratings-file and trust-file users (a
trust-only user with no ratings still gets a row/column in social_csr, with an
all-zero row in train_csr). This matches YelpDataLoader's and FilmTrustLoader's
EXISTING behavior; it differs from AcademicDataLoader's existing ratings-only
behavior. This is a deliberate, documented choice for the new canonical loader --
it does not change AcademicDataLoader's own untouched behavior.
"""
from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.data_loaders.base_loader import ArenaDataset, BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DatasetConfig

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ExplicitTrustLoader(BaseDatasetLoader):
    """
    Generic loader for any dataset with real, explicit trust data, configured
    entirely via a DatasetConfig.

    Usage:
        loader = ExplicitTrustLoader(CIAO_CONFIG)
        dataset = loader.load()
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._ratings_path = ""
        self._trust_path = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> ArenaDataset:
        """Full pipeline: download -> parse -> feedback-mode filter -> k-core -> split -> matrices."""
        cfg = self.config
        self._ensure_downloaded()
        self._resolve_paths()

        print(f"  [ExplicitTrustLoader:{cfg.name}] Parsing raw files ...", flush=True)
        df_ratings = self._parse_rows(self._ratings_path, is_ratings=True)
        df_trust = self._parse_rows(self._trust_path, is_ratings=False)

        n_raw = len(df_ratings)
        n_raw_users = df_ratings["user"].nunique()
        n_raw_items = df_ratings["item"].nunique()
        print(f"    Raw: {n_raw:,} interactions, {n_raw_users:,} users, {n_raw_items:,} items", flush=True)

        # Feedback mode (BEFORE k-core, matching AcademicDataLoader's existing order)
        if cfg.feedback_mode == "threshold_binarize":
            df_ratings = df_ratings[df_ratings["rating"] >= cfg.rating_threshold].copy()
            print(f"    After threshold >= {cfg.rating_threshold}: {len(df_ratings):,} interactions", flush=True)

        # Optional k-core filter (on the already-threshold-filtered rows)
        filtering_rounds = 0
        if cfg.k_core is not None:
            df_ratings, filtering_rounds = self._k_core_filter(df_ratings, cfg.k_core)
            print(f"    After {cfg.k_core}-core: {len(df_ratings):,} interactions ({filtering_rounds} rounds)", flush=True)

        # Contiguous ID mappings -- union of ratings users and trust users (see module docstring)
        all_users: Set[str] = set(df_ratings["user"].unique())
        all_users.update(df_trust["src"].unique())
        all_users.update(df_trust["dst"].unique())
        all_items: Set[str] = set(df_ratings["item"].unique())

        def _sort_key(x: str):
            try:
                return (0, int(x))
            except ValueError:
                return (1, x)

        sorted_users = sorted(all_users, key=_sort_key)
        sorted_items = sorted(all_items, key=_sort_key)
        user_map = {u: i for i, u in enumerate(sorted_users)}
        item_map = {it: i for i, it in enumerate(sorted_items)}
        num_users = len(user_map)
        num_items = len(item_map)

        df_ratings["u_idx"] = df_ratings["user"].map(user_map)
        df_ratings["i_idx"] = df_ratings["item"].map(item_map)
        df_ratings = df_ratings.dropna(subset=["u_idx", "i_idx"])
        df_ratings["u_idx"] = df_ratings["u_idx"].astype(int)
        df_ratings["i_idx"] = df_ratings["i_idx"].astype(int)

        print(f"    Contiguous: {num_users:,} users, {num_items:,} items", flush=True)

        # Split
        df_train, df_test = self._stratified_split(df_ratings, cfg.test_ratio, cfg.seed)
        print(f"    Split: Train={len(df_train):,} | Test={len(df_test):,}", flush=True)

        # Matrices
        train_csr = self._build_interaction_matrix(df_train, num_users, num_items, cfg.feedback_mode)
        train_dict = self._build_dict(df_train)
        test_dict = self._build_dict(df_test)

        social_csr = self._build_social_matrix(df_trust, user_map, num_users)
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)

        return ArenaDataset(
            num_users=num_users,
            num_items=num_items,
            train_csr=train_csr,
            test_dict=test_dict,
            train_dict=train_dict,
            social_csr=social_csr,
            mode="explicit",
            n_train_interactions=len(df_train),
            n_test_interactions=len(df_test),
            n_trust_links=social_csr.nnz,
            n_raw_interactions=n_raw,
            n_raw_users=n_raw_users,
            n_raw_items=n_raw_items,
            filtering_rounds=filtering_rounds,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def _ensure_downloaded(self) -> None:
        cfg = self.config
        os.makedirs(cfg.data_dir, exist_ok=True)

        if self._files_exist():
            print(f"  [ExplicitTrustLoader:{cfg.name}] Dataset files already present in {cfg.data_dir}", flush=True)
            return

        unique_urls = list(dict.fromkeys(cfg.ratings_urls + cfg.trust_urls))
        for url in unique_urls:
            try:
                print(f"  [ExplicitTrustLoader:{cfg.name}] Downloading from {url} ...", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
            except Exception as e:
                print(f"  [ExplicitTrustLoader:{cfg.name}] Failed to fetch {url}: {e}", flush=True)
                continue

            if zipfile.is_zipfile(io.BytesIO(data)):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    zf.extractall(cfg.data_dir)
                print(f"  [ExplicitTrustLoader:{cfg.name}] Extracted zip ({len(data)/1024/1024:.1f} MB) -> {cfg.data_dir}", flush=True)
            else:
                dest_name = self._guess_filename_for_url(url)
                dest_path = os.path.join(cfg.data_dir, dest_name)
                with open(dest_path, "wb") as f:
                    f.write(data)
                print(f"  [ExplicitTrustLoader:{cfg.name}] Saved {len(data):,} bytes -> {dest_path}", flush=True)

        if not self._files_exist():
            raise RuntimeError(
                f"Could not obtain a usable ratings/trust file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place files named one of "
                f"{cfg.ratings_filenames} (ratings) and {cfg.trust_filenames} (trust) "
                f"directly into {cfg.data_dir}."
            )

    @staticmethod
    def _guess_filename_for_url(url: str) -> str:
        """Pick a destination filename for a raw (non-zip) download from its URL."""
        basename = url.rstrip("/").split("/")[-1].split("?")[0]
        return basename if basename else "downloaded_file.txt"

    def _files_exist(self) -> bool:
        cfg = self.config
        found_ratings = False
        found_trust = False
        for root, _, files in os.walk(cfg.data_dir):
            lower_files = {f.lower() for f in files}
            if any(fn.lower() in lower_files for fn in cfg.ratings_filenames):
                found_ratings = True
            if any(fn.lower() in lower_files for fn in cfg.trust_filenames):
                found_trust = True
        return found_ratings and found_trust

    def _resolve_paths(self) -> None:
        cfg = self.config
        ratings_lower = [n.lower() for n in cfg.ratings_filenames]
        trust_lower = [n.lower() for n in cfg.trust_filenames]
        for root, _, files in os.walk(cfg.data_dir):
            for f in files:
                fl = f.lower()
                full = os.path.join(root, f)
                if fl in ratings_lower and not self._ratings_path:
                    self._ratings_path = full
                elif fl in trust_lower and not self._trust_path:
                    self._trust_path = full

        if not self._ratings_path:
            raise FileNotFoundError(f"No ratings file found under {cfg.data_dir} matching {cfg.ratings_filenames}")
        if not self._trust_path:
            raise FileNotFoundError(f"No trust file found under {cfg.data_dir} matching {cfg.trust_filenames}")

        print(f"  [ExplicitTrustLoader:{cfg.name}] Resolved: ratings={self._ratings_path}", flush=True)
        print(f"  [ExplicitTrustLoader:{cfg.name}] Resolved: trust  ={self._trust_path}", flush=True)

    # ------------------------------------------------------------------
    # Parsing (shared generic parser for both ratings and trust rows)
    # ------------------------------------------------------------------
    def _parse_rows(self, path: str, is_ratings: bool) -> pd.DataFrame:
        """
        Generic row parser shared by ratings and trust files. Branches on column
        count after delimiter-splitting:
          >=5 columns -> rating/weight at config.explicit_rating_col_index (Ciao's
                         categoryId/reviewId layout)
          >=3 columns -> rating/weight at column index 2 (generic "user item rating")
          ==2 columns -> implicit, weight defaults to 1.0
        """
        cfg = self.config
        cols = ["user", "item", "rating"] if is_ratings else ["src", "dst", "weight"]
        rows: List[Tuple[str, str, float]] = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue

                parts = self._split_line(line, cfg.delimiter)

                if len(parts) >= 5:
                    idx = cfg.explicit_rating_col_index
                    if idx < len(parts):
                        rows.append((parts[0].strip(), parts[1].strip(), float(parts[idx].strip())))
                elif len(parts) >= 3:
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
                elif len(parts) == 2:
                    rows.append((parts[0].strip(), parts[1].strip(), 1.0))

        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _split_line(line: str, delimiter: str) -> List[str]:
        if delimiter == "comma":
            return line.split(",")
        if delimiter == "space":
            return line.split()
        # "auto": try comma first (Ciao's existing heuristic), else whitespace
        if "," in line:
            return line.split(",")
        return line.split()

    # ------------------------------------------------------------------
    # k-core filtering (dataset-agnostic, extracted from AcademicDataLoader)
    # ------------------------------------------------------------------
    @staticmethod
    def _k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]:
        """Iteratively remove users/items with fewer than k interactions until convergence."""
        n_rounds = 0
        while True:
            n_before = len(df)

            user_counts = df["user"].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df["user"].isin(valid_users)]

            item_counts = df["item"].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df["item"].isin(valid_items)]

            n_rounds += 1
            if len(df) == n_before:
                break

        return df.reset_index(drop=True), n_rounds

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Per-user leave-N-out: hold out test_ratio of each user's items."""
        rng = np.random.default_rng(seed)
        train_idx: List[int] = []
        test_idx: List[int] = []

        for _, group in df.groupby("u_idx"):
            indices = group.index.tolist()
            n_test = max(1, int(len(indices) * test_ratio))
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
    def _build_interaction_matrix(
        df: pd.DataFrame, n_users: int, n_items: int, feedback_mode: str
    ) -> sp.csr_matrix:
        rows = df["u_idx"].values.astype(np.int64)
        cols = df["i_idx"].values.astype(np.int64)

        if feedback_mode == "threshold_binarize":
            vals = np.ones(len(rows), dtype=np.float32)
        else:
            vals = df["rating"].values.astype(np.float32)

        mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        if feedback_mode == "threshold_binarize":
            mat.data[:] = 1.0
        return mat

    @staticmethod
    def _build_social_matrix(
        df_trust: pd.DataFrame, user_map: Dict[str, int], n_users: int
    ) -> sp.csr_matrix:
        """Build symmetric undirected trust matrix: A = A_raw + A_raw^T, clipped binary."""
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
        for u, i in zip(df["u_idx"].values, df["i_idx"].values):
            result.setdefault(int(u), set()).add(int(i))
        return result
