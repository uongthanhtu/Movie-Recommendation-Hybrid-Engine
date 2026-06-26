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

The dataset-agnostic download/parse/filter/split/matrix-construction logic this class
needs is shared with pipeline/data_loaders/implicit_trust_loader.py via
pipeline/data_loaders/loader_utils.py -- this file owns only what's specific to having
a real, downloadable trust file (the union-of-users universe and the trust-matrix
construction itself).
"""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
import scipy.sparse as sp

from pipeline.data_loaders import loader_utils as lu
from pipeline.data_loaders.base_loader import ArenaDataset, BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DatasetConfig


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
        df_ratings = lu.parse_rows(self._ratings_path, cfg.delimiter, cfg.explicit_rating_col_index, ("user", "item", "rating"))
        df_trust = lu.parse_rows(self._trust_path, cfg.delimiter, cfg.explicit_rating_col_index, ("src", "dst", "weight"))

        if cfg.filter_negative_trust:
            n_trust_before = len(df_trust)
            df_trust = df_trust[df_trust["weight"] > 0].copy()
            print(f"    Filtered distrust: {n_trust_before:,} -> {len(df_trust):,} trust edges", flush=True)

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
            df_ratings, filtering_rounds = lu.k_core_filter(df_ratings, cfg.k_core)
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
        df_train, df_test = lu.stratified_split(df_ratings, cfg.test_ratio, cfg.seed)
        print(f"    Split: Train={len(df_train):,} | Test={len(df_test):,}", flush=True)

        # Matrices
        train_csr = lu.build_interaction_matrix(df_train, num_users, num_items, binarize=(cfg.feedback_mode == "threshold_binarize"))
        train_dict = lu.build_dict(df_train)
        test_dict = lu.build_dict(df_test)

        if cfg.denoise_social_graph:
            df_trust = lu.denoise_social_edges(df_trust, user_map, train_csr, cfg.denoise_jaccard_threshold)

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
        if lu.files_exist(cfg.data_dir, cfg.ratings_filenames) and lu.files_exist(cfg.data_dir, cfg.trust_filenames):
            print(f"  [ExplicitTrustLoader:{cfg.name}] Dataset files already present in {cfg.data_dir}", flush=True)
            return

        unique_urls = list(dict.fromkeys(cfg.ratings_urls + cfg.trust_urls))
        lu.download_with_fallback(unique_urls, cfg.data_dir, cfg.name, "ExplicitTrustLoader")

        if not (lu.files_exist(cfg.data_dir, cfg.ratings_filenames) and lu.files_exist(cfg.data_dir, cfg.trust_filenames)):
            generic_message = (
                f"Could not obtain a usable ratings/trust file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place files named one of "
                f"{cfg.ratings_filenames} (ratings) and {cfg.trust_filenames} (trust) "
                f"directly into {cfg.data_dir}."
            )
            if cfg.manual_download_instructions:
                raise lu.ManualDownloadRequiredError(f"{generic_message}\n\n{cfg.manual_download_instructions}")
            raise lu.ManualDownloadRequiredError(generic_message)

    def _resolve_paths(self) -> None:
        cfg = self.config
        self._ratings_path = lu.resolve_path(cfg.data_dir, cfg.ratings_filenames)
        self._trust_path = lu.resolve_path(cfg.data_dir, cfg.trust_filenames)
        print(f"  [ExplicitTrustLoader:{cfg.name}] Resolved: ratings={self._ratings_path}", flush=True)
        print(f"  [ExplicitTrustLoader:{cfg.name}] Resolved: trust  ={self._trust_path}", flush=True)

    # ------------------------------------------------------------------
    # Sparse Matrix Construction (explicit-trust-specific; not shared with ImplicitTrustLoader)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_social_matrix(
        df_trust, user_map: Dict[str, int], n_users: int
    ) -> sp.csr_matrix:
        """Build symmetric undirected trust matrix: A = A_raw + A_raw^T, clipped binary.

        Self-loop rows (src == dst) are dropped before construction -- a self-trust
        edge would otherwise survive symmetrization as a nonzero diagonal entry,
        violating the zero-diagonal invariant every consumer of social_csr assumes.
        """
        df = df_trust.copy()
        df["s_idx"] = df["src"].map(user_map)
        df["d_idx"] = df["dst"].map(user_map)
        df = df.dropna(subset=["s_idx", "d_idx"])
        df["s_idx"] = df["s_idx"].astype(int)
        df["d_idx"] = df["d_idx"].astype(int)
        df = df[df["s_idx"] != df["d_idx"]]

        if len(df) == 0:
            return sp.csr_matrix((n_users, n_users), dtype=np.float32)

        rows = df["s_idx"].values
        cols = df["d_idx"].values
        vals = np.ones(len(rows), dtype=np.float32)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_users))
        A_sym = (A + A.T).tocsr()
        A_sym.data = np.minimum(A_sym.data, 1.0)
        return A_sym
