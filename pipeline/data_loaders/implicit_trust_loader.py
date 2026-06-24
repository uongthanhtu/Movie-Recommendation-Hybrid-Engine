"""
Implicit Trust Loader -- Mode B / ablation-study loader for datasets with no real
social network (MovieLens-100K for now; ML-1M/10M and Jester deferred until their
real source URLs/formats are verified).

Trust is SYNTHETIC: a Jaccard-similarity graph derived from rating co-occurrence via
pipeline/utils/sparse_jaccard.py::compute_sparse_jaccard_trust (the OOM-safe engine
built in sub-project 1), computed from train_csr ONLY -- never the full pre-split
interactions -- to avoid leaking test-set co-occurrence into the trust side-channel
that TrustSVD-style models consume.

THIS IS NOT A REAL SOCIAL BENCHMARK. Results produced via this loader must always be
presented as an explicitly-labeled ablation study (see ArenaDataset.mode == "implicit"
and the console banner this module prints), never as evidence about real social
recommendation. This mirrors the binding decision recorded in
docs/superpowers/specs/2026-06-24-sparse-jaccard-design.md.

Shares its download/parse/filter/split/matrix-construction logic with
pipeline/data_loaders/explicit_trust_loader.py via
pipeline/data_loaders/loader_utils.py -- this file owns only what's specific to having
no real trust file (calling compute_sparse_jaccard_trust instead).
"""
from __future__ import annotations

import gc

from pipeline.data_loaders import loader_utils as lu
from pipeline.data_loaders.base_loader import ArenaDataset, BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import ImplicitDatasetConfig
from pipeline.utils.sparse_jaccard import compute_sparse_jaccard_trust


class ImplicitTrustLoader(BaseDatasetLoader):
    """
    Generic loader for datasets with no real trust data, configured entirely via an
    ImplicitDatasetConfig. Trust is synthesized via Jaccard similarity on train-only
    rating co-occurrence -- an ablation study, not a real social benchmark.

    Usage:
        loader = ImplicitTrustLoader(ML_100K_CONFIG)
        dataset = loader.load()
    """

    def __init__(self, config: ImplicitDatasetConfig):
        self.config = config
        self._ratings_path = ""

    def load(self) -> ArenaDataset:
        """Full pipeline: download -> parse -> (optional k-core) -> split -> matrices -> synthesize trust."""
        cfg = self.config

        print("=" * 80, flush=True)
        print(f"[ImplicitTrustLoader:{cfg.name}] ABLATION STUDY ONLY -- Mode B", flush=True)
        print("  Trust graph is SYNTHETIC (Jaccard co-occurrence similarity), NOT real social data.", flush=True)
        print("  Do not present results using this loader as a genuine social-aware benchmark.", flush=True)
        print("=" * 80, flush=True)

        self._ensure_downloaded()
        self._ratings_path = lu.resolve_path(cfg.data_dir, cfg.ratings_filenames)
        print(f"  [ImplicitTrustLoader:{cfg.name}] Resolved: ratings={self._ratings_path}", flush=True)

        print(f"  [ImplicitTrustLoader:{cfg.name}] Parsing raw file ...", flush=True)
        df_ratings = lu.parse_rows(self._ratings_path, cfg.delimiter, cfg.rating_col_index, ("user", "item", "rating"))

        n_raw = len(df_ratings)
        n_raw_users = df_ratings["user"].nunique()
        n_raw_items = df_ratings["item"].nunique()
        print(f"    Raw: {n_raw:,} interactions, {n_raw_users:,} users, {n_raw_items:,} items", flush=True)

        filtering_rounds = 0
        if cfg.k_core is not None:
            df_ratings, filtering_rounds = lu.k_core_filter(df_ratings, cfg.k_core)
            print(f"    After {cfg.k_core}-core: {len(df_ratings):,} interactions ({filtering_rounds} rounds)", flush=True)

        # Contiguous ID mappings -- ratings-file users only (no trust file to union against in Mode B)
        def _sort_key(x: str):
            try:
                return (0, int(x))
            except ValueError:
                return (1, x)

        sorted_users = sorted(df_ratings["user"].unique(), key=_sort_key)
        sorted_items = sorted(df_ratings["item"].unique(), key=_sort_key)
        user_map = {u: i for i, u in enumerate(sorted_users)}
        item_map = {it: i for i, it in enumerate(sorted_items)}
        num_users = len(user_map)
        num_items = len(item_map)

        df_ratings["u_idx"] = df_ratings["user"].map(user_map)
        df_ratings["i_idx"] = df_ratings["item"].map(item_map)
        df_ratings["u_idx"] = df_ratings["u_idx"].astype(int)
        df_ratings["i_idx"] = df_ratings["i_idx"].astype(int)

        print(f"    Contiguous: {num_users:,} users, {num_items:,} items", flush=True)

        df_train, df_test = lu.stratified_split(df_ratings, cfg.test_ratio, cfg.seed)
        print(f"    Split: Train={len(df_train):,} | Test={len(df_test):,}", flush=True)

        train_csr = lu.build_interaction_matrix(df_train, num_users, num_items, binarize=False)
        train_dict = lu.build_dict(df_train)
        test_dict = lu.build_dict(df_test)
        print(f"    train_csr: {train_csr.nnz:,} explicit-rating entries", flush=True)

        print(f"  [ImplicitTrustLoader:{cfg.name}] Synthesizing trust via Jaccard "
              f"(train-only, threshold={cfg.jaccard_threshold}, top_k={cfg.jaccard_top_k}) ...", flush=True)
        social_csr = compute_sparse_jaccard_trust(
            train_csr,
            threshold=cfg.jaccard_threshold,
            top_k=cfg.jaccard_top_k,
            chunk_size=cfg.jaccard_chunk_size,
        )
        gc.collect()
        print(f"    Synthetic social_csr (Mode B): {social_csr.nnz:,} edges (symmetric)", flush=True)

        return ArenaDataset(
            num_users=num_users,
            num_items=num_items,
            train_csr=train_csr,
            test_dict=test_dict,
            train_dict=train_dict,
            social_csr=social_csr,
            mode="implicit",
            n_train_interactions=len(df_train),
            n_test_interactions=len(df_test),
            n_trust_links=social_csr.nnz,
            n_raw_interactions=n_raw,
            n_raw_users=n_raw_users,
            n_raw_items=n_raw_items,
            filtering_rounds=filtering_rounds,
        )

    def _ensure_downloaded(self) -> None:
        cfg = self.config
        if lu.files_exist(cfg.data_dir, cfg.ratings_filenames):
            print(f"  [ImplicitTrustLoader:{cfg.name}] Dataset files already present in {cfg.data_dir}", flush=True)
            return

        lu.download_with_fallback(cfg.ratings_urls, cfg.data_dir, cfg.name, "ImplicitTrustLoader")

        if not lu.files_exist(cfg.data_dir, cfg.ratings_filenames):
            raise RuntimeError(
                f"Could not obtain a usable ratings file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place a file named one of "
                f"{cfg.ratings_filenames} directly into {cfg.data_dir}."
            )
