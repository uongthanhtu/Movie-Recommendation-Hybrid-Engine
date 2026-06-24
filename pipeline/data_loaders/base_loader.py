"""
Base Dataset Loader -- Abstract contract and canonical output type for the
Grand Unified Benchmark Arena's dataset loading system.

ArenaDataset generalizes the dataclass already used by
pipeline/unified_arena/academic_data_loader.py (judged the cleanest of the three
existing, inconsistent loader output shapes) into the canonical contract for all
loaders produced by pipeline/data_loaders/dataset_factory.py.

BaseDatasetLoader is the abstract product in a Factory Method pattern: concrete
loaders (ExplicitTrustLoader now; ImplicitTrustLoader once Mode B lands in a later
sub-project) all expose a single load() -> ArenaDataset entry point.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import numpy as np
import scipy.sparse as sp


@dataclass
class ArenaDataset:
    """
    Canonical, dataset-agnostic output of any BaseDatasetLoader.

    Fields:
        num_users, num_items: contiguous 0-indexed counts.
        train_csr: (num_users, num_items) CSR. Binary (1.0 per interaction) if the
            loader's feedback_mode is "threshold_binarize"; real rating values if
            "explicit".
        test_dict, train_dict: {user_idx: set(item_idx)} for ranking evaluation.
        social_csr: (num_users, num_users) CSR, symmetric trust/social graph.
        mode: "explicit" (real trust data) or "implicit" (Jaccard-derived, Mode B --
            not produced by any loader yet; reserved for a future sub-project).
    """
    num_users: int
    num_items: int
    train_csr: sp.csr_matrix
    test_dict: Dict[int, Set[int]]
    train_dict: Dict[int, Set[int]]
    social_csr: sp.csr_matrix
    mode: str = "explicit"

    n_train_interactions: int = 0
    n_test_interactions: int = 0
    n_trust_links: int = 0
    n_raw_interactions: int = 0
    n_raw_users: int = 0
    n_raw_items: int = 0
    filtering_rounds: int = 0

    _sym_adj_mat_cache: Optional[sp.csr_matrix] = field(default=None, repr=False, compare=False)

    def get_sym_adj_mat(self) -> sp.csr_matrix:
        """
        Bipartite symmetric-normalized adjacency for LightGCN-style engines,
        built from train_csr lazily on first call and cached on this instance.
        Consumers that build their own normalization internally never call this
        and pay nothing for it.

        Returns:
            sp.csr_matrix of shape (num_users + num_items, num_users + num_items).
        """
        if self._sym_adj_mat_cache is not None:
            return self._sym_adj_mat_cache

        R_binary = self.train_csr.copy()
        R_binary.data = np.ones_like(R_binary.data, dtype=np.float32)

        adj_mat = sp.bmat([[None, R_binary], [R_binary.T, None]], format="csr")

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        sym_adj_mat = D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt).tocsr()
        self._sym_adj_mat_cache = sym_adj_mat
        return sym_adj_mat


class BaseDatasetLoader(abc.ABC):
    """Abstract product for the dataset factory. All loaders expose one entry point."""

    @abc.abstractmethod
    def load(self) -> ArenaDataset:
        """Full pipeline: download -> parse -> (optional filter) -> split -> matrices."""
