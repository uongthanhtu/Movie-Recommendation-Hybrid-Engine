"""
Sparse Jaccard Trust Engine -- OOM-safe, chunked, top-K User-User Jaccard similarity.

Computes a Jaccard-similarity-based trust graph from any sparse User-Item interaction
matrix. Unlike a naive full R @ R.T computation (which can produce a near-fully-dense
intermediate matrix even from sparse input, when items are widely shared across users),
this module bounds both intermediate and output memory by:
  1. Processing users in row-blocks ("chunks"), never materializing the full
     (num_users x num_users) intersection matrix at once.
  2. Truncating each user's neighbor list to the top-K highest-Jaccard matches before
     moving to the next chunk -- this is what actually bounds output size, independent
     of how dense the raw co-occurrence pattern is.

This module is dataset-agnostic: it has no knowledge of any specific dataset
(MovieLens, Jester, etc.) and operates purely on a scipy.sparse interaction matrix.

This fixes the two structural problems present in
pipeline/engines/unified_data_loader.py::build_implicit_trust_matrix(), which computes
the full intersection matrix in one shot (no chunking) and loops over nonzero entries
in pure Python (no vectorization). That function is intentionally left unmodified --
production code depends on it -- this is a new, separate utility.
"""
import gc
from typing import Optional

import numpy as np
import scipy.sparse as sp


def compute_sparse_jaccard_trust(
    interaction_matrix: sp.csr_matrix,
    threshold: float = 0.3,
    top_k: Optional[int] = 50,
    chunk_size: int = 2000,
    dtype: np.dtype = np.float32,
) -> sp.csr_matrix:
    """
    Compute a symmetric, Jaccard-weighted User-User trust graph from a sparse
    User-Item interaction matrix, processing users in memory-bounded row-blocks.

    Args:
        interaction_matrix: (num_users x num_items) CSR. Any nonzero entry counts as
            an interaction; the values themselves (e.g. ratings) are ignored.
        threshold: minimum Jaccard similarity required to keep an edge.
        top_k: max neighbors kept per user before symmetrization. Pass None to disable
            truncation -- NOT recommended above a few thousand users, since this removes
            the primary memory bound.
        chunk_size: number of user-rows processed per chunk. Lower values reduce peak
            memory at the cost of more chunk-loop overhead.
        dtype: dtype of the returned matrix's `.data` array.

    Returns:
        (num_users x num_users) CSR matrix, symmetric, Jaccard-weighted, zero diagonal.
    """
    num_users = interaction_matrix.shape[0]

    binary = interaction_matrix.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    binary = binary.tocsr()

    degrees = np.asarray(binary.sum(axis=1)).flatten().astype(np.float32)
    R_T = binary.T.tocsr()

    chunk_size = min(chunk_size, num_users)

    rows_acc = []
    cols_acc = []
    vals_acc = []

    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)

        intersection_block = binary[start:end] @ R_T
        block_coo = intersection_block.tocoo()

        global_rows = block_coo.row + start
        cols = block_coo.col
        inter = block_coo.data.astype(np.float32)

        not_self = global_rows != cols
        global_rows = global_rows[not_self]
        cols = cols[not_self]
        inter = inter[not_self]

        union = degrees[global_rows] + degrees[cols] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, 0.0).astype(np.float32)

        keep = jaccard > threshold
        global_rows = global_rows[keep]
        cols = cols[keep]
        jaccard = jaccard[keep]

        if top_k is not None and len(global_rows) > 0:
            order = np.lexsort((-jaccard, global_rows))
            sorted_rows = global_rows[order]
            sorted_cols = cols[order]
            sorted_jaccard = jaccard[order]

            _, group_start_idx = np.unique(sorted_rows, return_index=True)
            group_sizes = np.diff(np.append(group_start_idx, len(sorted_rows)))
            rank = np.arange(len(sorted_rows)) - np.repeat(group_start_idx, group_sizes)

            keep_topk = rank < top_k
            global_rows = sorted_rows[keep_topk]
            cols = sorted_cols[keep_topk]
            jaccard = sorted_jaccard[keep_topk]

        rows_acc.append(global_rows.copy())
        cols_acc.append(cols.copy())
        vals_acc.append(jaccard.copy())

        del intersection_block, block_coo, inter, union, jaccard, keep
        gc.collect()

    if rows_acc:
        final_rows = np.concatenate(rows_acc)
        final_cols = np.concatenate(cols_acc)
        final_vals = np.concatenate(vals_acc).astype(dtype)
    else:
        final_rows = np.array([], dtype=np.int64)
        final_cols = np.array([], dtype=np.int64)
        final_vals = np.array([], dtype=dtype)

    A = sp.coo_matrix(
        (final_vals, (final_rows, final_cols)), shape=(num_users, num_users)
    ).tocsr()

    A_sym = A.maximum(A.T).tocsr()
    A_sym.data = A_sym.data.astype(dtype)

    gc.collect()
    return A_sym
