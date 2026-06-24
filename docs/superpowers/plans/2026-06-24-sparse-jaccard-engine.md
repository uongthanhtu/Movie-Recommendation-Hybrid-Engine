# Sparse Jaccard Trust Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, dataset-agnostic, OOM-safe module that computes a User-User Jaccard-similarity trust graph from any sparse User-Item interaction matrix, verified correct at small scale and verified not to OOM at a synthetic 120,000-user scale.

**Architecture:** A single pure function, `compute_sparse_jaccard_trust`, in a new `pipeline/utils/` package. It processes users in row-blocks ("chunks") to bound peak memory, vectorizes the Jaccard computation per chunk (no Python-level per-pair loop), truncates to the top-K highest-similarity neighbors per user before moving to the next chunk (bounding output size), and symmetrizes the result via `A.maximum(A.T)` (correct for continuous weights, unlike the `+`-then-clip pattern used elsewhere in this codebase for binary weights).

**Tech Stack:** Python, `scipy.sparse`, `numpy`, `pandas` (already in `requirements.txt`), stdlib `gc`.

## Global Constraints

- New code lives only in `pipeline/utils/sparse_jaccard.py` (+ `pipeline/utils/__init__.py`, since `pipeline/utils/` does not yet exist).
- Do not modify `pipeline/engines/unified_data_loader.py` or any other existing file — this module is new and standalone; the existing (slower, less memory-safe) Jaccard code stays frozen, since production code depends on it.
- No new third-party dependencies — only `numpy`, `pandas`, `scipy.sparse` (already in `requirements.txt`) and stdlib `gc`/`typing`.
- Symmetrization MUST use `A.maximum(A.T)`, never `A + A.T`. Jaccard weights are continuous in `[0,1]`; if both directions of a pair independently survive top-K pruning, addition double-counts an already-symmetric value (e.g. `0.4 + 0.4 = 0.8` misrepresenting a true similarity of `0.4`). This is a deliberate departure from the `+`-then-clip pattern in `pipeline/filmtrust_arena/filmtrust_loader.py`/`pipeline/academic_sandbox/yelp_data_loader.py`, which is only correct for their binary (0/1) trust weights.
- This codebase has no pytest/unit-test framework (no `tests/` directory, no pytest config). Verification in this plan uses direct script execution with documented expected output, matching the convention used for every other module in this codebase.
- This sub-project does not touch any dataset-specific loader, downloader, or CLI — that is out of scope (covered by separate future sub-projects).

---

### Task 1: Core implementation + small-scale correctness verification

**Files:**
- Create: `pipeline/utils/__init__.py`
- Create: `pipeline/utils/sparse_jaccard.py`

**Interfaces:**
- Produces: `compute_sparse_jaccard_trust(interaction_matrix: scipy.sparse.csr_matrix, threshold: float = 0.3, top_k: Optional[int] = 50, chunk_size: int = 2000, dtype: np.dtype = np.float32) -> scipy.sparse.csr_matrix`. Returns a `(num_users x num_users)` CSR matrix, symmetric, Jaccard-weighted, zero diagonal.

- [ ] **Step 1: Create the package `__init__.py`**

```python
# Shared, dataset-agnostic utilities for the benchmarking pipeline.
```

- [ ] **Step 2: Write `pipeline/utils/sparse_jaccard.py`**

```python
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
```

Note: top-K selection uses `np.lexsort` + a group-rank trick (not a pandas
`groupby().apply(nlargest)`, which the original design spec sketched) — this is faster
and avoids pandas deprecation-warning noise from `apply` on `DataFrameGroupBy`. No
`pandas` import is needed in the final module.

- [ ] **Step 3: Write and run the small-scale correctness verification script**

Run:
```bash
python -c "
import numpy as np
import scipy.sparse as sp
from pipeline.utils.sparse_jaccard import compute_sparse_jaccard_trust

# --- Check 1: hand-checkable correctness ---
# User0: items {0,1}; User1: items {0,1,2}; User2: items {2,3}; User3: items {0}; User4: items {} (isolated)
rows = [0,0, 1,1,1, 2,2, 3]
cols = [0,1, 0,1,2, 2,3, 0]
data = [1]*8
interaction = sp.csr_matrix((data, (rows, cols)), shape=(5, 4))

result = compute_sparse_jaccard_trust(interaction, threshold=0.3, top_k=10, chunk_size=5).toarray()

# Jaccard(0,1) = |{0,1} & {0,1,2}| / |{0,1} | {0,1,2}| = 2/3
# Jaccard(0,3) = |{0,1} & {0}| / |{0,1} | {0}| = 1/2
# Jaccard(1,3) = |{0,1,2} & {0}| / |{0,1,2} | {0}| = 1/3
# Jaccard(1,2) = 1/4 -> below threshold 0.3, excluded
# Jaccard(0,2) = 0; Jaccard(2,3) = 0
expected_pairs = {(0, 1): 2/3, (0, 3): 0.5, (1, 3): 1/3}
ok = True
for (u, v), val in expected_pairs.items():
    if not (np.isclose(result[u, v], val, atol=1e-5) and np.isclose(result[v, u], val, atol=1e-5)):
        ok = False
        print(f'MISMATCH at ({u},{v}): got {result[u,v]}, {result[v,u]}, expected {val}')
mask = np.ones((5, 5), dtype=bool)
for (u, v) in expected_pairs:
    mask[u, v] = False
    mask[v, u] = False
np.fill_diagonal(mask, False)
if not np.allclose(result[mask], 0.0):
    ok = False
    print('MISMATCH: unexpected nonzero entries outside expected pairs')
print('Check 1 (hand-checkable correctness):', 'PASS' if ok else 'FAIL')

# --- Check 2: chunking equivalence (single chunk vs many small chunks must match) ---
rng = np.random.default_rng(42)
n_users, n_items, density = 200, 80, 0.05
nnz = int(n_users * n_items * density)
u_idx = rng.integers(0, n_users, size=nnz)
i_idx = rng.integers(0, n_items, size=nnz)
vals = np.ones(nnz)
big = sp.csr_matrix((vals, (u_idx, i_idx)), shape=(n_users, n_items))

result_single_chunk = compute_sparse_jaccard_trust(big, threshold=0.2, top_k=10, chunk_size=200)
result_multi_chunk = compute_sparse_jaccard_trust(big, threshold=0.2, top_k=10, chunk_size=20)

equal = np.allclose(result_single_chunk.toarray(), result_multi_chunk.toarray(), atol=1e-6)
print('Check 2 (chunking equivalence):', 'PASS' if equal else 'FAIL')
"
```

Expected output:
```
Check 1 (hand-checkable correctness): PASS
Check 2 (chunking equivalence): PASS
```
If either prints `FAIL` or a `MISMATCH` line, the implementation has a bug — do not proceed until both print `PASS`.

- [ ] **Step 4: Commit**

```bash
git add pipeline/utils/__init__.py pipeline/utils/sparse_jaccard.py
git commit -m "feat(utils): add OOM-safe chunked sparse Jaccard trust engine

Standalone, dataset-agnostic User-User Jaccard similarity computation
with row-block chunking and top-K truncation, bounding both
intermediate and output memory independent of co-occurrence density."
```

---

### Task 2: Large-scale OOM/timing smoke test

**Files:**
- No new files. This task only runs a verification command (no committed code change)
  unless the command reveals a bug in Task 1's implementation, in which case fix
  `pipeline/utils/sparse_jaccard.py` and commit the fix.

**Interfaces:**
- Consumes: `compute_sparse_jaccard_trust` from Task 1, signature unchanged.

This is the task that actually demonstrates the OOM-safety claim from the spec, using a
synthetic dataset shaped like a real one: most items are rarely interacted with, but a
small subset of "popular" items are shared by a large fraction of users (power-law/Zipf
popularity) — this is the specific scenario that makes a naive `R @ R.T` blow up, since
popular items create large near-complete cliques of co-interacting users. A uniform-random
sparse matrix would NOT stress-test this risk; the popularity skew below is deliberate.

- [ ] **Step 1: Run the large-scale smoke test**

This generates a synthetic 120,000-user x 50,000-item interaction matrix with skewed
(Zipf-like) item popularity, runs `compute_sparse_jaccard_trust` on it, and checks it
completes without `MemoryError`, the output edge count respects the `2 x top_k x num_users`
bound, and the result is symmetric. Allow up to 10 minutes wall-clock for this command —
it is doing real work at production scale, not a unit test.

Run (timeout: 600000ms / 10 minutes):
```bash
python -c "
import time
import numpy as np
import scipy.sparse as sp
from pipeline.utils.sparse_jaccard import compute_sparse_jaccard_trust

rng = np.random.default_rng(7)
n_users, n_items, avg_interactions = 120_000, 50_000, 80

# Zipf-like item popularity: a small fraction of items dominate interaction volume,
# deliberately stress-testing the 'popular items create dense cliques' OOM risk.
item_weights = 1.0 / (np.arange(1, n_items + 1) ** 1.2)
item_probs = item_weights / item_weights.sum()

counts = np.clip(rng.poisson(avg_interactions, size=n_users), 1, None)

rows_list, cols_list = [], []
for start in range(0, n_users, 10_000):
    end = min(start + 10_000, n_users)
    block_counts = counts[start:end]
    total = int(block_counts.sum())
    block_items = rng.choice(n_items, size=total, p=item_probs)
    block_users = np.repeat(np.arange(start, end), block_counts)
    rows_list.append(block_users)
    cols_list.append(block_items)

rows_arr = np.concatenate(rows_list)
cols_arr = np.concatenate(cols_list)
vals_arr = np.ones(len(rows_arr), dtype=np.float32)

interaction = sp.csr_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_users, n_items))
print(f'Synthetic matrix: {n_users:,} users x {n_items:,} items, nnz={interaction.nnz:,}')

t0 = time.time()
trust = compute_sparse_jaccard_trust(interaction, threshold=0.3, top_k=50, chunk_size=2000)
elapsed = time.time() - t0

print(f'Completed in {elapsed:.1f}s')
print(f'Trust matrix: shape={trust.shape}, nnz={trust.nnz:,}')

max_possible_edges = 2 * 50 * n_users
assert trust.nnz <= max_possible_edges, f'nnz {trust.nnz} exceeds bound {max_possible_edges}'
assert (trust != trust.T).nnz == 0, 'trust matrix is not symmetric'
print('Check 3 (large-scale OOM/timing smoke test): PASS')
"
```

Expected output: the script runs to completion (no `MemoryError`, no `AssertionError`,
no traceback), printing the synthetic matrix's stats, the elapsed time, the trust
matrix's shape/nnz, and a final `Check 3 (large-scale OOM/timing smoke test): PASS`
line. **Record the actual elapsed time** — if it is unreasonably long (e.g. > 8 minutes),
note this as a concern even if it technically passes, since `chunk_size`/`top_k` may need
retuning before this function is used inside a real arena runner in a later sub-project.

- [ ] **Step 2: If the smoke test reveals a bug, fix and re-verify**

If `Check 3` does not print `PASS` (an exception, an `AssertionError`, or a `MemoryError`),
this is a real defect in Task 1's implementation, not an acceptable outcome — diagnose
and fix `pipeline/utils/sparse_jaccard.py`, then re-run Step 1 in full (the smoke test) AND
Task 1's Step 3 script (to confirm the fix didn't break small-scale correctness) before
proceeding.

- [ ] **Step 3: Commit**

If Step 2 required a code fix:
```bash
git add pipeline/utils/sparse_jaccard.py
git commit -m "fix(utils): resolve large-scale smoke test failure in sparse Jaccard engine"
```

If no fix was needed (Step 1 passed cleanly on the first run), there is nothing new to
commit for this task — record the elapsed time and nnz figures in your report instead.
