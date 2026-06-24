# Sparse Jaccard Trust Engine — Design Spec

Date: 2026-06-24

## Context: this is sub-project 1 of 5

This is the first slice of a larger "Grand Unified Benchmark Arena" initiative. The
full initiative was decomposed (with user approval) into:

1. **`pipeline/utils/sparse_jaccard.py`** — OOM-safe Jaccard trust engine (THIS SPEC)
2. `dataset_factory.py` + consolidation of the existing Ciao/Yelp/FilmTrust arenas
   (`pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/`)
   and the legacy dead code (`pipeline/engines/academic_benchmark_arena.py` +
   `academic_data_loader.py`) into one Factory-pattern loader hierarchy
3. `ImplicitTrustLoader` (Mode B) — Jaccard-based trust for MovieLens/Jester, built on (1)
4. New explicit-trust datasets (Douban, Epinions variants, Flixster) — each needs its
   real source/format verified before being spec'd (no guessed URLs)
5. `grand_arena_runner.py` — config-driven orchestrator tying 1-4 together

Sub-projects 2-5 are NOT covered by this spec and will each get their own
brainstorm → spec → plan cycle.

## Decisions already made (binding on later sub-projects too)

- **Mode B framing:** Jaccard-derived "trust" must always be presented as an explicit,
  clearly-labeled ablation/limitation study — never as a real social benchmark. This
  reverses nothing from the prior FilmTrust Social Arena work (`docs/superpowers/specs/2026-06-24-filmtrust-social-arena-design.md`):
  real trust data (Mode A) remains the only thing presented as a legitimate Social-Aware
  benchmark. Mode B is an explicitly-caveated "what if trust were fake" comparison.
- **Consolidation scope:** the existing Ciao (`unified_arena`), Yelp (`academic_sandbox`),
  and FilmTrust (`filmtrust_arena`) loaders will eventually be migrated into the new
  factory, and the legacy `academic_benchmark_arena.py`/`academic_data_loader.py` dead
  code deleted. That migration happens in sub-project 2, not here.

## Problem (this sub-project)

`pipeline/engines/unified_data_loader.py::build_implicit_trust_matrix()` already computes
a Jaccard-based "trust" matrix, but it does so by:
1. Computing the full `R @ R.T` co-occurrence matrix in one shot (no chunking) — at
   71K-120K users (ML-10M, Epinions scale) this intermediate can become extremely dense
   even though the input `R` is sparse, because popular items create large cliques of
   co-interacting users. This risks `MemoryError`.
2. Looping over every nonzero entry in pure Python to compute and threshold Jaccard —
   slow at scale (potentially hundreds of millions of iterations), independent of the
   memory problem above.

Neither problem is fixable by "just use scipy.sparse" alone — `unified_data_loader.py`
already uses scipy.sparse and still has both problems. The fix requires bounding the
*output* size deterministically (top-K truncation) and bounding the *intermediate* size
(row-block chunking), not just choosing a sparse container.

## Goal

A standalone, dataset-agnostic module that computes a User-User Jaccard-similarity trust
graph from any sparse User-Item interaction matrix, with memory usage that does not scale
with the density of item popularity — verified at a synthetic 120,000-user scale.

## Interface

```python
def compute_sparse_jaccard_trust(
    interaction_matrix: scipy.sparse.csr_matrix,
    threshold: float = 0.3,
    top_k: int = 50,
    chunk_size: int = 2000,
    dtype: np.dtype = np.float32,
) -> scipy.sparse.csr_matrix:
    """
    Args:
        interaction_matrix: (num_users x num_items) CSR. Any nonzero entry counts as
            an interaction; values themselves (ratings) are ignored — this is a
            structural (binary) Jaccard, matching unified_data_loader.py's existing
            semantics.
        threshold: minimum Jaccard similarity to keep an edge.
        top_k: max neighbors kept per user BEFORE symmetrization (the final symmetrized
            graph may have more than top_k edges for a given user, if other users chose
            it as one of their top_k). Pass None to disable truncation (NOT recommended
            above a few thousand users -- removes the primary memory bound).
        chunk_size: number of user-rows processed per chunk. Lower = less peak memory,
            slower (more Python-level loop overhead across chunks). Higher = faster,
            more peak memory. 2000 is a reasonable default for ~100K-user datasets.
        dtype: dtype of the returned matrix's data array.

    Returns:
        (num_users x num_users) CSR, symmetric, Jaccard-weighted, zero diagonal.
    """
```

Pure and dataset-agnostic: no knowledge of MovieLens, Jester, or any other dataset.
This makes it directly reusable by the future `ImplicitTrustLoader` (sub-project 3)
without modification.

## Algorithm

1. **Binarize** the input (`data[:] = 1`) — defensive, in case the caller passes raw
   ratings rather than a 0/1 matrix.
2. **Precompute once, reuse across all chunks:** per-user degree (`binary.sum(axis=1)`,
   as `float32`) and `R_T = binary.T.tocsr()`.
3. **For each row-block** of `chunk_size` users:
   a. `intersection_block = binary[start:end] @ R_T` — sparse × sparse, scoped to this
      chunk only. This is the step that must NOT be done for the full matrix at once:
      scoping it to a chunk bounds the intermediate's possible size to
      `chunk_size × num_users` worth of *candidate* nonzeros, not `num_users²`.
   b. Convert to COO; vectorized (not Python-looped) Jaccard:
      `jaccard = intersection / (deg_u[row] + deg_v[col] - intersection)`,
      with `np.errstate(divide="ignore")` + zero-fill for any zero-union pairs.
   c. Mask self-pairs (`global_row == col`) out.
   d. Filter to `jaccard > threshold`.
   e. **Top-K per row:** group remaining (row, col, jaccard) triplets by row, keep the
      `top_k` highest-jaccard entries per row (via `pandas.DataFrame.groupby(...).apply`
      + `nlargest`, consistent with this codebase's existing pandas-heavy style in
      `yelp_data_loader.py`/`filmtrust_loader.py`).
   f. Append surviving triplets to accumulator lists (lists of numpy arrays, NOT
      repeated `np.concatenate` per chunk — that would be O(n²) copying). `del` the
      chunk's large intermediates and call `gc.collect()` before moving to the next
      chunk, per the stated requirement.
4. **Concatenate once** at the end into final `(rows, cols, values)` arrays, build a
   COO matrix, convert to CSR.
5. **Symmetrize via `A.maximum(A.T)`, not `A + A.T`.** This is a deliberate departure
   from the `+`-then-clip pattern used in `filmtrust_loader.py`/`yelp_data_loader.py`:
   those work because their trust weights are binary (0/1), where doubling and clipping
   back to 1 is harmless. Jaccard weights are continuous in `[0,1]`; if both directions
   of a pair independently survive top-K pruning, naive addition would double-count an
   already-symmetric value (e.g. `0.4 + 0.4 = 0.8`, misrepresenting a true similarity of
   `0.4`). `.maximum()` is correct for continuous weights; `+`-then-clip is not.
6. Cast the result's `.data` to the requested `dtype` and return.

## Memory bound

Final edge count is capped at roughly `2 × top_k × num_users` (the factor of 2 from
symmetrization), independent of how dense the raw co-occurrence pattern is. This is
the actual fix — not "uses scipy.sparse" (the existing buggy code already does that),
but "output size is bounded by a constant times the user count, by construction."

## Edge cases

- Users with zero interactions: their row in `binary` is all-zero, so their row of
  `intersection_block` is all-zero too — no candidate pairs, no special-casing needed.
- `chunk_size > num_users`: clamp to `num_users` (single chunk).
- `top_k=None`: skip step 3e entirely (threshold-only filtering, matching the legacy
  behavior) — document in the docstring that this removes the memory bound and should
  only be used on small datasets.

## Out of scope

- Replacing `unified_data_loader.py::build_implicit_trust_matrix()`'s call site — that
  module is intentionally frozen (production code depends on it; see the FilmTrust
  Social Arena spec). This sub-project only adds the new, isolated utility.
- Any dataset-specific loading, downloading, or CLI surface — that's sub-projects 2-5.
- Approximate methods (MinHash/LSH) — noted as a future option if real datasets ever
  exceed ~500K users, not needed at the stated 71K-120K user scale.

## Verification plan

No pytest/unit-test framework exists in this repo (confirmed during the prior FilmTrust
work). Verification follows the same convention as every other module in this codebase:
a runnable script with documented expected output, executed for real.

1. **Hand-checkable correctness:** a ~5-user, ~4-item synthetic matrix with manually
   computed expected Jaccard values; assert the function's output matches exactly.
2. **Chunking-equivalence check:** a ~200-user synthetic matrix, run once with
   `chunk_size=200` (single chunk) and once with `chunk_size=20` (10 chunks); assert
   both produce identical results — this isolates whether the chunking logic itself
   (not just the math) is correct.
3. **Large-scale OOM/timing smoke test:** a synthetic `scipy.sparse.random` matrix at
   ~120,000 users × 50,000 items, realistic density (~0.1-0.3%, comparable to real
   rating datasets), run through the function and confirm it completes without
   `MemoryError` and in a reasonable wall-clock time. This is the actual evidence behind
   the OOM claim, since no real dataset at this scale is loaded yet (that's sub-project 4).
