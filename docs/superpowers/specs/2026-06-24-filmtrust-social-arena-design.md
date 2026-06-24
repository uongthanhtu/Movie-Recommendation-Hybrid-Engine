# FilmTrust Social Arena — Design Spec

Date: 2026-06-24

## Problem

The current `pipeline/engines/benchmark_arena.py` ("Classic Arena") compares 5 engines —
Funk-SVD, TrustSVD, LightGCN, Social-LightGCN, SASRec — on MovieLens-100k. TrustSVD and
Social-LightGCN are Social-Aware models, but MovieLens has no real social graph, so
`pipeline/engines/unified_data_loader.py::build_implicit_trust_matrix()` fabricates one
via Jaccard similarity over co-interacted items. Per mentor feedback, this is not a
scientifically valid way to benchmark Social-Aware models — Jaccard-derived "trust" is
just another collaborative-filtering signal, not a trust network, so comparing TrustSVD /
Social-LightGCN against it doesn't isolate what social regularization actually contributes.

## Goal

Split benchmarking into two strictly separated arenas:

1. **Classic Arena** (existing, MovieLens-100k): Funk-SVD, SASRec, LightGCN only. No
   social models, no Jaccard trust generation in this path.
2. **Social Arena** (new, FilmTrust): LightGCN (no-social baseline), TrustSVD,
   Social-LightGCN — evaluated against FilmTrust's real, explicit, user-asserted trust
   network (`trust.txt`), not a derived proxy.

## Dataset: FilmTrust

Verified directly against the canonical source (`guoguibing/librec` GitHub repo, path
`librec/demo/Datasets/FilmTrust/`) rather than assumed from the task description:

- `ratings.txt`: 35,497 ratings, 1,508 users, 2,071 items. **Space-delimited**,
  3 columns: `userId itemId rating` (ratings observed in 0.5 increments, e.g. `1 3 3.5`).
- `trust.txt`: 1,853 directed trust edges. **Space-delimited**, 3 columns:
  `trustorId trusteeId trustValue` (trust value is always `1` — binary, directed).
- Raw URLs:
  - `https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/ratings.txt`
  - `https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/trust.txt`
- No timestamp column exists, so a leave-last-one-out split (used by Classic Arena) is
  not possible. We use a stratified 80/20 per-user split instead — the same pattern
  already implemented in `pipeline/academic_sandbox/yelp_data_loader.py::_stratified_split`.

## Architecture

### New package: `pipeline/filmtrust_arena/`

Mirrors the existing `pipeline/unified_arena/` (CiaoDVD) and `pipeline/academic_sandbox/`
(Yelp) packages, each of which bundles a loader + CLI runner for one external dataset.

```
pipeline/filmtrust_arena/
├── __init__.py
├── filmtrust_loader.py   # FilmTrustLoader: download, parse, split, build CSR matrices
└── run_filmtrust.py      # CLI orchestrator: train 3 engines, evaluate, print table
```

#### `filmtrust_loader.py` — `FilmTrustLoader`

Lifecycle mirrors `YelpDataLoader` for consistency with existing conventions:

- `__init__(data_dir="data/filmtrust", test_ratio=0.2, seed=42)`
- `download(force=False)`: fetches `ratings.txt`/`trust.txt` via `urllib.request` (project
  convention — no `requests` dependency) if not already present under `data_dir`. No ZIP
  involved (unlike Yelp) since these are two plain text files.
- `load_data()`: parses both files (space-delimited), builds contiguous 0-indexed user/item
  ID mappings (union of users appearing in ratings or trust), stratified 80/20 per-user
  split of ratings into train/test.
- `get_train_interaction_matrix() -> sp.csr_matrix`: (num_users × num_items), **explicit**
  rating values (not binarized) — same convention as
  `UnifiedDataLoader.build_sparse_matrix()`.
- `get_sym_adj_mat() -> sp.csr_matrix`: symmetric-normalized bipartite adjacency
  ((num_users+num_items) × (num_users+num_items)), built from train interactions only —
  required by `LightGCNEngine.fit()`. Same construction as
  `UnifiedDataLoader.build_bipartite_graph()`.
- `get_trust_matrix() -> sp.csr_matrix`: (num_users × num_users), **raw** (unnormalized)
  weights — symmetrized as `A_trust + A_trust.T`, clipped to binary. Normalization happens
  inside `TrustSVDEngine`/`SocialLightGCNEngine.fit()` already, so the loader must NOT
  pre-normalize (matching what `benchmark_arena.py` currently passes in).
- `get_test_dict() -> Dict[int, Set[int]]` / `get_train_dict() -> Dict[int, Set[int]]`.

No Jaccard / implicit-trust code anywhere in this file.

#### `run_filmtrust.py`

Orchestration follows `pipeline/engines/benchmark_arena.py`'s per-engine loop, not the
Yelp sandbox's custom-wrapper approach — because FilmTrust has explicit ratings + explicit
trust, the production engines (built for exactly that data shape) apply directly with zero
new model code:

- Initializes `LightGCNEngine`, `TrustSVDEngine`, `SocialLightGCNEngine` (imported from
  `pipeline.engines.*`, unmodified).
- Feeds each its expected `fit(data)` dict shape (`sym_adj_mat`+`interaction_matrix` for
  LightGCN; `interaction_matrix`+`trust_matrix` for the other two).
- Evaluation: imports `recall_at_k`/`ndcg_at_k` from `pipeline.engines.benchmark_arena`
  (reused, not redefined) plus per-user latency timing around `recommend_top_n`, same
  pattern as the existing Classic Arena loop.
- Prints a pandas `.to_string(index=False)` ASCII table (Engine, Train Time, Recall@10,
  NDCG@10, Latency ms, P95 Latency ms) and saves
  `models/filmtrust_arena_results.csv` — consistent with `benchmark_arena.py`'s output
  convention. (No `rich` dependency is added — it isn't in `requirements.txt` and no other
  arena script uses it; "rich ASCII table" is interpreted as a clean formatted table, not
  the literal `rich` library.)
- CLI args: `--data_dir`, `--epochs`, `--dim` (embedding dim), `--layers`, `-k` — matching
  the flag conventions of `run_yelp_benchmark.py` / `run_arena.py`.

### Changes to existing files

- **`pipeline/engines/benchmark_arena.py`**: remove `TrustSVD` and `Social-LightGCN` from
  the `engines` dict and their associated `fit()` branch + trust-matrix wiring. Funk-SVD,
  LightGCN, SASRec remain. The `build_implicit_trust_matrix()` step is no longer invoked
  from this script.
- **`pipeline/engines/unified_data_loader.py`**: **no changes**. `build_implicit_trust_matrix`
  stays — `pipeline/run_pipeline.py` (production training feeding the live FastAPI hybrid
  recommender in `app/main.py`) still depends on it. Out of scope for this refactor.
- **`README.md`**:
  - §2.1 table: drop TrustSVD/Social-LightGCN rows (Classic Arena is now 3 models);
    adjust prose.
  - New §2.4 "Social-Aware Arena (FilmTrust — Explicit Trust)": explains the academic
    rationale (Jaccard-derived trust is not a valid trust network), the FilmTrust dataset,
    and presents the 3-model comparison table — populated with **real measured numbers**
    from actually running `run_filmtrust.py` (not fabricated).
  - §7 directory tree: add `pipeline/filmtrust_arena/`.
  - §8 Quick Start: add a subsection with the `python -m pipeline.filmtrust_arena.run_filmtrust`
    invocation, alongside the existing CiaoDVD/Yelp examples.

## Out of scope

- `pipeline/run_pipeline.py` and production serving (`app/main.py`'s hybrid recommender) —
  unchanged, continues using Jaccard-based TrustSVD for live recommendations.
- No new third-party dependencies (`rich`, `requests`) — everything implemented with
  `urllib.request`, `scipy.sparse`, `pandas`, `torch`, matching existing project convention.

## Verification plan

1. Run `python -m pipeline.filmtrust_arena.run_filmtrust` end-to-end after implementation —
   confirms the loader downloads/parses correctly and all 3 engines train and evaluate
   without error.
2. Run `python -m pipeline.engines.benchmark_arena` after trimming — confirms Classic
   Arena still runs cleanly with 3 engines.
3. Capture the real output table from step 1 and transcribe it into README §2.4 verbatim
   (no invented numbers).
