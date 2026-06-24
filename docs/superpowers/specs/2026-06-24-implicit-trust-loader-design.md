# ImplicitTrustLoader (Mode B) — Design Spec

Date: 2026-06-24

## Context: this is sub-project 3 of 5

Part of the "Grand Unified Benchmark Arena" initiative:

1. ~~`pipeline/utils/sparse_jaccard.py`~~ — DONE, merged.
2. ~~`dataset_factory.py` + consolidation of Ciao/Yelp/FilmTrust~~ — DONE, merged.
3. **`ImplicitTrustLoader` (Mode B)** (THIS SPEC)
4. New explicit-trust datasets (Douban, Epinions variants, Flixster)
5. `grand_arena_runner.py` — config-driven orchestrator tying 1-4 together

## Decisions already made (binding on this sub-project)

- **Mode B framing** (from sub-project 1's spec): Jaccard-derived "trust" must always
  be presented as an explicit, clearly-labeled ablation/limitation study — never as a
  real social benchmark. This sub-project's console logging and `ArenaDataset.mode`
  field exist specifically to make this distinction impossible to miss downstream.
- **Dataset scope for this sub-project: MovieLens-100K only.** ML-100K is already
  downloaded locally (`data/ml-100k/u.data`) and has a simple, well-understood format.
  ML-1M/10M use a different raw format (`::`-delimited `ratings.dat`) and Jester is a
  dense matrix, not a sparse triplet list — neither's real source URL/format has been
  verified in this session. Per the same "no guessed URLs/formats" rule that governed
  the FilmTrust and sub-project-2 work, registering ML-1M/10M/Jester is deferred to a
  follow-up once each is individually verified — same precedent as deferring
  Douban/Epinions/Flixster to sub-project 4. The architecture below is fully
  config-driven so adding them later is "add a config entry," not new code, same as
  `ExplicitTrustLoader`.
- **Config layer: separate `ImplicitDatasetConfig` + separate `IMPLICIT_DATASET_REGISTRY`**,
  not an extension of the existing `DatasetConfig`. Implicit datasets need
  `jaccard_threshold`/`jaccard_top_k`/`jaccard_chunk_size` and have no real trust file
  to download, so a shared `DatasetConfig` would either carry permanently-`None`
  explicit-only fields (`trust_urls`, `feedback_mode`, `rating_threshold`) or
  permanently-irrelevant implicit-only fields — Optional-soup either way. Registry
  *membership* is the dispatch signal for `DatasetFactory.create()`, not a `mode` field
  on the config.
- **Jaccard trust is computed from `train_csr` only, not the full pre-split
  interactions.** This deliberately diverges from the existing, frozen
  `pipeline/engines/unified_data_loader.py::build_implicit_trust_matrix()`, which
  computes trust from the full dataset before any split (that module doesn't split at
  all). Computing the synthetic trust graph from train-only interactions avoids
  leaking test-set co-occurrence information into the trust side-channel that
  TrustSVD-style models consume — a real evaluation-leakage risk specific to *derived*
  trust (this doesn't apply to Mode A's real trust data, which is independent ground
  truth, not derived from ratings).
- **`pipeline/data_loaders/explicit_trust_loader.py` (sub-project 2, already shipped)
  is refactored, not frozen.** Unlike `pipeline/engines/unified_data_loader.py` (which
  stays frozen because production code depends on it) or the `unified_arena`/
  `academic_sandbox`/`filmtrust_arena` packages (frozen because they're separate,
  parallel-built arenas), `pipeline/data_loaders/` is this initiative's own package,
  still under active construction across sub-projects. Extracting its ~150 lines of
  dataset-agnostic helpers (download-with-fallback, generic row parsing, k-core
  filtering, stratified split, interaction-matrix construction) into a shared module
  is a pure code-move with no behavior change, verified by re-running sub-project 2's
  existing Ciao/Yelp/FilmTrust equivalence script afterward and confirming identical
  output.

## Problem

`ImplicitTrustLoader` needs ~90% of the same machinery `ExplicitTrustLoader` already
has (download, parse, optionally k-core filter, split, build the interaction matrix) —
only the *trust* construction differs (download a real file vs. synthesize one via
Jaccard). Duplicating that machinery into a new file would directly contradict the
Open/Closed intent the whole factory was built for in sub-project 2.

## Goal

A `BaseDatasetLoader` implementation for datasets with no real social network, that:
- Parses standard MovieLens-style rating files into an `ArenaDataset`.
- Synthesizes `social_csr` via `pipeline/utils/sparse_jaccard.py::compute_sparse_jaccard_trust`
  (the OOM-safe engine built in sub-project 1), computed from train-only interactions.
- Is unmistakably labeled, in both data (`ArenaDataset.mode == "implicit"`) and console
  output, as an ablation study — never confusable with a real social benchmark.
- Integrates into `DatasetFactory` without special-casing per dataset.

## Architecture

### File structure

```
pipeline/data_loaders/
├── loader_utils.py            # NEW -- shared, dataset-agnostic helpers
├── base_loader.py             # unchanged (Task 1, sub-project 2)
├── dataset_configs.py         # MODIFIED -- adds ImplicitDatasetConfig + IMPLICIT_DATASET_REGISTRY
├── explicit_trust_loader.py   # MODIFIED -- delegates to loader_utils, no behavior change
├── implicit_trust_loader.py   # NEW -- ImplicitTrustLoader
└── dataset_factory.py         # MODIFIED -- checks both registries
```

### `loader_utils.py` (new)

Pure extraction of currently-private `ExplicitTrustLoader` methods into standalone,
dataset-agnostic functions:

```python
def download_with_fallback(urls: List[str], data_dir: str) -> None: ...
def files_exist(data_dir: str, filenames: List[str]) -> bool: ...
def resolve_path(data_dir: str, filenames: List[str]) -> str: ...
def parse_rows(path: str, delimiter: str, rating_col_index: int = 2,
                explicit_rating_col_index: Optional[int] = None) -> pd.DataFrame: ...
def split_line(line: str, delimiter: str) -> List[str]: ...
def k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]: ...
def stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int,
                               binarize: bool) -> sp.csr_matrix: ...
def build_dict(df: pd.DataFrame) -> Dict[int, Set[int]]: ...
```

`ExplicitTrustLoader` is refactored to call these instead of defining them inline — its
public interface (`ExplicitTrustLoader(config).load() -> ArenaDataset`) and observable
behavior are unchanged. `ImplicitTrustLoader` imports the same functions for the parts
of its pipeline that are identical (download, parse, optional k-core, split, build
train matrix), and adds only what's genuinely new: calling `compute_sparse_jaccard_trust`
instead of downloading a trust file.

### `ImplicitDatasetConfig` (new, in `dataset_configs.py`)

```python
@dataclass
class ImplicitDatasetConfig:
    """
    Declarative configuration for ImplicitTrustLoader (Mode B / ablation study).

    Fields:
        name, data_dir: same meaning as DatasetConfig.
        ratings_urls, ratings_filenames: same meaning as DatasetConfig -- no trust_urls/
            trust_filenames, since trust is synthesized, not downloaded.
        delimiter: "space" handles ML-100K's tab-delimited u.data correctly, since
            Python's str.split() with no argument splits on any whitespace run
            (including tabs), exactly matching ExplicitTrustLoader's existing "space"
            mode -- no new delimiter type needed for this dataset.
        rating_col_index: column index of the rating value in a parsed row (ML-100K's
            "user item rating timestamp" layout puts it at index 2).
        k_core: minimum interactions per user/item. None for ML-100K -- GroupLens
            already guarantees every user has >=20 ratings, so no further filtering
            is needed.
        jaccard_threshold, jaccard_top_k, jaccard_chunk_size: passed directly to
            compute_sparse_jaccard_trust (pipeline/utils/sparse_jaccard.py).
        test_ratio, seed: same meaning as DatasetConfig.
    """
    name: str
    data_dir: str
    ratings_urls: List[str]
    ratings_filenames: List[str]
    delimiter: str = "space"
    rating_col_index: int = 2
    k_core: Optional[int] = None
    jaccard_threshold: float = 0.3
    jaccard_top_k: Optional[int] = 50
    jaccard_chunk_size: int = 2000
    test_ratio: float = 0.2
    seed: int = 42


ML_100K_CONFIG = ImplicitDatasetConfig(
    name="ml-100k",
    data_dir="data/ml-100k",
    ratings_urls=["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
    ratings_filenames=["u.data"],
)

IMPLICIT_DATASET_REGISTRY: Dict[str, ImplicitDatasetConfig] = {
    "ml-100k": ML_100K_CONFIG,
}
```

The `ratings_urls` entry was verified live via WebFetch during design (returns a real
4.7MB zip containing `u.data`, `README`, `allbut.pl`, `mku.sh` — matching the locally
present `data/ml-100k/` contents exactly), not guessed.

### `ImplicitTrustLoader` (new, in `implicit_trust_loader.py`)

```python
class ImplicitTrustLoader(BaseDatasetLoader):
    def __init__(self, config: ImplicitDatasetConfig): ...

    def load(self) -> ArenaDataset:
        """
        download -> parse -> (optional k-core) -> split -> build train_csr (explicit
        ratings) -> compute_sparse_jaccard_trust(train_csr only) -> gc.collect() ->
        return ArenaDataset(mode="implicit").
        """
```

Pipeline detail:
1. `loader_utils.download_with_fallback(cfg.ratings_urls, cfg.data_dir)` (skips if
   `loader_utils.files_exist(...)` already true).
2. `loader_utils.parse_rows(path, cfg.delimiter, cfg.rating_col_index)`.
3. `loader_utils.k_core_filter(...)` only if `cfg.k_core is not None`.
4. Build contiguous user/item ID maps (ratings-file users only — there's no trust file
   to union against in Mode B, so this question from sub-project 2 doesn't arise here).
5. `loader_utils.stratified_split(df, cfg.test_ratio, cfg.seed)`.
6. `train_csr = loader_utils.build_interaction_matrix(df_train, n_users, n_items, binarize=False)`
   — explicit rating values, matching the existing frozen loader's convention for
   Funk-SVD/TrustSVD-style consumers.
7. `social_csr = compute_sparse_jaccard_trust(train_csr, threshold=cfg.jaccard_threshold, top_k=cfg.jaccard_top_k, chunk_size=cfg.jaccard_chunk_size)`
   — **train_csr only**, per the binding decision above.
8. `gc.collect()` immediately after, on top of `compute_sparse_jaccard_trust`'s own
   internal `gc.collect()` calls — explicit, defensive, per requirement.
9. Return `ArenaDataset(..., mode="implicit", ...)`.

### Console logging (verbose, per requirement)

A loud, unmissable banner at the start of every `load()` call:

```
================================================================================
[ImplicitTrustLoader:ml-100k] ABLATION STUDY ONLY -- Mode B
  Trust graph is SYNTHETIC (Jaccard co-occurrence similarity), NOT real social data.
  Do not present results using this loader as a genuine social-aware benchmark.
================================================================================
```

Plus progress/memory lines at each step (row counts after parse/filter/split,
`train_csr.nnz`, and after the Jaccard call: `social_csr.nnz` labeled
"(synthetic, Mode B)"), consistent with `ExplicitTrustLoader`'s existing logging style.

### `DatasetFactory.create(name)` (modified)

```python
@staticmethod
def create(name: str) -> BaseDatasetLoader:
    key = name.lower()
    if key in DATASET_REGISTRY:
        return ExplicitTrustLoader(DATASET_REGISTRY[key])
    if key in IMPLICIT_DATASET_REGISTRY:
        return ImplicitTrustLoader(IMPLICIT_DATASET_REGISTRY[key])
    available = ", ".join(sorted(list(DATASET_REGISTRY.keys()) + list(IMPLICIT_DATASET_REGISTRY.keys())))
    raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
```

Explicit registry is checked first (arbitrary but documented precedence; no name
collisions exist today between the two registries). No `mode` field is needed on
either config to drive this dispatch — registry membership is the signal.

## Out of scope

- ML-1M, ML-10M, Jester — deferred until their real source URLs/formats are verified
  (architecture supports adding them as config entries with no new loader code, same
  as Douban/Epinions/Flixster in sub-project 4).
- `pipeline/engines/unified_data_loader.py` — remains frozen; production's
  `pipeline/run_pipeline.py` depends on it. This sub-project's loader is a new,
  parallel implementation, not a replacement, and deliberately differs from it (see
  the train-only-Jaccard decision above).
- `grand_arena_runner.py` — sub-project 5.
- Retiring/modifying `unified_arena/`, `academic_sandbox/`, `filmtrust_arena/` and
  their CLI runners — still deferred, unrelated to this sub-project.

## Verification plan

No pytest framework — direct script execution with documented exact expected output,
consistent with every other module in this codebase.

1. **Refactor-safety check:** after extracting `loader_utils.py` and refactoring
   `ExplicitTrustLoader` to use it, re-run sub-project 2's existing Ciao/Yelp/FilmTrust
   equivalence script unchanged and confirm it still produces
   `ALL EQUIVALENCE CHECKS PASSED` with the same numbers as before — proof the
   extraction changed nothing observable.
2. **ML-100K load smoke test:** `DatasetFactory.create("ml-100k").load()` — confirm
   `ArenaDataset.mode == "implicit"`, `num_users`/`num_items` match the known ML-100K
   shape (943 users x 1682 items), `train_csr` holds explicit rating values (not
   binary), the Mode B disclaimer banner is printed, and `get_sym_adj_mat()` (inherited
   from `ArenaDataset`) still works against this loader's output.
3. **Jaccard output sanity check:** `social_csr` is symmetric
   (`(social_csr != social_csr.T).nnz == 0`), zero-diagonal, and `nnz` is bounded
   roughly by `2 * jaccard_top_k * num_users` (the same bound `sparse_jaccard.py`'s own
   spec documents) — confirming the OOM-safe engine is actually being exercised, not
   bypassed.
4. **Factory dispatch check:** `DatasetFactory.create("ml-100k")` returns an
   `ImplicitTrustLoader` instance; `DatasetFactory.create("ciao")` still returns an
   `ExplicitTrustLoader` instance (no regression); an unknown name's `ValueError`
   message lists all four dataset names across both registries.
