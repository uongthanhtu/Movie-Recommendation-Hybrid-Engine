# Dataset Factory Consolidation — Design Spec

Date: 2026-06-24

## Context: this is sub-project 2 of 5

Part of the "Grand Unified Benchmark Arena" initiative, decomposed (with user approval,
recorded in `docs/superpowers/specs/2026-06-24-sparse-jaccard-design.md`) into:

1. ~~`pipeline/utils/sparse_jaccard.py`~~ — DONE, merged.
2. **`dataset_factory.py` + consolidation** (THIS SPEC)
3. `ImplicitTrustLoader` (Mode B) — Jaccard-based trust for MovieLens/Jester, built on (1)
4. New explicit-trust datasets (Douban, Epinions variants, Flixster)
5. `grand_arena_runner.py` — config-driven orchestrator tying 1-4 together

## Problem

Reading the three existing "explicit trust" dataset loaders in full (not just their
docstrings) surfaced two distinct duplication problems:

1. **Three different output shapes for the same conceptual data.** `pipeline/unified_arena/academic_data_loader.py::AcademicDataLoader.load()`
   returns a clean `ArenaDataset` dataclass. `pipeline/academic_sandbox/yelp_data_loader.py::YelpDataLoader`
   and `pipeline/filmtrust_arena/filmtrust_loader.py::FilmTrustLoader` instead expose a bag
   of getter methods with no shared base, and `FilmTrustLoader` alone also exposes a
   `get_sym_adj_mat()` that the other two don't (because only its consumer,
   `run_filmtrust.py`, feeds a `LightGCNEngine` that needs it).
2. **Real, non-cosmetic per-dataset parsing differences**, currently hand-coded
   per-loader: Ciao auto-detects comma-vs-space delimiters, runs iterative 5-core
   filtering, and binarizes ratings via a 3.0 threshold before building `train_csr`.
   FilmTrust keeps explicit ratings untouched (TrustSVD needs them). Yelp's raw data is
   already binary. None of this is dataset-specific *logic* so much as dataset-specific
   *parameters* — Ciao's and Yelp's row parsers are in fact nearly identical already
   (both branch on column count: `>=5` / `>=3` / `==2`).

Separately (noted but explicitly out of scope, see below): `pipeline/unified_arena/model_adapters.py`
defines a second model interface, `BaseAdapter`, whose `SocialLightGCNAdapter`/`VanillaLightGCNAdapter`
reimplement training loops that already exist in `pipeline/engines/social_lightgcn_engine.py`/`lightgcn_engine.py`
(the `BaseRecommenderEngine` versions that `run_filmtrust.py` already reuses directly).

## Goal

A single, declarative, Factory-Method-based dataset loading system
(`pipeline/data_loaders/`) that:
- Produces one canonical output type (`ArenaDataset`) for all explicit-trust datasets.
- Lets adding a new dataset (Douban/Epinions/Flixster, sub-project 4) be "add a config
  entry," not "write a new parsing class" — genuine Open/Closed compliance.
- Is built *alongside* the three existing arenas without breaking them.

## Decisions made (binding scope boundaries for this sub-project)

- **Existing arenas (`unified_arena/`, `academic_sandbox/`, `filmtrust_arena/`) and their
  CLI runners (`run_arena.py`, `run_yelp_benchmark.py`, `run_filmtrust.py`) are left
  completely untouched and stay working.** The new factory is built in parallel, with
  fresh (config-driven) implementations of the same parsing logic — not by importing or
  modifying the old loaders. This accepts temporary duplication between the old loaders
  and the new factory as the cost of not breaking three working, already-shipped,
  already-reviewed scripts in this sub-project. Retiring the old loaders/runners is
  deferred to sub-project 5, once `grand_arena_runner.py` can fully supersede all three
  in one clean swap.
- **`pipeline/unified_arena/model_adapters.py` (`BaseAdapter` and its model duplication)
  is out of scope.** `dataset_factory.py` only concerns data loading. The model-interface
  duplication is a real, separate problem, deferred to whichever future sub-project picks
  one model interface for the grand arena (most likely sub-project 5).
- **`pipeline/engines/academic_benchmark_arena.py` and `pipeline/engines/academic_data_loader.py`
  are deleted in this sub-project.** Confirmed via `grep` that the only reference to
  `pipeline.engines.academic_data_loader` is `academic_benchmark_arena.py`'s own import,
  and nothing else in the repo references `academic_benchmark_arena` except our own
  design docs. This is genuinely dead code (an early Ciao/Epinions sandbox fully
  superseded by `pipeline/unified_arena/`), distinct from the *working* loaders named
  above.
- **`pipeline/engines/unified_data_loader.py` remains untouched** (per the prior
  sub-project's spec — production's `pipeline/run_pipeline.py` depends on it).

## Architecture

### File structure

```
pipeline/data_loaders/
├── __init__.py
├── base_loader.py            # ArenaDataset dataclass + BaseDatasetLoader ABC
├── dataset_configs.py        # DatasetConfig dataclass + registry (ciao, yelp, filmtrust)
├── explicit_trust_loader.py  # ExplicitTrustLoader(BaseDatasetLoader) -- the one generic implementation
└── dataset_factory.py        # DatasetFactory.create(name) -> BaseDatasetLoader
```

This is the **Factory Method** pattern: `DatasetFactory.create("ciao")` looks up the
registry in `dataset_configs.py` and returns a configured `ExplicitTrustLoader`.
`BaseDatasetLoader` is the abstract product; `ExplicitTrustLoader` (and, once
sub-project 3 lands, `ImplicitTrustLoader`) are concrete products. Adding Douban,
Epinions, or Flixster later means adding one `DatasetConfig` entry to the registry, not
writing a new class.

### `ArenaDataset` (in `base_loader.py`)

Generalizes the dataclass already used by `unified_arena/academic_data_loader.py`
(judged the cleanest of the three existing shapes) rather than inventing a new one:

```python
@dataclass
class ArenaDataset:
    num_users: int
    num_items: int
    train_csr: sp.csr_matrix         # (num_users, num_items)
    test_dict: Dict[int, Set[int]]   # {user_idx: set(item_idx)}
    train_dict: Dict[int, Set[int]]  # {user_idx: set(item_idx)}
    social_csr: sp.csr_matrix        # (num_users, num_users), symmetric
    mode: str = "explicit"           # "explicit" now; "implicit" once sub-project 3 lands

    # Raw stats for reporting (grand_arena_runner's structural metadata logging, sub-project 5)
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
        built from train_csr lazily on first call and cached. Consumers that build
        their own normalization internally (e.g. unified_arena's BaseAdapter-style
        adapters) never construct this and pay nothing for it.
        """
```

### `BaseDatasetLoader` (in `base_loader.py`)

```python
class BaseDatasetLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> ArenaDataset:
        """Full pipeline: download -> parse -> (optional filter) -> split -> matrices."""
```

One method, matching `AcademicDataLoader.load()`'s existing convention exactly (download
is handled internally, not as a separate public step) — simplifies the factory's usage
to `DatasetFactory.create(name).load()`.

### `DatasetConfig` (in `dataset_configs.py`)

Declarative; covers the real differences found by reading the three existing loaders:

```python
@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    ratings_urls: List[str]          # fallback mirrors, tried in order
    trust_urls: List[str]
    ratings_filenames: List[str]     # candidate filenames to resolve after download/extraction
    trust_filenames: List[str]
    delimiter: str = "auto"          # "auto" | "comma" | "space" -- "auto" replicates Ciao's try-comma-else-space
    explicit_rating_col_index: int = 2  # only consulted for Ciao-style 5+-column rows
    k_core: Optional[int] = None     # None = skip filtering (Yelp/FilmTrust); 5 = Ciao
    feedback_mode: str = "explicit"  # "explicit" | "threshold_binarize"
    rating_threshold: float = 0.0    # only used for threshold_binarize (Ciao: 3.0)
    test_ratio: float = 0.2
    seed: int = 42
```

Registered instances for the 3 known datasets (`CIAO_CONFIG`, `YELP_CONFIG`,
`FILMTRUST_CONFIG`) live in a `DATASET_REGISTRY: Dict[str, DatasetConfig]` in the same
file, keyed by lowercase name (`"ciao"`, `"yelp"`, `"filmtrust"`).

### `ExplicitTrustLoader` (in `explicit_trust_loader.py`)

The one generic implementation, parameterized entirely by a `DatasetConfig`:

1. **Download:** for each URL in `ratings_urls`/`trust_urls` (tried in order until one
   succeeds, matching Ciao's existing multi-mirror fallback pattern), detect zip-vs-raw
   by attempting `zipfile.is_zipfile()` on the downloaded bytes (not by URL extension,
   which can be misleading) — extract if it's a zip, save directly if not. Resolve actual
   file paths by walking `data_dir` for any of `ratings_filenames`/`trust_filenames`
   (case-insensitive), matching the existing loaders' file-resolution convention.
2. **Parse:** one shared, generic row parser used for both ratings and trust files,
   branching on column count after delimiter-splitting (`auto`/`comma`/`space`):
   `>=5` columns → use `explicit_rating_col_index` (Ciao's `categoryId`/`reviewId` case);
   `>=3` → column 2 is the rating/weight; `==2` → implicit, weight defaults to `1.0`.
   This single function replaces Ciao's `_parse_ratings`/`_parse_trust`, Yelp's
   `_read_interaction_file`/`_read_trust_file`, and FilmTrust's `_read_space_delimited` —
   all three are already doing near-identical column-count branching today.
3. **Feedback mode (runs BEFORE k-core, matching Ciao's existing order exactly):** if
   `threshold_binarize`, filter rows to `rating >= config.rating_threshold` first, then
   mark the surviving rows for binary storage (the rating *value* itself is dropped after
   this filter — only row presence matters from here on); if `explicit`, keep raw values
   and apply no filter (Yelp's already-binary raw data passes through unchanged either
   way). Getting this order right matters: k-core counts in the next step must reflect
   the *post-threshold* row set, exactly as `AcademicDataLoader.load()` currently does
   (threshold filter at the top of `load()`, `_k_core_filter` called after).
4. **Optional k-core filter:** only runs if `config.k_core is not None`, on the
   already-threshold-filtered rows. Extracted verbatim (dataset-agnostic) from
   `AcademicDataLoader._k_core_filter`.
5. **Split:** per-user stratified 80/20 (test_ratio configurable), identical logic to all
   three existing loaders' `_stratified_split`.
6. **Build matrices:** `train_csr` stores binary `1.0` per interaction if
   `feedback_mode == "threshold_binarize"` (matching `AcademicDataLoader._build_interaction_matrix`,
   which always binarizes regardless of the original rating value), or the real rating
   values if `feedback_mode == "explicit"` (matching `FilmTrustLoader.get_train_interaction_matrix`).
   `social_csr` via `A + A.T` clipped to binary (all three datasets' raw trust weights are
   already binary `1`, so `+`-then-clip is correct here — unlike the *continuous* Jaccard
   weights in sub-project 1, where `.maximum()` was required instead).
7. Return a populated `ArenaDataset`.

### `DatasetFactory` (in `dataset_factory.py`)

```python
class DatasetFactory:
    @staticmethod
    def create(name: str) -> BaseDatasetLoader:
        """Look up name in DATASET_REGISTRY, return a configured loader."""
```

### Memory safety, scoped honestly

Ciao/Yelp/FilmTrust are all tens-of-thousands of rows. The existing per-line file reads
(`for line in f:`) are already streaming and memory-safe at this scale; this sub-project
does not need new chunking for file parsing. The real OOM risk zone (100K+ users) lives
in trust-graph *computation* for Mode B (sub-project 3, where `pipeline/utils/sparse_jaccard.py`
already applies) and potentially very large new datasets in sub-project 4 — not in
parsing three small-to-medium text files. No chunking is retrofitted here where the
existing pattern already suffices.

### Error handling

Matches existing precedent: multi-mirror download fallback (raise a clear, actionable
error only after all mirrors fail, naming the manual-placement path as a fallback, per
`FilmTrustLoader`'s existing convention); malformed/short rows are skipped during
parsing rather than crashing the whole load (matching Ciao's existing tolerant parsing,
which already skips comment lines and rows with too few columns).

## Out of scope

- Modifying or deleting `unified_arena/`, `academic_sandbox/`, `filmtrust_arena/`, or
  their CLI runners — confirmed decision, deferred to sub-project 5.
- `unified_arena/model_adapters.py` / `BaseAdapter` consolidation — confirmed out of
  scope, a model-layer concern, not a data-loader one.
- `pipeline/engines/unified_data_loader.py` — frozen, production dependency.
- `ImplicitTrustLoader` (Mode B) — sub-project 3.
- New datasets beyond Ciao/Yelp/FilmTrust (Douban, Epinions, Flixster) — sub-project 4.
  This spec's `DatasetConfig` schema is designed so those *should* need no new code, but
  their real source URLs/formats are not yet verified and will be confirmed in
  sub-project 4's own spec (no guessed URLs, same lesson learned from FilmTrust).

## In scope, confirmed for deletion

- `pipeline/engines/academic_benchmark_arena.py`
- `pipeline/engines/academic_data_loader.py`

(Both confirmed dead: the only reference to the latter is the former's own import; the
former is referenced nowhere else in the repository.)

## Verification plan

No pytest/unit-test framework exists in this repo (established convention from prior
sub-projects). Verification follows the same pattern: runnable scripts with documented
expected output, run for real against the actual Ciao/Yelp/FilmTrust datasets (already
downloaded locally from prior sessions, or re-downloaded fresh by the new factory code
— both should work and produce equivalent statistics).

1. **Per-dataset equivalence check:** for each of Ciao/Yelp/FilmTrust, load via the new
   `DatasetFactory.create(name).load()` and compare key statistics (`num_users`,
   `num_items`, `train_csr.nnz`, `social_csr.nnz`, `len(test_dict)`) against what the
   *existing* (untouched) loader produces for the same dataset. They should match
   closely — exact equality isn't guaranteed for the stratified split's RNG path (the new
   shared splitter may consume randomness in a different order than each old loader's
   bespoke implementation, even with the same seed), but the same dataset size,
   train/test ratio, and approximate trust-edge count should hold.
2. **`get_sym_adj_mat()` smoke test:** confirm it returns a correctly-shaped, symmetric
   matrix once called, and that calling it twice returns the same cached object (not
   recomputed).
3. **Dead-code removal verification:** after deleting the two legacy files, confirm
   nothing in the repo still imports them (re-run the same `grep` used during design),
   and that the existing, untouched arena scripts still run (a quick smoke run of
   `run_arena.py`/`run_yelp_benchmark.py`/`run_filmtrust.py`, or at minimum a clean
   import check, to confirm the deletion had zero blast radius on working code).
