# Dataset Factory Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Factory-Method-based, declarative dataset loading system (`pipeline/data_loaders/`) that produces one canonical output (`ArenaDataset`) for explicit-trust datasets, consolidating the parsing logic currently duplicated across three inconsistent loaders, while leaving those three loaders and their CLI runners completely untouched and working.

**Architecture:** `DatasetFactory.create(name)` (Factory Method) looks up a `DatasetConfig` (declarative, one per dataset) in a registry and returns a configured `ExplicitTrustLoader` (the one generic concrete loader, parameterized entirely by its config). `ArenaDataset` (generalized from the dataclass already used by `pipeline/unified_arena/academic_data_loader.py`) is the canonical output type, with a lazily-computed, cached `get_sym_adj_mat()` for LightGCN-style consumers.

**Tech Stack:** Python, `scipy.sparse`, `numpy`, `pandas`, stdlib `zipfile`/`io`/`urllib.request` (no new dependencies).

## Global Constraints

- No new third-party dependencies.
- `pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/` packages and their CLI runners (`run_arena.py`, `run_yelp_benchmark.py`, `run_filmtrust.py`) MUST remain completely untouched and working. The new factory is built in parallel, not as a replacement — retirement is deferred to sub-project 5.
- `pipeline/unified_arena/model_adapters.py` (`BaseAdapter`) is out of scope — do not touch.
- `pipeline/engines/unified_data_loader.py` is out of scope — frozen, `pipeline/run_pipeline.py` depends on it.
- `pipeline/engines/academic_benchmark_arena.py` and `pipeline/engines/academic_data_loader.py` ARE to be deleted in this plan — confirmed dead code (the only reference to the latter is the former's own import; the former is referenced nowhere else in the repository).
- **New canonical behavior, by design:** the new loader unions ratings-file and trust-file users into one user universe. This matches `YelpDataLoader`'s and `FilmTrustLoader`'s *existing* behavior already, but differs from `AcademicDataLoader` (Ciao)'s existing behavior, which only counts users that appear in the ratings file. This is intentional — see Task 3 for why and how the equivalence check accounts for it. It does not change Ciao's own existing, untouched loader; it only applies to the new, parallel `ExplicitTrustLoader`.
- Trust-matrix symmetrization uses `A + A.T` then clip-to-1 (correct here — all 3 datasets' raw trust weights are binary `1`). This is different from `pipeline/utils/sparse_jaccard.py`'s `.maximum()`, which was required there specifically because Jaccard weights are continuous; do not "fix" this to `.maximum()` here, it would be a no-op on binary data and inconsistent with the existing precedent in `AcademicDataLoader`/`YelpDataLoader`/`FilmTrustLoader`.
- Feedback-mode threshold filtering MUST run before k-core filtering (k-core counts must reflect the post-threshold row set, matching `AcademicDataLoader.load()`'s existing exact order).
- This codebase has no pytest/unit-test framework. Verification is direct script execution against real, already-downloaded data (`data/ciao/`, `data/yelp/` already exist from prior sessions; `data/filmtrust/` will be freshly downloaded by Task 3's verification, exactly as it was in the prior FilmTrust sub-project).

---

### Task 1: `ArenaDataset` + `BaseDatasetLoader`

**Files:**
- Create: `pipeline/data_loaders/__init__.py`
- Create: `pipeline/data_loaders/base_loader.py`

**Interfaces:**
- Produces: `ArenaDataset` dataclass (`num_users: int`, `num_items: int`, `train_csr: sp.csr_matrix`, `test_dict: Dict[int, Set[int]]`, `train_dict: Dict[int, Set[int]]`, `social_csr: sp.csr_matrix`, `mode: str = "explicit"`, plus stats fields `n_train_interactions`, `n_test_interactions`, `n_trust_links`, `n_raw_interactions`, `n_raw_users`, `n_raw_items`, `filtering_rounds`, all `int = 0`) with method `get_sym_adj_mat(self) -> sp.csr_matrix`. `BaseDatasetLoader` ABC with abstract method `load(self) -> ArenaDataset`.

- [ ] **Step 1: Create the package `__init__.py`**

```python
# Grand Unified Benchmark Arena -- dataset loading factory (Factory Method pattern).
```

- [ ] **Step 2: Write `pipeline/data_loaders/base_loader.py`**

```python
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
```

- [ ] **Step 3: Verify**

Run:
```bash
python -c "
import numpy as np
import scipy.sparse as sp
from pipeline.data_loaders.base_loader import ArenaDataset

train_csr = sp.csr_matrix(([1.0,1.0,1.0], ([0,1,1],[0,0,1])), shape=(3,2))
ds = ArenaDataset(
    num_users=3, num_items=2, train_csr=train_csr,
    test_dict={}, train_dict={}, social_csr=sp.csr_matrix((3,3), dtype=np.float32),
)
adj1 = ds.get_sym_adj_mat()
adj2 = ds.get_sym_adj_mat()
assert adj1.shape == (5, 5), f'expected shape (5,5), got {adj1.shape}'
assert adj1 is adj2, 'expected the cached object to be returned on the second call'
assert (adj1 != adj1.T).nnz == 0, 'adjacency must be symmetric'
print('Check 1 (ArenaDataset.get_sym_adj_mat shape/symmetry/caching): PASS')
"
```
Expected output:
```
Check 1 (ArenaDataset.get_sym_adj_mat shape/symmetry/caching): PASS
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/data_loaders/__init__.py pipeline/data_loaders/base_loader.py
git commit -m "feat(data_loaders): add ArenaDataset contract and BaseDatasetLoader ABC

Canonical output type for the new dataset factory, generalized from
the cleanest of the three existing, inconsistent loader shapes
(pipeline/unified_arena/academic_data_loader.py's ArenaDataset)."
```

---

### Task 2: `DatasetConfig` + registry

**Files:**
- Create: `pipeline/data_loaders/dataset_configs.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure data, no imports from `base_loader.py`).
- Produces: `DatasetConfig` dataclass (`name: str`, `data_dir: str`, `ratings_urls: List[str]`, `trust_urls: List[str]`, `ratings_filenames: List[str]`, `trust_filenames: List[str]`, `delimiter: str = "auto"`, `explicit_rating_col_index: int = 2`, `k_core: Optional[int] = None`, `feedback_mode: str = "explicit"`, `rating_threshold: float = 0.0`, `test_ratio: float = 0.2`, `seed: int = 42`); `DATASET_REGISTRY: Dict[str, DatasetConfig]` with keys `"ciao"`, `"yelp"`, `"filmtrust"`. Task 3's `ExplicitTrustLoader` consumes this config's fields by name exactly as listed.

- [ ] **Step 1: Write `pipeline/data_loaders/dataset_configs.py`**

```python
"""
Dataset Configs -- Declarative per-dataset parameters for ExplicitTrustLoader.

Adding a new dataset (Douban, Epinions, Flixster -- a future sub-project) means
adding one DatasetConfig entry to DATASET_REGISTRY here, not writing a new loader
class -- this is what makes the factory Open for extension / Closed for modification.

Each config's fields were derived by reading the three existing, real loaders
(pipeline/unified_arena/academic_data_loader.py, pipeline/academic_sandbox/yelp_data_loader.py,
pipeline/filmtrust_arena/filmtrust_loader.py) and capturing their actual behavior,
not guessed from documentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """
    Declarative configuration for ExplicitTrustLoader.

    Fields:
        name: lowercase registry key, also used in log messages.
        data_dir: local directory for downloaded/extracted files.
        ratings_urls, trust_urls: fallback mirrors, tried in order until one succeeds.
            May point to the same URL for both (a shared zip containing both files,
            like Ciao/Yelp) or to two independent raw files (like FilmTrust) --
            the loader deduplicates and downloads each unique URL once.
        ratings_filenames, trust_filenames: candidate filenames (case-insensitive) to
            resolve on disk after download/extraction.
        delimiter: "auto" (try comma, fall back to whitespace -- Ciao's existing
            heuristic), "comma", or "space".
        explicit_rating_col_index: column index used for the rating value ONLY when a
            row has 5+ columns (Ciao's categoryId/reviewId layout). Rows with 3-4
            columns always use column index 2; rows with exactly 2 columns are
            implicit (weight defaults to 1.0).
        k_core: minimum interactions per user/item for iterative core filtering.
            None skips filtering entirely (Yelp, FilmTrust).
        feedback_mode: "threshold_binarize" filters rows to rating >= rating_threshold
            THEN stores binary 1.0 per surviving interaction (Ciao). "explicit" applies
            no filter and stores real rating values (FilmTrust; also correct for Yelp,
            whose raw values are already binary, so passthrough is a no-op).
        rating_threshold: only consulted when feedback_mode == "threshold_binarize".
        test_ratio, seed: stratified per-user train/test split parameters.
    """
    name: str
    data_dir: str
    ratings_urls: List[str]
    trust_urls: List[str]
    ratings_filenames: List[str]
    trust_filenames: List[str]
    delimiter: str = "auto"
    explicit_rating_col_index: int = 2
    k_core: Optional[int] = None
    feedback_mode: str = "explicit"
    rating_threshold: float = 0.0
    test_ratio: float = 0.2
    seed: int = 42


CIAO_CONFIG = DatasetConfig(
    name="ciao",
    data_dir="data/ciao",
    ratings_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
        "https://raw.githubusercontent.com/daicoolb/RecommenderSystem-DataSet/master/CiaoDVD/CiaoDVD.zip",
    ],
    trust_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
        "https://raw.githubusercontent.com/daicoolb/RecommenderSystem-DataSet/master/CiaoDVD/CiaoDVD.zip",
    ],
    ratings_filenames=["movie-ratings.txt", "ratings.txt"],
    trust_filenames=["trusts.txt"],
    delimiter="auto",
    explicit_rating_col_index=4,
    k_core=5,
    feedback_mode="threshold_binarize",
    rating_threshold=3.0,
    test_ratio=0.2,
    seed=42,
)

YELP_CONFIG = DatasetConfig(
    name="yelp",
    data_dir="data/yelp",
    ratings_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
    trust_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
    ratings_filenames=["ratings.txt", "train.txt"],
    trust_filenames=["trusts.txt", "trust.txt", "trustnetwork.txt", "links.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
)

FILMTRUST_CONFIG = DatasetConfig(
    name="filmtrust",
    data_dir="data/filmtrust",
    ratings_urls=[
        "https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/ratings.txt",
    ],
    trust_urls=[
        "https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/trust.txt",
    ],
    ratings_filenames=["ratings.txt"],
    trust_filenames=["trust.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
)

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "ciao": CIAO_CONFIG,
    "yelp": YELP_CONFIG,
    "filmtrust": FILMTRUST_CONFIG,
}
```

- [ ] **Step 2: Verify**

Run:
```bash
python -c "
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY

assert set(DATASET_REGISTRY.keys()) == {'ciao', 'yelp', 'filmtrust'}, DATASET_REGISTRY.keys()

ciao = DATASET_REGISTRY['ciao']
assert ciao.k_core == 5
assert ciao.feedback_mode == 'threshold_binarize'
assert ciao.rating_threshold == 3.0
assert ciao.explicit_rating_col_index == 4

yelp = DATASET_REGISTRY['yelp']
assert yelp.k_core is None
assert yelp.feedback_mode == 'explicit'

filmtrust = DATASET_REGISTRY['filmtrust']
assert filmtrust.k_core is None
assert filmtrust.feedback_mode == 'explicit'
assert filmtrust.ratings_urls[0].endswith('FilmTrust/ratings.txt')
assert filmtrust.trust_urls[0].endswith('FilmTrust/trust.txt')

print('Check 1 (DATASET_REGISTRY contents): PASS')
"
```
Expected output:
```
Check 1 (DATASET_REGISTRY contents): PASS
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/dataset_configs.py
git commit -m "feat(data_loaders): add DatasetConfig and registry for Ciao/Yelp/FilmTrust

Declarative parameters derived from reading the three existing
loaders' actual behavior. Adding a new dataset means adding one
config entry here, not a new loader class."
```

---

### Task 3: `ExplicitTrustLoader` + real-data equivalence verification

**Files:**
- Create: `pipeline/data_loaders/explicit_trust_loader.py`

**Interfaces:**
- Consumes: `ArenaDataset`, `BaseDatasetLoader` from Task 1 (`pipeline.data_loaders.base_loader`); `DatasetConfig` fields from Task 2 (`pipeline.data_loaders.dataset_configs`), used by exact name as documented in Task 2.
- Produces: `ExplicitTrustLoader(BaseDatasetLoader)` with `__init__(self, config: DatasetConfig)` and `load(self) -> ArenaDataset`. Task 4's `DatasetFactory` instantiates this class directly.

This is the core consolidation: one generic implementation that reproduces the
behavior of `AcademicDataLoader` (Ciao), `YelpDataLoader`, and `FilmTrustLoader`
via configuration alone.

- [ ] **Step 1: Write `pipeline/data_loaders/explicit_trust_loader.py`**

```python
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
```

- [ ] **Step 2: Run the real-data equivalence verification against all 3 existing loaders**

This is the key correctness check for the whole consolidation: confirm the new
generic loader reproduces the existing loaders' behavior on the SAME real,
already-downloaded data. `data/ciao/` and `data/yelp/` already exist from prior
sessions; `data/filmtrust/` will be freshly downloaded by this script (small,
fast, proven reliable in a prior sub-project).

Run:
```bash
python -c "
import numpy as np
from pipeline.data_loaders.dataset_configs import CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

# ---- Ciao ----
print('='*70); print('CIAO: new ExplicitTrustLoader vs old AcademicDataLoader'); print('='*70)
from pipeline.unified_arena.academic_data_loader import AcademicDataLoader

new_ciao = ExplicitTrustLoader(CIAO_CONFIG).load()
old_ciao = AcademicDataLoader(data_dir='data/ciao').load()

print(f'new: n_raw={new_ciao.n_raw_interactions}, num_items={new_ciao.num_items}, '
      f'num_users={new_ciao.num_users}, train={new_ciao.n_train_interactions}, '
      f'test={new_ciao.n_test_interactions}, social_nnz={new_ciao.social_csr.nnz}')
print(f'old: n_raw={old_ciao.n_raw_interactions}, num_items={old_ciao.num_items}, '
      f'num_users={old_ciao.num_users}, train={old_ciao.n_train_interactions}, '
      f'test={old_ciao.n_test_interactions}, social_nnz={old_ciao.social_csr.nnz}')

assert new_ciao.n_raw_interactions == old_ciao.n_raw_interactions, 'raw interaction count must match exactly'
assert new_ciao.num_items == old_ciao.num_items, 'item count must match exactly (trust file never references items)'
assert new_ciao.n_train_interactions == old_ciao.n_train_interactions, 'train split size must match exactly'
assert new_ciao.n_test_interactions == old_ciao.n_test_interactions, 'test split size must match exactly'
assert new_ciao.num_users >= old_ciao.num_users, (
    f'new num_users ({new_ciao.num_users}) should be >= old ({old_ciao.num_users}) '
    f'since the new loader additionally includes trust-only users'
)
print(f'Check (Ciao): PASS -- num_users delta = {new_ciao.num_users - old_ciao.num_users} '
      f'(trust-only users included by the new loader\\'s union-of-users convention)')

# ---- Yelp ----
print('='*70); print('YELP: new ExplicitTrustLoader vs old YelpDataLoader'); print('='*70)
from pipeline.academic_sandbox.yelp_data_loader import YelpDataLoader

new_yelp = ExplicitTrustLoader(YELP_CONFIG).load()
old_yelp_loader = YelpDataLoader(data_dir='data/yelp')
old_yelp_loader.download()
old_yelp_loader.load_data()
old_yelp_train = old_yelp_loader.get_train_interaction_matrix()
old_yelp_trust = old_yelp_loader.get_trust_matrix()
old_yelp_test = old_yelp_loader.get_test_dict()

print(f'new: num_users={new_yelp.num_users}, num_items={new_yelp.num_items}, '
      f'train_nnz={new_yelp.train_csr.nnz}, social_nnz={new_yelp.social_csr.nnz}, '
      f'test_users={len(new_yelp.test_dict)}')
print(f'old: num_users={old_yelp_loader.num_users}, num_items={old_yelp_loader.num_items}, '
      f'train_nnz={old_yelp_train.nnz}, social_nnz={old_yelp_trust.nnz}, '
      f'test_users={len(old_yelp_test)}')

assert new_yelp.num_users == old_yelp_loader.num_users, 'Yelp num_users should match exactly (old loader already unions users)'
assert new_yelp.num_items == old_yelp_loader.num_items, 'Yelp num_items should match exactly'
print('Check (Yelp): PASS -- num_users/num_items match exactly as expected')

# ---- FilmTrust ----
print('='*70); print('FILMTRUST: new ExplicitTrustLoader vs old FilmTrustLoader'); print('='*70)
from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader

new_filmtrust = ExplicitTrustLoader(FILMTRUST_CONFIG).load()
old_ft_loader = FilmTrustLoader(data_dir='data/filmtrust')
old_ft_loader.download()
old_ft_loader.load_data()
old_ft_train = old_ft_loader.get_train_interaction_matrix()
old_ft_trust = old_ft_loader.get_trust_matrix()

print(f'new: num_users={new_filmtrust.num_users}, num_items={new_filmtrust.num_items}, '
      f'train_nnz={new_filmtrust.train_csr.nnz}, social_nnz={new_filmtrust.social_csr.nnz}')
print(f'old: num_users={old_ft_loader.num_users}, num_items={old_ft_loader.num_items}, '
      f'train_nnz={old_ft_train.nnz}, social_nnz={old_ft_trust.nnz}')

assert new_filmtrust.num_users == old_ft_loader.num_users, 'FilmTrust num_users should match exactly (old loader already unions users)'
assert new_filmtrust.num_items == old_ft_loader.num_items, 'FilmTrust num_items should match exactly'
print('Check (FilmTrust): PASS -- num_users/num_items match exactly as expected')

print('='*70); print('ALL EQUIVALENCE CHECKS PASSED'); print('='*70)
"
```

Expected output: each of the three sections prints `new:`/`old:` stat lines followed by
a `Check (...): PASS` line, ending in `ALL EQUIVALENCE CHECKS PASSED`, with no
`AssertionError` or traceback. If any assertion fails, this is a real discrepancy in the
generic loader's logic relative to the existing, proven-correct loaders — diagnose by
comparing the failing dataset's specific `DatasetConfig` fields against that dataset's
existing loader's actual behavior (re-read the relevant existing loader file) before
changing anything.

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/explicit_trust_loader.py
git commit -m "feat(data_loaders): add ExplicitTrustLoader, consolidating Ciao/Yelp/FilmTrust parsing

One generic, config-driven implementation verified against all three
existing loaders' real output on real downloaded data. The existing
loaders themselves are untouched."
```

---

### Task 4: `DatasetFactory`

**Files:**
- Create: `pipeline/data_loaders/dataset_factory.py`

**Interfaces:**
- Consumes: `BaseDatasetLoader` (Task 1), `DATASET_REGISTRY` (Task 2), `ExplicitTrustLoader` (Task 3).
- Produces: `DatasetFactory.create(name: str) -> BaseDatasetLoader` (static method), raising `ValueError` for an unregistered name.

- [ ] **Step 1: Write `pipeline/data_loaders/dataset_factory.py`**

```python
"""
Dataset Factory -- Factory Method entry point for the Grand Unified Benchmark Arena's
dataset loading system.

DatasetFactory.create(name) looks up a DatasetConfig in dataset_configs.DATASET_REGISTRY
and returns a configured BaseDatasetLoader. Adding a new dataset means registering a
new DatasetConfig, not modifying this file or writing a new loader class (Open for
extension, Closed for modification).
"""
from __future__ import annotations

from pipeline.data_loaders.base_loader import BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader


class DatasetFactory:
    """Factory Method for constructing dataset loaders by name."""

    @staticmethod
    def create(name: str) -> BaseDatasetLoader:
        """
        Look up `name` (case-insensitive) in the dataset registry and return a
        configured loader ready to call .load() on.

        Raises:
            ValueError: if `name` is not a registered dataset.
        """
        key = name.lower()
        if key not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

        config = DATASET_REGISTRY[key]
        return ExplicitTrustLoader(config)
```

- [ ] **Step 2: Verify**

Run:
```bash
python -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

for name in ['ciao', 'yelp', 'filmtrust', 'CIAO', 'Yelp']:
    loader = DatasetFactory.create(name)
    assert isinstance(loader, ExplicitTrustLoader), f'{name} did not return an ExplicitTrustLoader'
print('Check 1 (DatasetFactory.create returns correct loader type, case-insensitive): PASS')

try:
    DatasetFactory.create('nonexistent_dataset')
    print('Check 2 (unknown dataset raises ValueError): FAIL -- no exception raised')
except ValueError as e:
    msg = str(e)
    assert 'nonexistent_dataset' in msg
    assert 'ciao' in msg and 'yelp' in msg and 'filmtrust' in msg
    print('Check 2 (unknown dataset raises ValueError with helpful message): PASS')
"
```
Expected output:
```
Check 1 (DatasetFactory.create returns correct loader type, case-insensitive): PASS
Check 2 (unknown dataset raises ValueError with helpful message): PASS
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/dataset_factory.py
git commit -m "feat(data_loaders): add DatasetFactory.create() factory method

Looks up the dataset registry and returns a configured loader.
Adding a new dataset is a config registry entry, not a code change."
```

---

### Task 5: Delete confirmed-dead legacy code

**Files:**
- Delete: `pipeline/engines/academic_benchmark_arena.py`
- Delete: `pipeline/engines/academic_data_loader.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-4. This task is independent and could be done in any order relative to them; placed last here only to keep the plan's "new feature, then cleanup" narrative simple.

These two files are an early, fully-superseded Ciao/Epinions sandbox, confirmed dead
during design (the only reference to `pipeline.engines.academic_data_loader` is
`academic_benchmark_arena.py`'s own import; `academic_benchmark_arena` itself is
referenced nowhere else in the repository). They are distinct from the *working*
`pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/`
packages, which this plan does NOT touch.

- [ ] **Step 1: Re-confirm dead-code status before deleting**

Run:
```bash
grep -rn "academic_benchmark_arena\|pipeline\.engines\.academic_data_loader" --include="*.py" pipeline/ app/ evaluation/
```
Expected output: only matches inside `pipeline/engines/academic_benchmark_arena.py`
itself (its own module-level import line). No matches in any other `.py` file. If
anything else references either module, STOP and report back — do not delete.

- [ ] **Step 2: Delete the two files**

```bash
git rm pipeline/engines/academic_benchmark_arena.py pipeline/engines/academic_data_loader.py
```

- [ ] **Step 3: Verify nothing broke**

Run:
```bash
python -c "
import importlib

try:
    importlib.import_module('pipeline.engines.academic_benchmark_arena')
    print('Check 1 (legacy module removed): FAIL -- module still importable')
except ModuleNotFoundError:
    print('Check 1 (legacy module removed): PASS')

# Confirm the existing, untouched arena scripts still import cleanly (no broken
# cross-imports were introduced by this deletion).
from pipeline.unified_arena.run_arena import run_arena
from pipeline.academic_sandbox.run_yelp_benchmark import run_yelp_benchmark
from pipeline.filmtrust_arena.run_filmtrust import run_social_arena
assert callable(run_arena) and callable(run_yelp_benchmark) and callable(run_social_arena)
print('Check 2 (existing untouched arena scripts still import cleanly): PASS')
"
```
Expected output:
```
Check 1 (legacy module removed): PASS
Check 2 (existing untouched arena scripts still import cleanly): PASS
```

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: delete dead legacy Ciao/Epinions sandbox

pipeline/engines/academic_benchmark_arena.py and academic_data_loader.py
were fully superseded by pipeline/unified_arena/ and referenced by
nothing else in the repository (confirmed via grep before deletion)."
```
