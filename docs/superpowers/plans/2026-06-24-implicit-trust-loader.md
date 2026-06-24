# ImplicitTrustLoader (Mode B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ImplicitTrustLoader`, a `BaseDatasetLoader` implementation for datasets with no real social network (MovieLens-100K for now), which synthesizes its trust graph via the OOM-safe sparse Jaccard engine instead of downloading a real trust file, and is unmistakably labeled as an ablation study (Mode B) in both data and console output.

**Architecture:** Extract the ~150 lines of dataset-agnostic helpers currently private to `ExplicitTrustLoader` (download-with-fallback, generic row parsing, k-core filtering, stratified split, interaction-matrix construction) into a new shared module, `pipeline/data_loaders/loader_utils.py`, with no behavior change. `ImplicitTrustLoader` reuses those helpers and adds only what's genuinely new: calling `pipeline/utils/sparse_jaccard.py::compute_sparse_jaccard_trust` on `train_csr` instead of downloading a trust file. A new `ImplicitDatasetConfig` dataclass and `IMPLICIT_DATASET_REGISTRY` dict parallel the existing `DatasetConfig`/`DATASET_REGISTRY`, and `DatasetFactory.create()` checks both registries to decide which loader class to instantiate — registry membership is the dispatch signal, not a `mode` field on the config.

**Tech Stack:** Python, `scipy.sparse`, `numpy`, `pandas`, stdlib `zipfile`/`io`/`urllib.request`/`gc` (no new dependencies).

## Global Constraints

- No new third-party dependencies.
- `pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/` and their CLI runners MUST remain untouched (unaffected by this plan — nothing in this plan touches them).
- `pipeline/engines/unified_data_loader.py` MUST remain untouched (frozen, production dependency). This plan's `ImplicitTrustLoader` is a new, parallel implementation that deliberately differs from it (computes Jaccard trust from train-only interactions, not the full pre-split dataset — see Task 3).
- `pipeline/data_loaders/explicit_trust_loader.py` (shipped in a prior sub-project) IS modified by this plan (Task 1) — this is a sanctioned refactor (pure code-move into `loader_utils.py`, no behavior change), not a frozen file. Verified via re-running the prior sub-project's exact real-data equivalence script and confirming identical output.
- Dataset scope for this plan: **MovieLens-100K only**. ML-1M, ML-10M, and Jester are explicitly deferred (different raw formats, source URLs not yet verified) — do not add configs or loader code for them in this plan.
- Jaccard trust is computed from `train_csr` only, never the full pre-split interactions — this is a deliberate, binding decision (avoids leaking test-set co-occurrence into the synthetic trust side-channel).
- `ImplicitDatasetConfig` and `IMPLICIT_DATASET_REGISTRY` are separate from `DatasetConfig`/`DATASET_REGISTRY` — do not add a `mode` field to either config to drive `DatasetFactory` dispatch; registry membership alone is the signal.
- This codebase has no pytest framework — verification is direct script execution with documented exact expected output, run against real, already-downloaded data (`data/ml-100k/u.data`, confirmed locally present: 943 users, 1682 items, 100,000 ratings, per `data/ml-100k/u.info`) or real, already-downloaded `data/ciao/`, `data/yelp/`, `data/filmtrust/` (Task 1's regression check).
- `compute_sparse_jaccard_trust` (in `pipeline/utils/sparse_jaccard.py`) already calls `gc.collect()` internally per chunk and at the end. `ImplicitTrustLoader.load()` additionally calls `gc.collect()` itself immediately after receiving the result, per an explicit requirement — this is intentionally redundant/defensive, not a sign the library call is insufficient.

---

### Task 1: Extract `loader_utils.py`, refactor `ExplicitTrustLoader` (no behavior change)

**Files:**
- Create: `pipeline/data_loaders/loader_utils.py`
- Modify: `pipeline/data_loaders/explicit_trust_loader.py` (entire file rewritten to delegate to `loader_utils`)

**Interfaces:**
- Produces (consumed by Task 3's `ImplicitTrustLoader` and by the refactored `ExplicitTrustLoader` in this same task):
  - `download_with_fallback(urls: List[str], data_dir: str, dataset_name: str, loader_label: str) -> None`
  - `files_exist(data_dir: str, filenames: List[str]) -> bool`
  - `resolve_path(data_dir: str, filenames: List[str]) -> str`
  - `parse_rows(path: str, delimiter: str, explicit_rating_col_index: int = 2, col_names: Tuple[str, str, str] = ("user", "item", "rating")) -> pd.DataFrame`
  - `split_line(line: str, delimiter: str) -> List[str]`
  - `k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]`
  - `stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]`
  - `build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, binarize: bool) -> sp.csr_matrix`
  - `build_dict(df: pd.DataFrame) -> Dict[int, Set[int]]`

- [ ] **Step 1: Write `pipeline/data_loaders/loader_utils.py`**

```python
"""
Loader Utils -- Shared, dataset-agnostic helpers for BaseDatasetLoader implementations.

Extracted from pipeline/data_loaders/explicit_trust_loader.py (sub-project 2) with no
behavior change, so that pipeline/data_loaders/implicit_trust_loader.py (sub-project 3)
can reuse the same download/parse/filter/split/matrix-construction logic instead of
duplicating ~150 lines of it. Both ExplicitTrustLoader and ImplicitTrustLoader call
into this module; neither of them owns it.
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

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------
def download_with_fallback(urls: List[str], data_dir: str, dataset_name: str, loader_label: str) -> None:
    """
    Try each URL in order until one succeeds. If the response is a zip, extract it
    into data_dir; otherwise save it as a raw file named from the URL. Does not raise
    on total failure -- callers must check files_exist() afterward and raise their own
    dataset-specific error (different loaders need different files to be present).
    """
    os.makedirs(data_dir, exist_ok=True)
    for url in urls:
        try:
            print(f"  [{loader_label}:{dataset_name}] Downloading from {url} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            print(f"  [{loader_label}:{dataset_name}] Failed to fetch {url}: {e}", flush=True)
            continue

        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(data_dir)
            print(f"  [{loader_label}:{dataset_name}] Extracted zip ({len(data)/1024/1024:.1f} MB) -> {data_dir}", flush=True)
        else:
            dest_name = _guess_filename_for_url(url)
            dest_path = os.path.join(data_dir, dest_name)
            with open(dest_path, "wb") as f:
                f.write(data)
            print(f"  [{loader_label}:{dataset_name}] Saved {len(data):,} bytes -> {dest_path}", flush=True)


def _guess_filename_for_url(url: str) -> str:
    """Pick a destination filename for a raw (non-zip) download from its URL."""
    basename = url.rstrip("/").split("/")[-1].split("?")[0]
    return basename if basename else "downloaded_file.txt"


def files_exist(data_dir: str, filenames: List[str]) -> bool:
    """True if any of the candidate filenames (case-insensitive) exist anywhere under data_dir."""
    lower_targets = {f.lower() for f in filenames}
    for root, _, files in os.walk(data_dir):
        lower_files = {f.lower() for f in files}
        if lower_targets & lower_files:
            return True
    return False


def resolve_path(data_dir: str, filenames: List[str]) -> str:
    """Find the first file under data_dir matching one of the candidate filenames (case-insensitive)."""
    lower_targets = {f.lower() for f in filenames}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() in lower_targets:
                return os.path.join(root, f)
    raise FileNotFoundError(f"No file found under {data_dir} matching {filenames}")


# ------------------------------------------------------------------
# Parsing (shared generic parser for ratings/trust rows)
# ------------------------------------------------------------------
def parse_rows(
    path: str,
    delimiter: str,
    explicit_rating_col_index: int = 2,
    col_names: Tuple[str, str, str] = ("user", "item", "rating"),
) -> pd.DataFrame:
    """
    Generic row parser. Branches on column count after delimiter-splitting:
      >=5 columns -> rating/weight at explicit_rating_col_index (Ciao's categoryId/
                     reviewId layout)
      >=3 columns -> rating/weight at column index 2 (generic "user item rating[...]")
      ==2 columns -> implicit, weight defaults to 1.0
    """
    rows: List[Tuple[str, str, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            parts = split_line(line, delimiter)

            if len(parts) >= 5:
                idx = explicit_rating_col_index
                if idx < len(parts):
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[idx].strip())))
            elif len(parts) >= 3:
                rows.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
            elif len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip(), 1.0))

    return pd.DataFrame(rows, columns=list(col_names))


def split_line(line: str, delimiter: str) -> List[str]:
    if delimiter == "comma":
        return line.split(",")
    if delimiter == "space":
        return line.split()
    # "auto": try comma first (Ciao's existing heuristic), else whitespace
    if "," in line:
        return line.split(",")
    return line.split()


# ------------------------------------------------------------------
# k-core filtering
# ------------------------------------------------------------------
def k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]:
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
def stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user leave-N-out: hold out test_ratio of each user's items. Requires df to have a u_idx column."""
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
def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, binarize: bool) -> sp.csr_matrix:
    """Requires df to have u_idx/i_idx columns (and a rating column if binarize=False)."""
    rows = df["u_idx"].values.astype(np.int64)
    cols = df["i_idx"].values.astype(np.int64)

    if binarize:
        vals = np.ones(len(rows), dtype=np.float32)
    else:
        vals = df["rating"].values.astype(np.float32)

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    if binarize:
        mat.data[:] = 1.0
    return mat


def build_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Requires df to have u_idx/i_idx columns."""
    result: Dict[int, Set[int]] = {}
    for u, i in zip(df["u_idx"].values, df["i_idx"].values):
        result.setdefault(int(u), set()).add(int(i))
    return result
```

- [ ] **Step 2: Rewrite `pipeline/data_loaders/explicit_trust_loader.py` to delegate to `loader_utils` (no behavior change)**

Replace the entire file contents with:

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
            raise RuntimeError(
                f"Could not obtain a usable ratings/trust file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place files named one of "
                f"{cfg.ratings_filenames} (ratings) and {cfg.trust_filenames} (trust) "
                f"directly into {cfg.data_dir}."
            )

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
```

- [ ] **Step 3: Run the regression check -- re-verify the exact real-data equivalence script from sub-project 2 still passes unchanged**

This proves the extraction in Steps 1-2 changed nothing observable.

Run:
```bash
PYTHONPATH=. py -3 -c "
import numpy as np
from pipeline.data_loaders.dataset_configs import CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

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

assert new_ciao.n_raw_interactions == old_ciao.n_raw_interactions
assert new_ciao.num_items == old_ciao.num_items
assert new_ciao.n_train_interactions == old_ciao.n_train_interactions
assert new_ciao.n_test_interactions == old_ciao.n_test_interactions
assert new_ciao.num_users >= old_ciao.num_users
print(f'Check (Ciao): PASS -- num_users delta = {new_ciao.num_users - old_ciao.num_users}')

print('='*70); print('YELP: new ExplicitTrustLoader vs old YelpDataLoader'); print('='*70)
from pipeline.academic_sandbox.yelp_data_loader import YelpDataLoader

new_yelp = ExplicitTrustLoader(YELP_CONFIG).load()
old_yelp_loader = YelpDataLoader(data_dir='data/yelp')
old_yelp_loader.download()
old_yelp_loader.load_data()
old_yelp_train = old_yelp_loader.get_train_interaction_matrix()
old_yelp_trust = old_yelp_loader.get_trust_matrix()

print(f'new: num_users={new_yelp.num_users}, num_items={new_yelp.num_items}, '
      f'train_nnz={new_yelp.train_csr.nnz}, social_nnz={new_yelp.social_csr.nnz}')
print(f'old: num_users={old_yelp_loader.num_users}, num_items={old_yelp_loader.num_items}, '
      f'train_nnz={old_yelp_train.nnz}, social_nnz={old_yelp_trust.nnz}')

assert new_yelp.num_users == old_yelp_loader.num_users
assert new_yelp.num_items == old_yelp_loader.num_items
print('Check (Yelp): PASS')

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

assert new_filmtrust.num_users == old_ft_loader.num_users
assert new_filmtrust.num_items == old_ft_loader.num_items
print('Check (FilmTrust): PASS')

print('='*70); print('ALL EQUIVALENCE CHECKS PASSED (post-refactor regression check)'); print('='*70)
"
```

Expected output: identical in shape to sub-project 2's original run -- three `Check (...): PASS` lines, ending in `ALL EQUIVALENCE CHECKS PASSED (post-refactor regression check)`, no `AssertionError` or traceback. `data/ciao/` and `data/yelp/` should already be present locally; `data/filmtrust/` will download fresh if not already present (small, fast, proven reliable).

- [ ] **Step 4: Commit**

```bash
git add pipeline/data_loaders/loader_utils.py pipeline/data_loaders/explicit_trust_loader.py
git commit -m "refactor(data_loaders): extract shared loader_utils from ExplicitTrustLoader

Pure code-move, no behavior change -- verified by re-running the
real-data equivalence check from the prior sub-project unchanged.
Makes the dataset-agnostic download/parse/filter/split/matrix logic
reusable by the upcoming ImplicitTrustLoader instead of duplicating it."
```

---

### Task 2: `ImplicitDatasetConfig` + ML-100K registry entry

**Files:**
- Modify: `pipeline/data_loaders/dataset_configs.py` (append new dataclass + config + registry; existing `DatasetConfig`/`DATASET_REGISTRY` content is untouched)

**Interfaces:**
- Consumes: nothing from Task 1 (pure data, no imports from `loader_utils` or either loader class).
- Produces: `ImplicitDatasetConfig` dataclass (`name: str`, `data_dir: str`, `ratings_urls: List[str]`, `ratings_filenames: List[str]`, `delimiter: str = "space"`, `rating_col_index: int = 2`, `k_core: Optional[int] = None`, `jaccard_threshold: float = 0.3`, `jaccard_top_k: Optional[int] = 50`, `jaccard_chunk_size: int = 2000`, `test_ratio: float = 0.2`, `seed: int = 42`); `ML_100K_CONFIG` instance; `IMPLICIT_DATASET_REGISTRY: Dict[str, ImplicitDatasetConfig]` with key `"ml-100k"`. Task 3's `ImplicitTrustLoader` consumes these field names exactly. Task 4's `DatasetFactory` consumes `IMPLICIT_DATASET_REGISTRY` by name.

- [ ] **Step 1: Append to `pipeline/data_loaders/dataset_configs.py`**

Add this content to the end of the existing file (after the current `DATASET_REGISTRY` definition; do not modify anything above it):

```python


@dataclass
class ImplicitDatasetConfig:
    """
    Declarative configuration for ImplicitTrustLoader (Mode B / ablation study).

    Unlike DatasetConfig, there is no trust_urls/trust_filenames -- trust is
    synthesized via Jaccard similarity (pipeline/utils/sparse_jaccard.py), not
    downloaded. This is intentionally a separate dataclass from DatasetConfig rather
    than an extension of it, to avoid either type carrying fields that are always
    irrelevant/None for the other's use case.

    Fields:
        name, data_dir: same meaning as DatasetConfig.
        ratings_urls, ratings_filenames: same meaning as DatasetConfig.
        delimiter: "space" handles ML-100K's tab-delimited u.data correctly, since
            Python's str.split() with no argument splits on any whitespace run
            (including tabs) -- the same "space" mode DatasetConfig already uses.
        rating_col_index: column index of the rating value in a parsed row.
            ML-100K's "user item rating timestamp" layout puts it at index 2.
        k_core: minimum interactions per user/item. None for ML-100K -- GroupLens
            already guarantees every user has >=20 ratings.
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

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_configs import IMPLICIT_DATASET_REGISTRY, ML_100K_CONFIG

assert set(IMPLICIT_DATASET_REGISTRY.keys()) == {'ml-100k'}
assert ML_100K_CONFIG.k_core is None
assert ML_100K_CONFIG.jaccard_threshold == 0.3
assert ML_100K_CONFIG.jaccard_top_k == 50
assert ML_100K_CONFIG.ratings_urls[0] == 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
assert ML_100K_CONFIG.ratings_filenames == ['u.data']
assert ML_100K_CONFIG.delimiter == 'space'

# Confirm the existing explicit registry/config from sub-project 2 are unaffected.
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY
assert set(DATASET_REGISTRY.keys()) == {'ciao', 'yelp', 'filmtrust'}

print('Check 1 (IMPLICIT_DATASET_REGISTRY contents, DATASET_REGISTRY unaffected): PASS')
"
```
Expected output:
```
Check 1 (IMPLICIT_DATASET_REGISTRY contents, DATASET_REGISTRY unaffected): PASS
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/dataset_configs.py
git commit -m "feat(data_loaders): add ImplicitDatasetConfig and ML-100K registry entry

Separate dataclass/registry from DatasetConfig -- implicit datasets
need jaccard hyperparameters, not trust_urls/feedback_mode, and have
no real trust file to download. ML-1M/10M/Jester deferred until their
real source URLs/formats are verified (different raw formats)."
```

---

### Task 3: `ImplicitTrustLoader`

**Files:**
- Create: `pipeline/data_loaders/implicit_trust_loader.py`

**Interfaces:**
- Consumes: `ArenaDataset`, `BaseDatasetLoader` (Task 1's untouched `base_loader.py`); `loader_utils` functions exactly as named in Task 1's Interfaces block; `ImplicitDatasetConfig` fields exactly as named in Task 2; `compute_sparse_jaccard_trust(interaction_matrix: sp.csr_matrix, threshold: float = 0.3, top_k: Optional[int] = 50, chunk_size: int = 2000, dtype: np.dtype = np.float32) -> sp.csr_matrix` from `pipeline/utils/sparse_jaccard.py` (sub-project 1, unmodified).
- Produces: `ImplicitTrustLoader(BaseDatasetLoader)` with `__init__(self, config: ImplicitDatasetConfig)` and `load(self) -> ArenaDataset`. Task 4's `DatasetFactory` instantiates this class directly.

- [ ] **Step 1: Write `pipeline/data_loaders/implicit_trust_loader.py`**

```python
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
```

- [ ] **Step 2: Run the real-data verification against ML-100K**

`data/ml-100k/u.data` already exists locally (943 users, 1682 items, 100,000 ratings, confirmed via `data/ml-100k/u.info`), so this exercises the "already present" download-skip path, not a fresh download.

Run:
```bash
PYTHONPATH=. py -3 -c "
import numpy as np
from pipeline.data_loaders.dataset_configs import ML_100K_CONFIG
from pipeline.data_loaders.implicit_trust_loader import ImplicitTrustLoader

ds = ImplicitTrustLoader(ML_100K_CONFIG).load()

assert ds.mode == 'implicit', f'expected mode=implicit, got {ds.mode}'
assert ds.num_users == 943, f'expected 943 users, got {ds.num_users}'
assert ds.num_items == 1682, f'expected 1682 items, got {ds.num_items}'
assert ds.n_raw_interactions == 100000, f'expected 100000 raw interactions, got {ds.n_raw_interactions}'
print('Check 1 (shape matches known ML-100K dimensions, mode=implicit): PASS')

# train_csr must hold explicit rating values (1-5), not binary 0/1
unique_vals = set(np.unique(ds.train_csr.data).round(2).tolist())
assert unique_vals <= {1.0, 2.0, 3.0, 4.0, 5.0}, f'unexpected train_csr values: {unique_vals}'
assert len(unique_vals) > 1, 'train_csr should hold varied explicit ratings, not a single binary value'
print(f'Check 2 (train_csr holds explicit ratings, values={sorted(unique_vals)}): PASS')

# social_csr must be symmetric, zero-diagonal, and bounded by the documented Jaccard memory bound
assert (ds.social_csr != ds.social_csr.T).nnz == 0, 'social_csr must be symmetric'
assert ds.social_csr.diagonal().sum() == 0, 'social_csr must have zero diagonal'
bound = 2 * ML_100K_CONFIG.jaccard_top_k * ds.num_users
assert ds.social_csr.nnz <= bound, f'social_csr.nnz={ds.social_csr.nnz} exceeds documented bound {bound}'
print(f'Check 3 (social_csr symmetric, zero-diagonal, nnz={ds.social_csr.nnz} <= bound {bound}): PASS')

# get_sym_adj_mat() must still work against this loader's output
adj = ds.get_sym_adj_mat()
assert adj.shape == (ds.num_users + ds.num_items, ds.num_users + ds.num_items)
print('Check 4 (get_sym_adj_mat works against ImplicitTrustLoader output): PASS')
"
```
Expected output: the loader's own console banner/progress lines (starting with the
`ABLATION STUDY ONLY -- Mode B` banner), followed by:
```
Check 1 (shape matches known ML-100K dimensions, mode=implicit): PASS
Check 2 (train_csr holds explicit ratings, values=[1.0, 2.0, 3.0, 4.0, 5.0]): PASS
Check 3 (social_csr symmetric, zero-diagonal, nnz=<some number> <= bound 94300): PASS
Check 4 (get_sym_adj_mat works against ImplicitTrustLoader output): PASS
```
(The exact `nnz` number in Check 3 will vary slightly depending on the real Jaccard
similarity distribution in ML-100K at `threshold=0.3` -- the check only requires it stay
under the documented `2 * top_k * num_users` bound, not an exact value.)

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/implicit_trust_loader.py
git commit -m "feat(data_loaders): add ImplicitTrustLoader (Mode B ablation study)

Synthesizes trust via compute_sparse_jaccard_trust on train_csr only
(avoids test-set leakage through the trust side-channel). Verified
against real ML-100K data: correct shape, explicit ratings preserved,
synthetic social_csr symmetric and within the documented memory bound."
```

---

### Task 4: `DatasetFactory` dual-registry dispatch

**Files:**
- Modify: `pipeline/data_loaders/dataset_factory.py`

**Interfaces:**
- Consumes: `DATASET_REGISTRY` (existing), `IMPLICIT_DATASET_REGISTRY` (Task 2), `ExplicitTrustLoader` (existing), `ImplicitTrustLoader` (Task 3).
- Produces: `DatasetFactory.create(name: str) -> BaseDatasetLoader` (unchanged signature), now dispatching to either loader class depending on which registry contains `name`.

- [ ] **Step 1: Rewrite `pipeline/data_loaders/dataset_factory.py`**

Replace the entire file contents with:

```python
"""
Dataset Factory -- Factory Method entry point for the Grand Unified Benchmark Arena's
dataset loading system.

DatasetFactory.create(name) checks the explicit-trust registry first, then the
implicit-trust (Mode B / ablation study) registry, and returns a configured loader of
the matching type. Adding a new dataset means registering a new config in the
appropriate registry -- not modifying this file or writing a new loader class, for
either Mode A or Mode B (Open for extension, Closed for modification). Registry
membership alone is the dispatch signal; neither config type carries a "mode" field
for this purpose.
"""
from __future__ import annotations

from pipeline.data_loaders.base_loader import BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY, IMPLICIT_DATASET_REGISTRY
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader
from pipeline.data_loaders.implicit_trust_loader import ImplicitTrustLoader


class DatasetFactory:
    """Factory Method for constructing dataset loaders by name."""

    @staticmethod
    def create(name: str) -> BaseDatasetLoader:
        """
        Look up `name` (case-insensitive) in the explicit-trust registry first, then
        the implicit-trust (Mode B) registry, and return a configured loader of the
        matching type, ready to call .load() on.

        Raises:
            ValueError: if `name` is not registered in either registry.
        """
        key = name.lower()

        if key in DATASET_REGISTRY:
            return ExplicitTrustLoader(DATASET_REGISTRY[key])

        if key in IMPLICIT_DATASET_REGISTRY:
            return ImplicitTrustLoader(IMPLICIT_DATASET_REGISTRY[key])

        available = ", ".join(sorted(list(DATASET_REGISTRY.keys()) + list(IMPLICIT_DATASET_REGISTRY.keys())))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
```

- [ ] **Step 2: Verify**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader
from pipeline.data_loaders.implicit_trust_loader import ImplicitTrustLoader

# Explicit datasets still dispatch correctly (no regression from sub-project 2)
for name in ['ciao', 'yelp', 'filmtrust', 'CIAO']:
    loader = DatasetFactory.create(name)
    assert isinstance(loader, ExplicitTrustLoader), f'{name} did not return an ExplicitTrustLoader'
print('Check 1 (explicit datasets still dispatch to ExplicitTrustLoader): PASS')

# New implicit dataset dispatches to the new loader type
for name in ['ml-100k', 'ML-100K']:
    loader = DatasetFactory.create(name)
    assert isinstance(loader, ImplicitTrustLoader), f'{name} did not return an ImplicitTrustLoader'
print('Check 2 (ml-100k dispatches to ImplicitTrustLoader, case-insensitive): PASS')

try:
    DatasetFactory.create('nonexistent_dataset')
    print('Check 3 (unknown dataset raises ValueError): FAIL -- no exception raised')
except ValueError as e:
    msg = str(e)
    assert 'nonexistent_dataset' in msg
    for expected_name in ['ciao', 'yelp', 'filmtrust', 'ml-100k']:
        assert expected_name in msg, f'{expected_name} missing from error message: {msg}'
    print('Check 3 (unknown dataset raises ValueError listing all 4 datasets): PASS')
"
```
Expected output:
```
Check 1 (explicit datasets still dispatch to ExplicitTrustLoader): PASS
Check 2 (ml-100k dispatches to ImplicitTrustLoader, case-insensitive): PASS
Check 3 (unknown dataset raises ValueError listing all 4 datasets): PASS
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/dataset_factory.py
git commit -m "feat(data_loaders): dual-registry dispatch in DatasetFactory

create() now checks the implicit (Mode B) registry alongside the
explicit one and returns the matching loader type. Registry
membership is the dispatch signal -- no mode field needed on either
config."
```
