# Epinions / Douban Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a fully-verified `EPINIONS_CONFIG` (real, working mirror found during design research) and a manual-download-only `DOUBAN_CONFIG` in `DATASET_REGISTRY`, backed by a new `ManualDownloadRequiredError` exception shared by both loaders, plus a defensive distrust-edge filter for datasets (Epinions) where negative trust weights are a documented possibility.

**Architecture:** Two new, defaulted fields on the existing `DatasetConfig` (`manual_download_instructions: str = ""`, `filter_negative_trust: bool = False`) keep Ciao/Yelp/FilmTrust's behavior unchanged while enabling the new datasets. `ManualDownloadRequiredError` (in `loader_utils.py`, subclasses `RuntimeError`) replaces today's generic `RuntimeError` in both `ExplicitTrustLoader._ensure_downloaded` and `ImplicitTrustLoader._ensure_downloaded` — no new branching is needed, since a config with no URLs (Douban) already falls through the existing download-then-check flow into the failure path. Flixster is NOT implemented in this plan (no working URL or confirmed format found during design research — deferred).

**Tech Stack:** Python, `scipy.sparse`, `numpy`, `pandas` (no new dependencies).

## Global Constraints

- No new third-party dependencies.
- `pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/` and their CLI runners are unaffected by this plan.
- `pipeline/engines/unified_data_loader.py` is unaffected by this plan.
- `Ciao`/`Yelp`/`FilmTrust`'s observable behavior MUST NOT change — the two new `DatasetConfig` fields default to no-ops (`""` and `False`) for them. Verified by re-running the existing real-data equivalence script (from sub-project 2/3) unchanged.
- The dead `daicoolb/RecommenderSystem-DataSet` fallback URL currently present in `CIAO_CONFIG.ratings_urls`/`.trust_urls` and `YELP_CONFIG.ratings_urls`/`.trust_urls` is removed in this plan (confirmed via direct fetch to return 404 during design research) — their primary, working URLs (`guoguibing.github.io`, the Dropbox share) are untouched.
- Flixster is explicitly OUT OF SCOPE for this plan — do not add a config or any code for it. No working URL or confirmed column format was found during design research.
- `ManualDownloadRequiredError` subclasses `RuntimeError` and is defined once, in `pipeline/data_loaders/loader_utils.py` — both `ExplicitTrustLoader` and `ImplicitTrustLoader` raise the same class, not separate ones.
- The negative-trust filter (`filter_negative_trust`) MUST run immediately after parsing both files and BEFORE the union-of-users ID-mapping block in `ExplicitTrustLoader.load()` — not merely before `_build_social_matrix` — so a distrust-only user is excluded from the user universe entirely, not just from the final edge list.
- `EPINIONS_CONFIG`'s real downloaded `trust_data.txt` must be empirically checked for negative weights during Task 2's verification — report whichever is true (the filter is load-bearing or a no-op safety net), do not assume.
- `DOUBAN_CONFIG`'s `ratings_filenames`/`trust_filenames` (`uir.index`/`social.index`) come from secondary documentation, not a primary file inspection — this is explicitly flagged as unverified in the config's own `manual_download_instructions` text; do not present it as confirmed.
- This codebase has no pytest framework — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.` set (plain `python` is not on PATH in this environment).

---

### Task 1: `ManualDownloadRequiredError`, new `DatasetConfig` fields, dead-URL cleanup

**Files:**
- Modify: `pipeline/data_loaders/loader_utils.py` (add `ManualDownloadRequiredError`)
- Modify: `pipeline/data_loaders/dataset_configs.py` (add two fields to `DatasetConfig`; remove the dead `daicoolb` URL from `CIAO_CONFIG`/`YELP_CONFIG`)
- Modify: `pipeline/data_loaders/explicit_trust_loader.py` (raise the new exception; add the negative-trust filter step)
- Modify: `pipeline/data_loaders/implicit_trust_loader.py` (raise the new exception)

**Interfaces:**
- Produces: `loader_utils.ManualDownloadRequiredError` (subclasses `RuntimeError`, no custom `__init__`, just a documented class). `DatasetConfig.manual_download_instructions: str = ""` and `DatasetConfig.filter_negative_trust: bool = False` — consumed by Task 2's `EPINIONS_CONFIG` and Task 3's `DOUBAN_CONFIG`.

- [ ] **Step 1: Add `ManualDownloadRequiredError` to `pipeline/data_loaders/loader_utils.py`**

Add this class right after the module's imports (before the `_USER_AGENT` constant):

```python
class ManualDownloadRequiredError(RuntimeError):
    """
    Raised when a dataset's files cannot be obtained via automated download --
    either no URLs are configured, or every configured URL failed -- and must be
    placed manually. Subclasses RuntimeError for compatibility with any existing
    generic exception handling. Both ExplicitTrustLoader and ImplicitTrustLoader
    raise this same class so callers only need to catch one exception type.
    """
```

- [ ] **Step 2: Add the two new fields to `DatasetConfig` in `pipeline/data_loaders/dataset_configs.py`**

In the `DatasetConfig` dataclass, add two lines after `seed: int = 42` (the last existing
field), and extend the class docstring's `Fields:` list with their description.

Replace:
```python
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
```

With:
```python
        rating_threshold: only consulted when feedback_mode == "threshold_binarize".
        test_ratio, seed: stratified per-user train/test split parameters.
        manual_download_instructions: extra text folded into ManualDownloadRequiredError's
            message when automated download fails (e.g. a dead-mirror citation and a
            contact for manual access). Empty string (default) falls back to the
            generic "place files named X into Y" message only.
        filter_negative_trust: if True, drop trust rows with weight <= 0 before the
            union-of-users computation and before building social_csr -- for datasets
            where a negative weight encodes explicit distrust (Epinions), as opposed
            to a plain undirected friendship/trust concept with no distrust notion
            (Ciao, Yelp, FilmTrust, Douban). Defaults to False (no-op) for all of them.
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
    manual_download_instructions: str = ""
    filter_negative_trust: bool = False
```

- [ ] **Step 3: Remove the dead `daicoolb` fallback URL from `CIAO_CONFIG` and `YELP_CONFIG`**

Replace:
```python
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
```

With:
```python
CIAO_CONFIG = DatasetConfig(
    name="ciao",
    data_dir="data/ciao",
    ratings_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
    ],
    trust_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
    ],
```

Replace:
```python
YELP_CONFIG = DatasetConfig(
    name="yelp",
    data_dir="data/yelp",
    ratings_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
    trust_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
```

(Yelp's config already only has one URL each -- it never had the dead fallback. No
change needed there beyond confirming this during verification.)

- [ ] **Step 4: Update `ExplicitTrustLoader` to raise `ManualDownloadRequiredError` and filter negative trust**

In `pipeline/data_loaders/explicit_trust_loader.py`, replace the `_ensure_downloaded` method:

```python
    def _ensure_downloaded(self) -> None:
        cfg = self.config
        if lu.files_exist(cfg.data_dir, cfg.ratings_filenames) and lu.files_exist(cfg.data_dir, cfg.trust_filenames):
            print(f"  [ExplicitTrustLoader:{cfg.name}] Dataset files already present in {cfg.data_dir}", flush=True)
            return

        unique_urls = list(dict.fromkeys(cfg.ratings_urls + cfg.trust_urls))
        lu.download_with_fallback(unique_urls, cfg.data_dir, cfg.name, "ExplicitTrustLoader")

        if not (lu.files_exist(cfg.data_dir, cfg.ratings_filenames) and lu.files_exist(cfg.data_dir, cfg.trust_filenames)):
            generic_message = (
                f"Could not obtain a usable ratings/trust file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place files named one of "
                f"{cfg.ratings_filenames} (ratings) and {cfg.trust_filenames} (trust) "
                f"directly into {cfg.data_dir}."
            )
            if cfg.manual_download_instructions:
                raise lu.ManualDownloadRequiredError(f"{generic_message}\n\n{cfg.manual_download_instructions}")
            raise lu.ManualDownloadRequiredError(generic_message)
```

And insert the negative-trust filter into `load()`, immediately after both `parse_rows`
calls and before the `n_raw = len(df_ratings)` stats line. Replace:

```python
        print(f"  [ExplicitTrustLoader:{cfg.name}] Parsing raw files ...", flush=True)
        df_ratings = lu.parse_rows(self._ratings_path, cfg.delimiter, cfg.explicit_rating_col_index, ("user", "item", "rating"))
        df_trust = lu.parse_rows(self._trust_path, cfg.delimiter, cfg.explicit_rating_col_index, ("src", "dst", "weight"))

        n_raw = len(df_ratings)
```

With:

```python
        print(f"  [ExplicitTrustLoader:{cfg.name}] Parsing raw files ...", flush=True)
        df_ratings = lu.parse_rows(self._ratings_path, cfg.delimiter, cfg.explicit_rating_col_index, ("user", "item", "rating"))
        df_trust = lu.parse_rows(self._trust_path, cfg.delimiter, cfg.explicit_rating_col_index, ("src", "dst", "weight"))

        if cfg.filter_negative_trust:
            n_trust_before = len(df_trust)
            df_trust = df_trust[df_trust["weight"] > 0].copy()
            print(f"    Filtered distrust: {n_trust_before:,} -> {len(df_trust):,} trust edges", flush=True)

        n_raw = len(df_ratings)
```

- [ ] **Step 5: Update `ImplicitTrustLoader` to raise `ManualDownloadRequiredError`**

In `pipeline/data_loaders/implicit_trust_loader.py`, replace `_ensure_downloaded`:

```python
    def _ensure_downloaded(self) -> None:
        cfg = self.config
        if lu.files_exist(cfg.data_dir, cfg.ratings_filenames):
            print(f"  [ImplicitTrustLoader:{cfg.name}] Dataset files already present in {cfg.data_dir}", flush=True)
            return

        lu.download_with_fallback(cfg.ratings_urls, cfg.data_dir, cfg.name, "ImplicitTrustLoader")

        if not lu.files_exist(cfg.data_dir, cfg.ratings_filenames):
            raise lu.ManualDownloadRequiredError(
                f"Could not obtain a usable ratings file for '{cfg.name}' from any "
                f"configured URL.\nManual fallback: place a file named one of "
                f"{cfg.ratings_filenames} directly into {cfg.data_dir}."
            )
```

- [ ] **Step 6: Verify -- new fields, dead URL removed, exception type**

Run:
```bash
PYTHONPATH=. py -3 -c "
import dataclasses
from pipeline.data_loaders.loader_utils import ManualDownloadRequiredError
from pipeline.data_loaders.dataset_configs import DatasetConfig, CIAO_CONFIG, YELP_CONFIG

assert issubclass(ManualDownloadRequiredError, RuntimeError)
print('Check 1 (ManualDownloadRequiredError is a RuntimeError subclass): PASS')

field_names = {f.name for f in dataclasses.fields(DatasetConfig)}
assert 'manual_download_instructions' in field_names
assert 'filter_negative_trust' in field_names
defaults = DatasetConfig(name='x', data_dir='x', ratings_urls=[], trust_urls=[], ratings_filenames=[], trust_filenames=[])
assert defaults.manual_download_instructions == ''
assert defaults.filter_negative_trust is False
print('Check 2 (DatasetConfig has new defaulted fields): PASS')

assert 'daicoolb' not in ' '.join(CIAO_CONFIG.ratings_urls + CIAO_CONFIG.trust_urls)
assert 'daicoolb' not in ' '.join(YELP_CONFIG.ratings_urls + YELP_CONFIG.trust_urls)
assert CIAO_CONFIG.ratings_urls == ['https://guoguibing.github.io/librec/datasets/CiaoDVD.zip']
print('Check 3 (dead daicoolb fallback URL removed from Ciao/Yelp configs): PASS')
"
```
Expected output:
```
Check 1 (ManualDownloadRequiredError is a RuntimeError subclass): PASS
Check 2 (DatasetConfig has new defaulted fields): PASS
Check 3 (dead daicoolb fallback URL removed from Ciao/Yelp configs): PASS
```

- [ ] **Step 7: Run the regression checks (Ciao/Yelp/FilmTrust real-data equivalence, ML-100K smoke test)**

This proves the edits in Steps 1-5 didn't break any existing dataset's behavior.
`data/ciao/`, `data/yelp/` should already be present locally (from prior sub-project
work); `data/filmtrust/` and `data/ml-100k/` download/resolve as before if not present.

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_configs import CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

print('='*70); print('CIAO: new ExplicitTrustLoader vs old AcademicDataLoader'); print('='*70)
from pipeline.unified_arena.academic_data_loader import AcademicDataLoader

new_ciao = ExplicitTrustLoader(CIAO_CONFIG).load()
old_ciao = AcademicDataLoader(data_dir='data/ciao').load()

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

assert new_yelp.num_users == old_yelp_loader.num_users
assert new_yelp.num_items == old_yelp_loader.num_items
print('Check (Yelp): PASS')

print('='*70); print('FILMTRUST: new ExplicitTrustLoader vs old FilmTrustLoader'); print('='*70)
from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader

new_filmtrust = ExplicitTrustLoader(FILMTRUST_CONFIG).load()
old_ft_loader = FilmTrustLoader(data_dir='data/filmtrust')
old_ft_loader.download()
old_ft_loader.load_data()

assert new_filmtrust.num_users == old_ft_loader.num_users
assert new_filmtrust.num_items == old_ft_loader.num_items
print('Check (FilmTrust): PASS')

print('='*70); print('ALL EQUIVALENCE CHECKS PASSED (post-Task-1 regression check)'); print('='*70)
"
```
Expected output: three `Check (...): PASS` lines, ending in
`ALL EQUIVALENCE CHECKS PASSED (post-Task-1 regression check)`, no traceback.

Then run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
ds = DatasetFactory.create('ml-100k').load()
assert ds.mode == 'implicit'
assert ds.num_users == 943
assert ds.num_items == 1682
print('Check (ML-100K, post-Task-1 regression check): PASS')
"
```
Expected output: the loader's usual console banner/progress lines, then
`Check (ML-100K, post-Task-1 regression check): PASS`, no traceback.

- [ ] **Step 8: Commit**

```bash
git add pipeline/data_loaders/loader_utils.py pipeline/data_loaders/dataset_configs.py pipeline/data_loaders/explicit_trust_loader.py pipeline/data_loaders/implicit_trust_loader.py
git commit -m "feat(data_loaders): add ManualDownloadRequiredError and negative-trust filter

New DatasetConfig fields (manual_download_instructions,
filter_negative_trust) default to no-ops for Ciao/Yelp/FilmTrust --
verified via the existing real-data equivalence check. Also removes
a dead daicoolb fallback URL from CIAO_CONFIG/YELP_CONFIG (confirmed
404 during design research for this sub-project)."
```

---

### Task 2: Register and verify `EPINIONS_CONFIG`

**Files:**
- Modify: `pipeline/data_loaders/dataset_configs.py` (add `EPINIONS_CONFIG`; add `"epinions"` to `DATASET_REGISTRY`)

**Interfaces:**
- Consumes: `DatasetConfig` (Task 1's two new fields), `ManualDownloadRequiredError` (Task 1, exercised only on failure -- not expected to fire here, since this URL is verified working).
- Produces: `EPINIONS_CONFIG: DatasetConfig`, registered as `"epinions"` in `DATASET_REGISTRY`.

- [ ] **Step 1: Add `EPINIONS_CONFIG` and register it**

Add this after `FILMTRUST_CONFIG`'s definition (before `DATASET_REGISTRY`):

```python
EPINIONS_CONFIG = DatasetConfig(
    name="epinions",
    data_dir="data/epinions",
    ratings_urls=["https://static.preferred.ai/cornac/datasets/epinions/ratings_data.zip"],
    trust_urls=["https://static.preferred.ai/cornac/datasets/epinions/trust_data.zip"],
    ratings_filenames=["ratings_data.txt"],
    trust_filenames=["trust_data.txt"],
    delimiter="space",
    k_core=5,
    feedback_mode="explicit",
    rating_threshold=0.0,
    filter_negative_trust=True,
    test_ratio=0.2,
    seed=42,
)
```

Then update `DATASET_REGISTRY` (replace the existing 3-entry dict with a 4-entry one):

```python
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "ciao": CIAO_CONFIG,
    "yelp": YELP_CONFIG,
    "filmtrust": FILMTRUST_CONFIG,
    "epinions": EPINIONS_CONFIG,
}
```

- [ ] **Step 2: Empirically check the real downloaded `trust_data.txt` for negative weights**

This is required evidence, not optional -- it determines whether
`filter_negative_trust=True` is load-bearing for this specific mirror or a defensive
no-op. Run this BEFORE the full loader verification in Step 3, so the file is freshly
downloaded by Step 3 if it isn't already present; if `data/epinions/trust_data.txt`
doesn't exist yet when you run this, run Step 3 first to trigger the download, then
come back to this check.

Run:
```bash
PYTHONPATH=. py -3 -c "
path = 'data/epinions/trust_data.txt'
neg_count = 0
total = 0
with open(path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            total += 1
            if float(parts[2]) <= 0:
                neg_count += 1
print(f'Real trust_data.txt: {total} total rows, {neg_count} with weight <= 0 ({100*neg_count/total:.2f}%)')
"
```
Expected output: a line reporting the real counts. There is no pre-determined "correct"
number here -- report exactly what this prints in your task report, and state
explicitly whether `filter_negative_trust=True` is therefore load-bearing
(`neg_count > 0`) or a no-op safety net (`neg_count == 0`) for this specific dataset.
Do not treat either outcome as a failure.

- [ ] **Step 3: Verify the full loader against real Epinions data**

Run:
```bash
PYTHONPATH=. py -3 -c "
import numpy as np
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY

ds = DatasetFactory.create('epinions').load()

assert ds.mode == 'explicit'
assert ds.num_users > 0 and ds.num_items > 0
print(f'Check 1 (loaded successfully): num_users={ds.num_users}, num_items={ds.num_items}, '
      f'train_nnz={ds.train_csr.nnz}, social_nnz={ds.social_csr.nnz}')

unique_vals = set(np.unique(ds.train_csr.data).round(2).tolist())
assert unique_vals <= {1.0, 2.0, 3.0, 4.0, 5.0}, f'unexpected train_csr values: {unique_vals}'
print(f'Check 2 (train_csr explicit ratings, values={sorted(unique_vals)}): PASS')

assert (ds.social_csr != ds.social_csr.T).nnz == 0, 'social_csr must be symmetric'
assert ds.social_csr.diagonal().sum() == 0, 'social_csr must have zero diagonal'
print('Check 3 (social_csr symmetric, zero-diagonal): PASS')

assert set(DATASET_REGISTRY.keys()) == {'ciao', 'yelp', 'filmtrust', 'epinions'}
print('Check 4 (registry has exactly 4 datasets): PASS')

print(f'Raw stats: n_raw_interactions={ds.n_raw_interactions:,}, n_raw_users={ds.n_raw_users:,}, '
      f'n_raw_items={ds.n_raw_items:,}, filtering_rounds={ds.filtering_rounds}')
"
```
Expected output: the loader's usual console progress lines (including the
`Filtered distrust: ... -> ...` line from Task 1's Step 4, since `filter_negative_trust=True`
for this config), then all four `Check (...): PASS` lines and the final raw-stats
line, with no traceback. The exact `num_users`/`num_items`/`nnz` numbers are real,
data-dependent values -- report them in your task report; they are not pre-determined.

- [ ] **Step 4: Commit**

```bash
git add pipeline/data_loaders/dataset_configs.py
git commit -m "feat(data_loaders): add EPINIONS_CONFIG, verified against real data

Real working mirror found during design research (static.preferred.ai,
maintained by the cornac recsys library) -- not a guessed URL. Loaded
and verified end-to-end against the real downloaded files."
```

---

### Task 3: Register `DOUBAN_CONFIG` (manual-download-only)

**Files:**
- Modify: `pipeline/data_loaders/dataset_configs.py` (add `DOUBAN_CONFIG`; add `"douban"` to `DATASET_REGISTRY`)

**Interfaces:**
- Consumes: `DatasetConfig` (Task 1's `manual_download_instructions` field), `ManualDownloadRequiredError` (Task 1) -- this task's verification specifically exercises the failure path, unlike Task 2's Epinions which exercises the success path.
- Produces: `DOUBAN_CONFIG: DatasetConfig`, registered as `"douban"` in `DATASET_REGISTRY`.

- [ ] **Step 1: Add `DOUBAN_CONFIG` and register it**

Add this after `EPINIONS_CONFIG`'s definition (before `DATASET_REGISTRY`):

```python
DOUBAN_CONFIG = DatasetConfig(
    name="douban",
    data_dir="data/douban",
    ratings_urls=[],
    trust_urls=[],
    ratings_filenames=["uir.index", "ratings.txt"],
    trust_filenames=["social.index", "trust.txt"],
    delimiter="space",
    k_core=5,
    feedback_mode="explicit",
    rating_threshold=0.0,
    filter_negative_trust=False,
    test_ratio=0.2,
    seed=42,
    manual_download_instructions=(
        "Douban (Hao Ma et al., 'Recommender systems with social regularization', "
        "WSDM 2011) has no working automated download as of 2026-06-24: the "
        "original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its "
        "'.new' variant both return 404, and the ASU Social Computing Data "
        "Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own "
        "description directs manual requests to 113333244@qq.com. Once obtained, "
        "place 'uir.index' (format: UserId ItemId Rating) and 'social.index' "
        "(format: UserId1 UserId2) into data/douban/. NOTE: this column layout is "
        "from secondary documentation, not a primary file inspection -- verify it "
        "against the real file before trusting any benchmark numbers."
    ),
)
```

Then update `DATASET_REGISTRY` (replace the 4-entry dict from Task 2 with a 5-entry one):

```python
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "ciao": CIAO_CONFIG,
    "yelp": YELP_CONFIG,
    "filmtrust": FILMTRUST_CONFIG,
    "epinions": EPINIONS_CONFIG,
    "douban": DOUBAN_CONFIG,
}
```

- [ ] **Step 2: Verify the manual-download-required failure path**

This must run with no `data/douban/` directory present (or an empty one) -- it proves
the graceful failure path, not a successful load (there is no real Douban data to load
in this sub-project; that's the whole point of this config).

Run:
```bash
PYTHONPATH=. py -3 -c "
import shutil, os
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.loader_utils import ManualDownloadRequiredError

if os.path.exists('data/douban'):
    shutil.rmtree('data/douban')

try:
    DatasetFactory.create('douban').load()
    print('Check 1 (ManualDownloadRequiredError raised): FAIL -- no exception raised')
except ManualDownloadRequiredError as e:
    msg = str(e)
    assert '113333244@qq.com' in msg, f'expected email contact in error message, got: {msg}'
    assert 'uir.index' in msg and 'social.index' in msg, f'expected filename guidance in error message, got: {msg}'
    print('Check 1 (ManualDownloadRequiredError raised with full manual instructions): PASS')

from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY
assert set(DATASET_REGISTRY.keys()) == {'ciao', 'yelp', 'filmtrust', 'epinions', 'douban'}
print('Check 2 (registry has exactly 5 datasets): PASS')
"
```
Expected output:
```
Check 1 (ManualDownloadRequiredError raised with full manual instructions): PASS
Check 2 (registry has exactly 5 datasets): PASS
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/dataset_configs.py
git commit -m "feat(data_loaders): add DOUBAN_CONFIG (manual-download-only)

No working automated source exists (dead CUHK/ASU mirrors, confirmed
during design research; manual access requires emailing the
dataset's maintainers). Verified that the loader fails gracefully
with ManualDownloadRequiredError and complete manual instructions."
```
