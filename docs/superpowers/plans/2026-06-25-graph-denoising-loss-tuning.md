# Graph Denoising & Loss Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vectorized homophily filter that prunes low-overlap social edges from `ExplicitTrustLoader` (enabled by default for Ciao/Yelp), and replace `SocialLightGCNEngine`'s adaptive social-loss weighting with a fixed, configurable `social_loss_weight`.

**Architecture:** Two independent layers, four tasks. Data layer (Tasks 1-3): two new `DatasetConfig` fields, a new pure function `denoise_social_edges` in `loader_utils.py`, wired into `ExplicitTrustLoader.load()` right before `_build_social_matrix`. Engine layer (Task 4): `SocialLightGCNEngine` gains a `social_loss_weight` constructor parameter that replaces the social task's adaptive log-variance scaling.

**Tech Stack:** Python, `numpy`, `scipy.sparse`, `pandas`, `torch`, the standard library `logging` module (new to this package).

## Global Constraints

- This is the first, intentional, scoped exception to the "engines are frozen" policy from every prior sub-project — ONLY `pipeline/engines/social_lightgcn_engine.py` is modified. The other four engines, the three existing arena scripts, `pipeline/run_pipeline.py`, and `pipeline/engines/unified_data_loader.py` are NOT touched.
- `denoise_social_graph: bool = False`, `denoise_jaccard_threshold: float = 0.05` are the exact new `DatasetConfig` field names/defaults. `CIAO_CONFIG` and `YELP_CONFIG` get `denoise_social_graph=True` explicitly; `FILMTRUST_CONFIG`/`EPINIONS_CONFIG`/`DOUBAN_CONFIG` are unchanged (inherit the `False` default — zero behavior change for those three).
- `denoise_social_edges` computes per-edge Jaccard similarity using **train-only** interactions (`train_csr`), never test-set interactions — matching the anti-leakage precedent from sub-project 3's `compute_sparse_jaccard_trust`.
- `denoise_social_edges` uses the edge-targeted vectorized approach (`binary[s_idx].multiply(binary[d_idx]).sum(axis=1)`) — bounded by edge count, zero Python loops over users or edges. Do NOT mirror `sparse_jaccard.py`'s all-pairs chunked search (it solves a different problem).
- The "Garbage Edges pruned" announcement uses Python's `logging` module (`logging.getLogger(__name__)`, `.info(...)`), not `print()` — new to `loader_utils.py`, used only for this one announcement. **Any verification script that needs to see this output must call `logging.basicConfig(level=logging.INFO)` before loading a dataset** — Python's logging defaults to WARNING level, so without this call the INFO message is silently swallowed, not printed.
- `SocialLightGCNEngine.__init__` gains `social_loss_weight: float = 0.01` as its new default — takes effect immediately, no opt-in flag. `loss_social = self.social_loss_weight * F.mse_loss(torch.sigmoid(social_preds), social_trust_t)` replaces the adaptive `log_vars[2]`-scaled version. BPR and Rating-MSE losses, and `log_vars[0]`/`log_vars[1]`, are unchanged. `log_vars` stays a `nn.Parameter(torch.zeros(3))` (same shape) — `log_vars[2]` becomes inert (receives an exactly-zero gradient forever, so under vanilla Adam with no weight decay it never moves from its `0.0` init value).
- No new third-party dependencies. No pytest framework in this repo — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.` (plain `python` is not on PATH in this environment).
- Known real-data baselines (from sub-project 4's regression checks, pre-denoising): Ciao `social_csr.nnz` = 66,232; Yelp `social_csr.nnz` = 1,001,010; FilmTrust `social_csr.nnz` = 2,618 (FilmTrust's must stay exactly this value, since denoising is not enabled for it).

---

### Task 1: `DatasetConfig` fields + Ciao/Yelp config updates

**Files:**
- Modify: `pipeline/data_loaders/dataset_configs.py`

**Interfaces:**
- Produces: `DatasetConfig.denoise_social_graph: bool = False`, `DatasetConfig.denoise_jaccard_threshold: float = 0.05`. Consumed by Task 3's `ExplicitTrustLoader.load()` change.

- [ ] **Step 1: Add the two new fields to `DatasetConfig`**

Replace:
```python
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

With:
```python
        filter_negative_trust: if True, drop trust rows with weight <= 0 before the
            union-of-users computation and before building social_csr -- for datasets
            where a negative weight encodes explicit distrust (Epinions), as opposed
            to a plain undirected friendship/trust concept with no distrust notion
            (Ciao, Yelp, FilmTrust, Douban). Defaults to False (no-op) for all of them.
        denoise_social_graph: if True, prune trust edges whose endpoints' TRAIN-ONLY
            item-interaction overlap (Jaccard similarity) falls below
            denoise_jaccard_threshold, right before social_csr is built -- removes
            "garbage" social edges between users who are connected but don't share
            tastes (a homophily filter). Defaults to False (no-op).
        denoise_jaccard_threshold: only consulted when denoise_social_graph is True.
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
    denoise_social_graph: bool = False
    denoise_jaccard_threshold: float = 0.05
```

- [ ] **Step 2: Enable denoising for `CIAO_CONFIG` and `YELP_CONFIG`**

Replace:
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
    ratings_filenames=["movie-ratings.txt", "ratings.txt"],
    trust_filenames=["trusts.txt"],
    delimiter="auto",
    explicit_rating_col_index=4,
    k_core=5,
    feedback_mode="threshold_binarize",
    rating_threshold=3.0,
    test_ratio=0.2,
    seed=42,
    denoise_social_graph=True,
)
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
    ratings_filenames=["ratings.txt", "train.txt"],
    trust_filenames=["trusts.txt", "trust.txt", "trustnetwork.txt", "links.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
)
```

With:
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
    ratings_filenames=["ratings.txt", "train.txt"],
    trust_filenames=["trusts.txt", "trust.txt", "trustnetwork.txt", "links.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
    denoise_social_graph=True,
)
```

- [ ] **Step 3: Verify**

Run:
```bash
PYTHONPATH=. py -3 -c "
import dataclasses
from pipeline.data_loaders.dataset_configs import DatasetConfig, CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG, EPINIONS_CONFIG, DOUBAN_CONFIG

field_names = {f.name for f in dataclasses.fields(DatasetConfig)}
assert 'denoise_social_graph' in field_names
assert 'denoise_jaccard_threshold' in field_names
defaults = DatasetConfig(name='x', data_dir='x', ratings_urls=[], trust_urls=[], ratings_filenames=[], trust_filenames=[])
assert defaults.denoise_social_graph is False
assert defaults.denoise_jaccard_threshold == 0.05
print('Check 1 (DatasetConfig has new defaulted fields): PASS')

assert CIAO_CONFIG.denoise_social_graph is True
assert CIAO_CONFIG.denoise_jaccard_threshold == 0.05
assert YELP_CONFIG.denoise_social_graph is True
assert YELP_CONFIG.denoise_jaccard_threshold == 0.05
print('Check 2 (Ciao/Yelp denoising enabled by default): PASS')

assert FILMTRUST_CONFIG.denoise_social_graph is False
assert EPINIONS_CONFIG.denoise_social_graph is False
assert DOUBAN_CONFIG.denoise_social_graph is False
print('Check 3 (FilmTrust/Epinions/Douban unaffected, denoising stays False): PASS')
"
```
Expected output:
```
Check 1 (DatasetConfig has new defaulted fields): PASS
Check 2 (Ciao/Yelp denoising enabled by default): PASS
Check 3 (FilmTrust/Epinions/Douban unaffected, denoising stays False): PASS
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/data_loaders/dataset_configs.py
git commit -m "feat(data_loaders): add denoise_social_graph config, enable for Ciao/Yelp"
```

---

### Task 2: `denoise_social_edges` in `loader_utils.py`

**Files:**
- Modify: `pipeline/data_loaders/loader_utils.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this is a pure function, independent of `DatasetConfig`).
- Produces: `denoise_social_edges(df_trust: pd.DataFrame, user_map: Dict[str, int], train_csr: sp.csr_matrix, jaccard_threshold: float) -> pd.DataFrame`. Consumed by Task 3's `ExplicitTrustLoader.load()` change.

- [ ] **Step 1: Add the `logging` import and module logger**

Replace:
```python
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
```

With:
```python
from __future__ import annotations

import io
import logging
import os
import zipfile
import urllib.request
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

_logger = logging.getLogger(__name__)

_USER_AGENT = (
```

- [ ] **Step 2: Add `denoise_social_edges` at the end of the file**

Append this new section after the existing "Sparse Matrix Construction" section (after the `build_dict` function, at the end of the file):

```python


# ------------------------------------------------------------------
# Social Graph Denoising (Homophily Filter)
# ------------------------------------------------------------------
def denoise_social_edges(
    df_trust: pd.DataFrame,
    user_map: Dict[str, int],
    train_csr: sp.csr_matrix,
    jaccard_threshold: float,
) -> pd.DataFrame:
    """
    Prune low-homophily trust edges: for each (src, dst) edge, compute the Jaccard
    similarity of src's and dst's TRAIN-ONLY item interaction sets (train_csr only --
    never test-set interactions, matching the anti-leakage precedent
    pipeline/utils/sparse_jaccard.py established for derived trust); drop edges whose
    similarity is below jaccard_threshold. Returns a filtered copy of df_trust (same
    columns/dtypes as the input, fresh 0..n-1 index). Logs how many edges were pruned.

    Fully vectorized: computes per-edge intersection counts via
    binary[s_idx].multiply(binary[d_idx]).sum(axis=1) -- bounded by edge count, not
    num_users^2 -- no Python loop over users or edges.
    """
    df = df_trust.copy()
    df["s_idx"] = df["src"].map(user_map)
    df["d_idx"] = df["dst"].map(user_map)
    df = df.dropna(subset=["s_idx", "d_idx"])
    df["s_idx"] = df["s_idx"].astype(int)
    df["d_idx"] = df["d_idx"].astype(int)

    n_before = len(df_trust)

    if len(df) == 0:
        _logger.info(
            "[HomophilyFilter] Garbage edges pruned: %d/%d (0.00%%, threshold=%.4f) -- no mappable edges",
            n_before, n_before, jaccard_threshold,
        )
        return df_trust.iloc[0:0].reset_index(drop=True)

    binary = train_csr.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    binary = binary.tocsr()
    degrees = np.asarray(binary.sum(axis=1)).flatten()

    s_idx = df["s_idx"].values
    d_idx = df["d_idx"].values

    intersection = np.asarray(binary[s_idx].multiply(binary[d_idx]).sum(axis=1)).flatten()
    union = degrees[s_idx] + degrees[d_idx] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    keep_mask = jaccard >= jaccard_threshold
    kept_index = df.index[keep_mask]

    result = df_trust.loc[kept_index].reset_index(drop=True)

    n_after = len(result)
    n_pruned = n_before - n_after
    pct = (100.0 * n_pruned / n_before) if n_before > 0 else 0.0
    _logger.info(
        "[HomophilyFilter] Garbage edges pruned: %d/%d (%.2f%%, threshold=%.4f) -- %d edges kept",
        n_pruned, n_before, pct, jaccard_threshold, n_after,
    )
    return result
```

- [ ] **Step 3: Verify with a synthetic, hand-computable example**

Run:
```bash
PYTHONPATH=. py -3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.data_loaders.loader_utils import denoise_social_edges

# 4 users, 9 items. user0={0,1,2}, user1={0,1,3}, user2={5,6,7}, user3={0,1,2,8}.
# Hand-computed Jaccard: (u0,u1)=2/4=0.5, (u0,u2)=0/6=0.0, (u0,u3)=3/4=0.75.
rows = [0,0,0, 1,1,1, 2,2,2, 3,3,3,3]
cols = [0,1,2, 0,1,3, 5,6,7, 0,1,2,8]
vals = [5,4,3, 4,5,2, 3,3,3, 5,5,4,1]
train_csr = sp.csr_matrix((vals, (rows, cols)), shape=(4, 9), dtype=np.float32)

user_map = {'u0': 0, 'u1': 1, 'u2': 2, 'u3': 3}
df_trust = pd.DataFrame({
    'src': ['u0', 'u0', 'u0'],
    'dst': ['u1', 'u2', 'u3'],
    'weight': [1.0, 1.0, 1.0],
})

result = denoise_social_edges(df_trust, user_map, train_csr, jaccard_threshold=0.05)
print(result)

kept_pairs = set(zip(result['src'], result['dst']))
assert kept_pairs == {('u0', 'u1'), ('u0', 'u3')}, f'expected (u0,u1) and (u0,u3) to survive, got {kept_pairs}'
assert len(result) == 2, f'expected 2 surviving edges, got {len(result)}'
print('Check 1 (low-overlap edge (u0,u2) correctly dropped, others kept): PASS')
"
```
Expected output: an `INFO: [HomophilyFilter] Garbage edges pruned: 1/3 (33.33%, threshold=0.0500) -- 2 edges kept` log line, the printed `result` DataFrame (2 rows: `u0->u1`, `u0->u3`), then:
```
Check 1 (low-overlap edge (u0,u2) correctly dropped, others kept): PASS
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/data_loaders/loader_utils.py
git commit -m "feat(data_loaders): add denoise_social_edges homophily filter"
```

---

### Task 3: Wire denoising into `ExplicitTrustLoader.load()`

**Files:**
- Modify: `pipeline/data_loaders/explicit_trust_loader.py`

**Interfaces:**
- Consumes: `cfg.denoise_social_graph`, `cfg.denoise_jaccard_threshold` (Task 1); `lu.denoise_social_edges` (Task 2).

- [ ] **Step 1: Insert the denoising call right before `_build_social_matrix`**

Replace:
```python
        social_csr = self._build_social_matrix(df_trust, user_map, num_users)
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)
```

With:
```python
        if cfg.denoise_social_graph:
            df_trust = lu.denoise_social_edges(df_trust, user_map, train_csr, cfg.denoise_jaccard_threshold)

        social_csr = self._build_social_matrix(df_trust, user_map, num_users)
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)
```

- [ ] **Step 2: Verify with the real Ciao/Yelp/FilmTrust regression + edge-reduction check**

This re-runs the same equivalence comparison from sub-project 4 (proving nothing else
broke) while updating the social-edge assertions to reflect the new, expected
denoising behavior. `data/ciao/`, `data/yelp/` should already be present from prior
sub-projects' real downloads; `data/filmtrust/` downloads fresh if not present (small,
proven reliable).

Run:
```bash
PYTHONPATH=. py -3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from pipeline.data_loaders.dataset_configs import CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

print('='*70); print('CIAO: new ExplicitTrustLoader (denoised) vs old AcademicDataLoader'); print('='*70)
from pipeline.unified_arena.academic_data_loader import AcademicDataLoader

new_ciao = ExplicitTrustLoader(CIAO_CONFIG).load()
old_ciao = AcademicDataLoader(data_dir='data/ciao').load()

assert new_ciao.n_raw_interactions == old_ciao.n_raw_interactions
assert new_ciao.num_items == old_ciao.num_items
assert new_ciao.n_train_interactions == old_ciao.n_train_interactions
assert new_ciao.n_test_interactions == old_ciao.n_test_interactions
assert new_ciao.num_users >= old_ciao.num_users
assert new_ciao.social_csr.nnz < 66232, f'expected denoising to reduce Ciao edges below the pre-denoising baseline 66,232, got {new_ciao.social_csr.nnz}'
print(f'Check (Ciao): PASS -- num_users delta={new_ciao.num_users - old_ciao.num_users}, social_nnz={new_ciao.social_csr.nnz:,} (denoised, was 66,232)')

print('='*70); print('YELP: new ExplicitTrustLoader (denoised) vs old YelpDataLoader'); print('='*70)
from pipeline.academic_sandbox.yelp_data_loader import YelpDataLoader

new_yelp = ExplicitTrustLoader(YELP_CONFIG).load()
old_yelp_loader = YelpDataLoader(data_dir='data/yelp')
old_yelp_loader.download()
old_yelp_loader.load_data()

assert new_yelp.num_users == old_yelp_loader.num_users
assert new_yelp.num_items == old_yelp_loader.num_items
assert new_yelp.social_csr.nnz < 1001010, f'expected denoising to reduce Yelp edges below the pre-denoising baseline 1,001,010, got {new_yelp.social_csr.nnz}'
print(f'Check (Yelp): PASS -- social_nnz={new_yelp.social_csr.nnz:,} (denoised, was 1,001,010)')

print('='*70); print('FILMTRUST: new ExplicitTrustLoader (denoise_social_graph=False, unaffected) vs old FilmTrustLoader'); print('='*70)
from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader

new_filmtrust = ExplicitTrustLoader(FILMTRUST_CONFIG).load()
old_ft_loader = FilmTrustLoader(data_dir='data/filmtrust')
old_ft_loader.download()
old_ft_loader.load_data()

assert new_filmtrust.num_users == old_ft_loader.num_users
assert new_filmtrust.num_items == old_ft_loader.num_items
assert new_filmtrust.social_csr.nnz == 2618, f'expected FilmTrust social_csr.nnz unchanged at 2,618 (denoise_social_graph=False), got {new_filmtrust.social_csr.nnz}'
print(f'Check (FilmTrust): PASS -- social_nnz={new_filmtrust.social_csr.nnz:,} (unchanged, denoising not enabled for this dataset)')

print('='*70); print('ALL CHECKS PASSED (post-denoising regression + edge-reduction check)'); print('='*70)
"
```
Expected output: an `INFO: [HomophilyFilter] Garbage edges pruned: ...` line for both Ciao and Yelp (real, data-dependent prune counts -- report the exact numbers you observe, they are not pre-determined), three `Check (...): PASS` lines, ending in `ALL CHECKS PASSED (post-denoising regression + edge-reduction check)`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add pipeline/data_loaders/explicit_trust_loader.py
git commit -m "feat(data_loaders): wire denoise_social_edges into ExplicitTrustLoader"
```

---

### Task 4: `SocialLightGCNEngine` loss reweighting

**Files:**
- Modify: `pipeline/engines/social_lightgcn_engine.py`

**Interfaces:**
- Produces: `SocialLightGCNEngine.__init__(..., social_loss_weight: float = 0.01)`. No other public interface changes -- `fit`/`predict_rating`/`recommend_top_n`/`save_model`/`load_model` signatures are unchanged.

- [ ] **Step 1: Update the module docstring for accuracy**

Replace:
```python
"""
Social-LightGCN Engine — Social-Aware Graph Convolutional Network for recommendation (PyTorch).

Integrates collaborative signals (from bipartite user-item graph structures) and social signals 
(from user-user trust networks) via an Adaptive Attention Gate at each layer (Early Fusion).
Optimized via an Adaptive Multi-Task Learning (MTL) Loss formulation combining BPR ranking, 
rating regression, and social reconstruction under self-adaptive log-variance weights.

Technical contract inherited from BaseRecommenderEngine.
"""
```

With:
```python
"""
Social-LightGCN Engine — Social-Aware Graph Convolutional Network for recommendation (PyTorch).

Integrates collaborative signals (from bipartite user-item graph structures) and social signals 
(from user-user trust networks) via an Adaptive Attention Gate at each layer (Early Fusion).
Optimized via a hybrid Multi-Task Learning (MTL) Loss: BPR ranking and rating regression are
self-adaptively weighted via learned log-variance parameters (Kendall et al.); the social
reconstruction term uses a fixed, explicitly-configurable `social_loss_weight` instead (sub-project
6: Graph Denoising & Loss Tuning) -- this prevents the social signal from dominating gradient
updates and crowding out the primary collaborative-filtering objective.

Technical contract inherited from BaseRecommenderEngine.
"""
```

- [ ] **Step 2: Add `social_loss_weight` to the constructor**

Replace:
```python
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 2048,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
```

With:
```python
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 2048,
        social_loss_weight: float = 0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.social_loss_weight = social_loss_weight
```

- [ ] **Step 3: Replace the adaptive social loss with the static-weight version**

Replace:
```python
                loss_social = 0.5 * torch.exp(-self.model.log_vars[2]) * F.mse_loss(
                    torch.sigmoid(social_preds), social_trust_t
                ) + 0.5 * self.model.log_vars[2]
```

With:
```python
                loss_social = self.social_loss_weight * F.mse_loss(
                    torch.sigmoid(social_preds), social_trust_t
                )
```

- [ ] **Step 4: Verify with a real, quick training smoke test on FilmTrust**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine

dataset = DatasetFactory.create('filmtrust').load()

engine = SocialLightGCNEngine(num_users=dataset.num_users, num_items=dataset.num_items, n_epochs=2)
engine.fit({'interaction_matrix': dataset.train_csr, 'trust_matrix': dataset.social_csr})

log_vars = engine.model.log_vars.detach().cpu().numpy()
print('log_vars after training:', log_vars)
assert log_vars[2] == 0.0, f'log_vars[2] should remain exactly 0.0 (no gradient ever flows to it), got {log_vars[2]}'
print('Check 1 (log_vars[2] stayed exactly 0.0 -- no longer adaptively driven): PASS')
assert log_vars[0] != 0.0, 'log_vars[0] (BPR) should still be adaptively updated'
assert log_vars[1] != 0.0, 'log_vars[1] (Rating) should still be adaptively updated'
print('Check 2 (log_vars[0]/log_vars[1] still adaptively updated): PASS')

recs = engine.recommend_top_n(0, top_n=5)
assert isinstance(recs, list) and len(recs) <= 5
print(f'Check 3 (recommend_top_n works after training): PASS -- sample_recs={recs}')
"
```
Expected output: FilmTrust's loader progress lines, a `SocialGCN Epoch  1/2 | Loss: ... | BPR: ... | Rating MSE: ... | Social MSE: ... | log_vars: [...]` line (prints at epoch 1 per the existing `epoch == 0` condition), then:
```
log_vars after training: [<nonzero> <nonzero> 0.0]
Check 1 (log_vars[2] stayed exactly 0.0 -- no longer adaptively driven): PASS
Check 2 (log_vars[0]/log_vars[1] still adaptively updated): PASS
Check 3 (recommend_top_n works after training): PASS -- sample_recs=[...]
```
(The exact `log_vars[0]`/`log_vars[1]` values and `sample_recs` are real, non-deterministic outputs -- report what you observe, just confirm the third array element prints as exactly `0.0` and the two assertions pass.)

- [ ] **Step 5: Commit**

```bash
git add pipeline/engines/social_lightgcn_engine.py
git commit -m "feat(engines): replace SocialLightGCN's adaptive social loss with a fixed social_loss_weight"
```
