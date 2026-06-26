# Deep Contextual Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the homophily filter from hard binary edges to Jaccard-weighted soft edges (enabling Soft-Weighted Message Passing), and make the InfoNCE social loss degree-aware (down-weighting users who already have rich CF interaction histories), to close the remaining Recall@10 gap against vanilla LightGCN on Ciao and Yelp.

**Architecture:** Two independent layers, two tasks. Data layer (Task 1): `denoise_social_edges` now overwrites the surviving edges' weight column with their computed Jaccard score instead of leaving the original raw weight untouched; `_build_social_matrix` gains an opt-in `use_weighted_edges` parameter that uses those real values instead of a hardcoded `1.0`, symmetrized via element-wise maximum (not addition) to avoid double-counting. Engine layer (Task 2): `SocialLightGCNEngine.fit()` computes a per-user adaptive social-loss weight from each user's CF interaction degree, and `_info_nce_loss` applies it per-user before averaging, instead of a single flat `ssl_weight` scalar.

**Tech Stack:** Python, `numpy`, `scipy.sparse`, `pandas`, `torch`.

## Global Constraints

- **Correction to the original request, verified against the actual current code before writing this plan:** the request's Step 2 describes `loss_social = F.mse_loss(...)` weighted by a flat `self.social_loss_weight` scalar — that mechanism was torn down two sub-projects ago. The current engine (`pipeline/engines/social_lightgcn_engine.py`, post sub-project 8) uses an InfoNCE contrastive loss (`_info_nce_loss`, cosine similarity + `F.cross_entropy` against the diagonal) weighted by `self.ssl_weight` (currently `0.005`), not an MSE reconstruction loss. This plan adapts the *intent* (down-weight social pressure for data-rich users, up-weight it for cold-start users) to the *actual* mechanism: degree-aware scaling is applied to `_info_nce_loss`'s per-user terms, not to a nonexistent MSE term.
- **Correction: "interaction degree" must come from the binarized interaction matrix, not raw rating sums.** The request's pseudocode (`interaction_matrix.sum(axis=1)`) would sum *rating values* (e.g. 1-5 stars) where the matrix holds explicit ratings (FilmTrust, Epinions) — a user with five 5-star ratings would get "degree" 25, a user with ten 1-star ratings would get "degree" 10, inverting which user is actually more cold-start. `fit()` already computes `R_binary` (binarized interactions) and `user_degrees = np.array(R_binary.sum(axis=1)).flatten()` for the CF graph's own symmetric normalization — this plan reuses that exact existing array (true interaction *counts*, not rating sums) for the degree-aware weight, rather than recomputing from raw values.
- **Correction: symmetrization must use element-wise maximum, not addition+clip.** `_build_social_matrix` currently does `A_sym = (A + A.T); A_sym.data = np.minimum(A_sym.data, 1.0)`. Jaccard similarity is itself symmetric (`Jaccard(u,v) == Jaccard(v,u)`), so if `df_trust` contains both `(u,v)` and `(v,u)` rows (common for undirected friendship data), both rows get an *identical* Jaccard weight from `denoise_social_edges` — summing them via `A + A.T` would double-count that weight (e.g. a true Jaccard of 0.6 would become 1.2, silently clipped down to 1.0, corrupting the value). `A.maximum(A.T)` (the exact idiom `pipeline/utils/sparse_jaccard.py::compute_sparse_jaccard_trust` already uses for this same reason) is the correct operator — and is proven, not just claimed, to be **byte-identical to the old binary behavior** for every existing dataset: for binary 0/1 weights, whether an edge exists in one direction (`max(1,0)=1`, same as `1+0=1`) or both (`max(1,1)=1`, same as `min(1+1,1)=1`), the result is identical to today's output. This means the symmetrization fix applies unconditionally to both the binary and weighted paths, with zero behavior change for every dataset that doesn't opt into weighted edges.
- **Backward compatibility, precisely scoped.** `use_weighted_edges` defaults to `False`. `ExplicitTrustLoader.load()` passes `use_weighted_edges=cfg.denoise_social_graph` — i.e., only Ciao and Yelp (the only configs with `denoise_social_graph=True`) get weighted edges, because only their `df_trust` has actually had its weight column overwritten with real Jaccard scores by `denoise_social_edges`. FilmTrust/Epinions/Douban (`denoise_social_graph=False`) never call `denoise_social_edges` at all, so their `df_trust["weight"]` still holds whatever raw value the dataset's own file format originally had (Epinions' real trust ratings, etc.) — using those values as graph weights would be a different, unvalidated experiment nobody asked for. Passing `use_weighted_edges=False` for them preserves their exact current binary behavior.
- This is the only engine file any sub-project in this initiative is authorized to touch: `pipeline/engines/social_lightgcn_engine.py`. The two data-layer files (`loader_utils.py`, `explicit_trust_loader.py`) were already authorized for modification in sub-project 6 and remain so. No other engine file, arena script, or `pipeline/run_pipeline.py` is touched.
- `_info_nce_loss` gains a `per_user_weight: torch.Tensor` parameter, indexed by the *same* deduplicated `unique_users` it already computes — each unique user's `cross_entropy(..., reduction="none")` term is multiplied by their own weight before the final `.mean()`. The function now returns an already-weighted scalar; **`fit()`'s `loss_total` must NOT separately multiply by `self.ssl_weight` anymore** (it's baked into the per-user weight vector via `adaptive_social_weight = self.ssl_weight / torch.log(user_degrees + 2.0)`) — `loss_total = loss_bpr + reg_loss + loss_scl`, not `loss_bpr + reg_loss + self.ssl_weight * loss_scl`. Double-applying `ssl_weight` (once in the per-user vector, again in `loss_total`) would silently shrink the social signal by another `ssl_weight` factor on top of the degree scaling, defeating the point.
- `adaptive_social_weight` is computed **once per `fit()` call** (degree doesn't change across epochs/batches), not recomputed per-batch.
- No new third-party dependencies. No pytest framework in this repo — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.`.
- Known real baselines from the last full sweep (Social Contrastive Learning, pre-this-optimization): Ciao `lightgcn` Recall@10=0.0787 vs `social_lightgcn` 0.0730 (-7.2%); Yelp `lightgcn` 0.0376 vs `social_lightgcn` 0.0362 (-3.7%).

---

### Task 1: Data Layer — Jaccard-Weighted Social Graph

**Files:**
- Modify: `pipeline/data_loaders/loader_utils.py` (`denoise_social_edges`)
- Modify: `pipeline/data_loaders/explicit_trust_loader.py` (`_build_social_matrix`, `load()`)

**Interfaces:**
- Produces: `denoise_social_edges(...)` now returns a `df_trust`-shaped DataFrame whose surviving rows' `weight` column holds the computed Jaccard score (previously: original raw weight, untouched). `_build_social_matrix(df_trust, user_map, n_users, use_weighted_edges: bool = False)` — new fourth parameter.
- Consumes: nothing new from other tasks.

- [ ] **Step 1: Overwrite the surviving edges' weight column with their Jaccard score**

In `pipeline/data_loaders/loader_utils.py`, replace:
```python
    keep_mask = jaccard >= jaccard_threshold
    kept_index = df.index[keep_mask]

    result = df_trust.loc[kept_index].reset_index(drop=True)

    n_after = len(result)
```

With:
```python
    keep_mask = jaccard >= jaccard_threshold
    kept_index = df.index[keep_mask]
    kept_jaccard = jaccard[keep_mask]

    result = df_trust.loc[kept_index].reset_index(drop=True)
    result["weight"] = kept_jaccard.astype(np.float32)

    n_after = len(result)
```

Also update the function's docstring -- replace:
```python
    similarity is below jaccard_threshold. Returns a filtered copy of df_trust (same
    columns/dtypes as the input, fresh 0..n-1 index). Logs how many edges were pruned.
```
With:
```python
    similarity is below jaccard_threshold. Returns a filtered copy of df_trust (same
    columns as the input, fresh 0..n-1 index) -- EXCEPT the "weight" column, which is
    overwritten with each surviving edge's computed Jaccard score (not the original
    raw weight) -- this is what enables Soft-Weighted Message Passing downstream in
    _build_social_matrix. Logs how many edges were pruned.
```

- [ ] **Step 2: Verify with a synthetic, hand-computable example**

Run:
```bash
PYTHONPATH=. py -3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.data_loaders.loader_utils import denoise_social_edges

# Same synthetic setup as sub-project 6's test: user0={0,1,2}, user1={0,1,3}, user3={0,1,2,8}.
# Hand-computed Jaccard: (u0,u1)=2/4=0.5, (u0,u3)=3/4=0.75.
rows = [0,0,0, 1,1,1, 3,3,3,3]
cols = [0,1,2, 0,1,3, 0,1,2,8]
vals = [5,4,3, 4,5,2, 5,5,4,1]
train_csr = sp.csr_matrix((vals, (rows, cols)), shape=(4, 9), dtype=np.float32)

user_map = {'u0': 0, 'u1': 1, 'u3': 3}
df_trust = pd.DataFrame({
    'src': ['u0', 'u0'],
    'dst': ['u1', 'u3'],
    'weight': [1.0, 1.0],  # original raw weight -- must be REPLACED by Jaccard score
})

result = denoise_social_edges(df_trust, user_map, train_csr, jaccard_threshold=0.05)
print(result)

weight_map = dict(zip(zip(result['src'], result['dst']), result['weight']))
assert abs(weight_map[('u0', 'u1')] - 0.5) < 1e-6, f'expected weight 0.5 for (u0,u1), got {weight_map[(\"u0\",\"u1\")]}'
assert abs(weight_map[('u0', 'u3')] - 0.75) < 1e-6, f'expected weight 0.75 for (u0,u3), got {weight_map[(\"u0\",\"u3\")]}'
print('Check 1 (weight column correctly overwritten with Jaccard scores, not left at original 1.0): PASS')
"
```
Expected output: the printed `result` DataFrame showing `weight` values `0.5` and `0.75` (NOT `1.0`), then:
```
Check 1 (weight column correctly overwritten with Jaccard scores, not left at original 1.0): PASS
```

- [ ] **Step 3: Add `use_weighted_edges` to `_build_social_matrix`, fix symmetrization**

In `pipeline/data_loaders/explicit_trust_loader.py`, replace:
```python
    @staticmethod
    def _build_social_matrix(
        df_trust, user_map: Dict[str, int], n_users: int
    ) -> sp.csr_matrix:
        """Build symmetric undirected trust matrix: A = A_raw + A_raw^T, clipped binary.

        Self-loop rows (src == dst) are dropped before construction -- a self-trust
        edge would otherwise survive symmetrization as a nonzero diagonal entry,
        violating the zero-diagonal invariant every consumer of social_csr assumes.
        """
        df = df_trust.copy()
        df["s_idx"] = df["src"].map(user_map)
        df["d_idx"] = df["dst"].map(user_map)
        df = df.dropna(subset=["s_idx", "d_idx"])
        df["s_idx"] = df["s_idx"].astype(int)
        df["d_idx"] = df["d_idx"].astype(int)
        df = df[df["s_idx"] != df["d_idx"]]

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

With:
```python
    @staticmethod
    def _build_social_matrix(
        df_trust, user_map: Dict[str, int], n_users: int, use_weighted_edges: bool = False,
    ) -> sp.csr_matrix:
        """Build symmetric undirected trust matrix.

        use_weighted_edges=False (default): binary trust matrix -- every edge gets
        weight 1.0, regardless of whatever is in df_trust's "weight" column. This is
        byte-identical to this method's behavior before Soft-Weighted Message Passing
        was added (see proof in the symmetrization note below) -- the safe default for
        every dataset that hasn't opted into denoise_social_graph.

        use_weighted_edges=True: uses df_trust's real "weight" column values as edge
        weights instead of a hardcoded 1.0. Only meaningful for callers that have
        already run denoise_social_edges, which overwrites surviving edges' weight
        with their computed Jaccard similarity score -- this is what enables Soft-
        Weighted Message Passing (PyTorch's sparse matmul naturally scales messages
        by homophily instead of treating every surviving edge as equally trustworthy).

        Symmetrized via element-wise maximum (A.maximum(A.T)), matching the precedent
        in pipeline/utils/sparse_jaccard.py -- NOT addition. Jaccard similarity is
        itself symmetric, so if df_trust contains both a (u,v) and a (v,u) row, both
        carry the identical weight; summing them would double-count it (e.g. a true
        Jaccard of 0.6 would become 1.2). Maximum is provably equivalent to the old
        addition+clip behavior for binary weights (whether an edge exists in one
        direction or both, max() and the old sum-then-clip-to-1.0 produce the same
        result) and is the mathematically correct operator for weighted edges.

        Self-loop rows (src == dst) are dropped before construction -- a self-trust
        edge would otherwise survive symmetrization as a nonzero diagonal entry,
        violating the zero-diagonal invariant every consumer of social_csr assumes.
        """
        df = df_trust.copy()
        df["s_idx"] = df["src"].map(user_map)
        df["d_idx"] = df["dst"].map(user_map)
        df = df.dropna(subset=["s_idx", "d_idx"])
        df["s_idx"] = df["s_idx"].astype(int)
        df["d_idx"] = df["d_idx"].astype(int)
        df = df[df["s_idx"] != df["d_idx"]]

        if len(df) == 0:
            return sp.csr_matrix((n_users, n_users), dtype=np.float32)

        rows = df["s_idx"].values
        cols = df["d_idx"].values
        if use_weighted_edges:
            vals = df["weight"].values.astype(np.float32)
        else:
            vals = np.ones(len(rows), dtype=np.float32)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_users)).tocsr()
        A_sym = A.maximum(A.T).tocsr()
        return A_sym
```

- [ ] **Step 4: Wire `use_weighted_edges` into `load()`**

Replace:
```python
        social_csr = self._build_social_matrix(df_trust, user_map, num_users)
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)
```

With:
```python
        social_csr = self._build_social_matrix(
            df_trust, user_map, num_users, use_weighted_edges=cfg.denoise_social_graph
        )
        print(f"    Social: {social_csr.nnz:,} edges (symmetric)", flush=True)
```

- [ ] **Step 5: Verify symmetrization equivalence and real Ciao/Yelp weighted output**

Run:
```bash
PYTHONPATH=. py -3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import numpy as np

from pipeline.data_loaders.dataset_configs import CIAO_CONFIG, YELP_CONFIG, FILMTRUST_CONFIG
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader

# Ciao and Yelp: denoise_social_graph=True -> weighted edges, real Jaccard values in (0,1]
ciao = ExplicitTrustLoader(CIAO_CONFIG).load()
assert ciao.social_csr.dtype == np.float32
unique_vals = np.unique(ciao.social_csr.data)
assert len(unique_vals) > 2, f'expected more than 2 distinct edge weights (real Jaccard scores), got {unique_vals}'
assert ciao.social_csr.data.max() <= 1.0 and ciao.social_csr.data.min() > 0.0, f'expected weights in (0,1], got min={ciao.social_csr.data.min()}, max={ciao.social_csr.data.max()}'
print(f'Check 1 (Ciao social_csr has real, varied Jaccard weights): PASS -- {len(unique_vals)} distinct weight values, range=[{ciao.social_csr.data.min():.4f}, {ciao.social_csr.data.max():.4f}]')

yelp = ExplicitTrustLoader(YELP_CONFIG).load()
unique_vals_yelp = np.unique(yelp.social_csr.data)
assert len(unique_vals_yelp) > 2
print(f'Check 2 (Yelp social_csr has real, varied Jaccard weights): PASS -- {len(unique_vals_yelp)} distinct weight values, range=[{yelp.social_csr.data.min():.4f}, {yelp.social_csr.data.max():.4f}]')

# FilmTrust: denoise_social_graph=False -> still binary, byte-identical to before
filmtrust = ExplicitTrustLoader(FILMTRUST_CONFIG).load()
assert filmtrust.social_csr.nnz == 2618, f'expected FilmTrust social_csr.nnz unchanged at 2,618, got {filmtrust.social_csr.nnz}'
assert set(np.unique(filmtrust.social_csr.data)) == {1.0}, f'expected FilmTrust to stay purely binary (all weights == 1.0), got {np.unique(filmtrust.social_csr.data)}'
print(f'Check 3 (FilmTrust unaffected, still pure binary, nnz unchanged at 2,618): PASS')
"
```
Expected output: `Check 1`, `Check 2`, `Check 3` all printing `PASS`, with Ciao/Yelp showing many distinct weight values in `(0, 1]` and FilmTrust showing exactly `{1.0}` and `nnz=2618`. No traceback.

- [ ] **Step 6: Commit**

```bash
git add pipeline/data_loaders/loader_utils.py pipeline/data_loaders/explicit_trust_loader.py
git commit -m "feat(data_loaders): upgrade homophily filter to Jaccard-weighted soft edges"
```

---

### Task 2: Engine Layer — Degree-Aware Social Loss

**Files:**
- Modify: `pipeline/engines/social_lightgcn_engine.py`

**Interfaces:**
- Consumes: nothing new from Task 1 directly (the engine doesn't care whether `adj_social`'s edges are weighted or binary -- `torch.sparse.mm` handles either transparently). Independently testable without Task 1.
- Produces: `_info_nce_loss(emb_cf, emb_social, user_ids, temperature, per_user_weight) -> torch.Tensor` (signature change: new required `per_user_weight` parameter, returns an already-weighted scalar).

- [ ] **Step 1: Add `per_user_weight` to `_info_nce_loss`**

Replace:
```python
def _info_nce_loss(
    emb_cf: torch.Tensor,
    emb_social: torch.Tensor,
    user_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss between a user's CF view and Social view.

    For each user u in the batch, (emb_cf[u], emb_social[u]) is the positive pair;
    (emb_cf[u], emb_social[v]) for every other user v in the batch is a negative
    pair (in-batch negatives). Implemented as cosine-similarity logits fed through
    cross_entropy against the diagonal -- mathematically identical to
    -log(exp(sim(pos)/tau) / sum(exp(sim(all)/tau))), fully vectorized, no Python
    loop over users.

    user_ids is deduplicated via torch.unique() before building the similarity
    matrix: _sample_training_batch samples with replacement, so a batch can contain
    the same user twice -- without deduplication, a repeated user would be scored as
    a "negative" against itself, corrupting the loss.
    """
    unique_users = torch.unique(user_ids)
    cf_view = F.normalize(emb_cf[unique_users], dim=1)
    social_view = F.normalize(emb_social[unique_users], dim=1)

    sim_matrix = torch.matmul(cf_view, social_view.T) / temperature  # [N, N]
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)
```

With:
```python
def _info_nce_loss(
    emb_cf: torch.Tensor,
    emb_social: torch.Tensor,
    user_ids: torch.Tensor,
    temperature: float,
    per_user_weight: torch.Tensor,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss between a user's CF view and Social view, with a
    per-user weight applied to each user's own loss term before averaging --
    enables degree-aware scaling (e.g. down-weighting users who already have rich
    CF interaction histories, up-weighting cold-start users).

    For each user u in the batch, (emb_cf[u], emb_social[u]) is the positive pair;
    (emb_cf[u], emb_social[v]) for every other user v in the batch is a negative
    pair (in-batch negatives). Implemented as cosine-similarity logits fed through
    cross_entropy against the diagonal -- mathematically identical to
    -log(exp(sim(pos)/tau) / sum(exp(sim(all)/tau))), fully vectorized, no Python
    loop over users.

    user_ids is deduplicated via torch.unique() before building the similarity
    matrix: _sample_training_batch samples with replacement, so a batch can contain
    the same user twice -- without deduplication, a repeated user would be scored as
    a "negative" against itself, corrupting the loss. per_user_weight is indexed by
    the same deduplicated unique_users, so each unique user's loss term is scaled by
    exactly their own weight once, regardless of how many times they were drawn in
    the pre-dedup batch. The returned scalar is already fully weighted -- callers
    must NOT separately multiply it by a global weight again.
    """
    unique_users = torch.unique(user_ids)
    cf_view = F.normalize(emb_cf[unique_users], dim=1)
    social_view = F.normalize(emb_social[unique_users], dim=1)

    sim_matrix = torch.matmul(cf_view, social_view.T) / temperature  # [N, N]
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    per_user_loss = F.cross_entropy(sim_matrix, labels, reduction="none")  # [N]

    weights = per_user_weight[unique_users]  # [N]
    return (per_user_loss * weights).mean()
```

- [ ] **Step 2: Compute `adaptive_social_weight` once per `fit()` call**

Replace:
```python
        T_norm = D_s_inv.dot(trust_mat).dot(D_s_inv)
        self.adj_social = _sparse_scipy_to_torch(T_norm.tocsr(), self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
```

With:
```python
        T_norm = D_s_inv.dot(trust_mat).dot(D_s_inv)
        self.adj_social = _sparse_scipy_to_torch(T_norm.tocsr(), self.device)

        # Degree-aware social loss scaling: down-weights InfoNCE pressure for users
        # who already have rich CF interaction histories, up-weights it for
        # cold-start users who stand to gain more from the social signal. Reuses
        # user_degrees (binarized interaction COUNTS, computed above for the CF
        # graph's own normalization) -- NOT raw rating sums, which would conflate
        # "many interactions" with "high ratings" and misrepresent cold-start status.
        # Computed once here since degree doesn't change across epochs/batches.
        user_degrees_t = torch.FloatTensor(user_degrees).to(self.device)
        adaptive_social_weight = self.ssl_weight / torch.log(user_degrees_t + 2.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
```

- [ ] **Step 3: Use the per-user weight at the call site, remove the now-redundant outer multiply**

Replace:
```python
                loss_scl = _info_nce_loss(
                    E_user_cf, E_user_social, users_t, self.temperature
                )

                loss_total = loss_bpr + reg_loss + self.ssl_weight * loss_scl
```

With:
```python
                loss_scl = _info_nce_loss(
                    E_user_cf, E_user_social, users_t, self.temperature, adaptive_social_weight
                )

                loss_total = loss_bpr + reg_loss + loss_scl
```

- [ ] **Step 4: Update the module docstring**

Replace:
```python
This sidesteps Late-Fusion's gate entirely (no per-user alpha to learn or interpret)
at the cost of losing the gate's per-user ability to suppress an unhelpful social
signal -- ssl_weight is a single global scalar, not an adaptive one.
```

With:
```python
This sidesteps Late-Fusion's gate entirely (no per-user alpha to learn or interpret).
Unlike the original version of this architecture, ssl_weight is no longer a single
global scalar applied flatly to every user -- it is scaled per-user by
1/log(degree+2), so users with rich CF interaction histories receive proportionally
less InfoNCE pressure (their CF embedding is already well-informed) and cold-start
users receive proportionally more (the social signal matters more when CF data is
sparse).
```

- [ ] **Step 5: Smoke test -- confirm no shape mismatch, both losses behave sanely**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine

ds = DatasetFactory.create('filmtrust').load()
engine = SocialLightGCNEngine(num_users=ds.num_users, num_items=ds.num_items, n_epochs=10)
engine.fit({'interaction_matrix': ds.train_csr, 'trust_matrix': ds.social_csr})
print('Check 1 (degree-aware loss runs through forward+backward with no shape mismatch): PASS')
"
```
Expected output: FilmTrust's loader progress lines, `SocialGCN Epoch  1/10 | BPR: <finite> | Reg: <finite> | SSL: <finite>` and `SocialGCN Epoch 10/10 | ...` lines with no traceback, no NaN, then `Check 1 (...): PASS`. FilmTrust has `denoise_social_graph=False` so this exercises the degree-aware loss against a binary (unweighted) social graph -- confirming Task 2 works independently of Task 1, as the interfaces section states.

- [ ] **Step 6: Dry-run on Ciao and Yelp -- both layers together, check whether the gap closed**

This is the real integration test: Ciao and Yelp now get BOTH Jaccard-weighted edges
(Task 1) AND degree-aware loss scaling (Task 2) together, and this directly answers
the question motivating this whole sub-project.

Run:
```bash
PYTHONPATH=. py -3 pipeline/benchmarks/grand_arena_runner.py --datasets ciao yelp
```
Expected output: both datasets' Mode A model tables print with no `FAILED` rows and
no traceback. Compare the printed `social_lightgcn` Recall@10/NDCG@10 against the
last full sweep's recorded baseline (Ciao: `lightgcn`=0.0787, `social_lightgcn`=0.0730;
Yelp: `lightgcn`=0.0376, `social_lightgcn`=0.0362) -- report the real, observed numbers
and whether the gap narrowed, closed, or reversed. This is a real, data-dependent
outcome, not a predetermined pass/fail -- report exactly what you observe.

- [ ] **Step 7: Commit**

```bash
git add pipeline/engines/social_lightgcn_engine.py
git commit -m "feat(engines): make InfoNCE social loss degree-aware (cold-start-weighted)"
```
