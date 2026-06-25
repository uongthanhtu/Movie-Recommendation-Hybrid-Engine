# Graph Denoising & Loss Tuning — Design Spec

Date: 2026-06-25

## Context: this is sub-project 6, outside the original 5-part initiative

The "Grand Unified Benchmark Arena" (sub-projects 1-5) is complete and merged. This is
a new, separate sub-project, motivated by the Grand Arena's own committed results
(`models/grand_arena_results.md`): `Social-LightGCN` underperforms vanilla `LightGCN`
on Ciao and Yelp (suspected social-noise / over-smoothing), and on Epinions it beats
TrustSVD but still trails LightGCN. The goal is to optimize `Social-LightGCN`'s
architecture to close that gap via two independent levers: pruning low-value social
edges before training (data layer), and reweighting the social loss term during
training (engine layer).

## Scope/policy note (binding, read this first)

Every prior sub-project in this initiative treated all five `pipeline/engines/*_engine.py`
files as frozen — production/parallel-arena code nothing was allowed to touch. **This
sub-project is the first, intentional, scoped exception**: `pipeline/engines/social_lightgcn_engine.py`
is modified. The other four engine files (`lightgcn_engine.py`, `trust_svd_engine.py`,
`funk_svd_engine.py`, `sasrec_engine.py`) remain frozen, as do the three existing arena
scripts and `pipeline/run_pipeline.py`.

## Research findings (verified against real code, not guessed)

- **The current `SocialLightGCNEngine` loss is NOT `L_bpr + lambda * L_social`** (the
  framing in the original request). It is a 3-task adaptive homoscedastic-uncertainty
  loss (Kendall et al.) over BPR ranking, Rating MSE, and Social MSE, each scaled by a
  *learned* `log_vars[i]` parameter — there is no existing static lambda to "lower."
  This was confirmed by reading `pipeline/engines/social_lightgcn_engine.py`'s `fit()`
  method directly (lines 367-401 in the pre-change file).
- **`_build_social_matrix` (in `explicit_trust_loader.py`) has no access to `train_csr`**
  today — it's a `@staticmethod` taking only `(df_trust, user_map, n_users)`. The new
  denoising filter needs both the trust edges AND the train-only interaction matrix, so
  it must run as a separate step between `train_csr` construction and the
  `_build_social_matrix` call, not inside that method.
- **A targeted, fully-vectorized Jaccard computation fits this problem better than
  mirroring `sparse_jaccard.py`'s all-pairs chunked search.** `sparse_jaccard.py` solves
  a different problem (search ALL `O(U^2)` candidate pairs, keep top-K) and chunks by
  user-row to bound that search's memory. Here, the exact `(u, v)` pairs to score are
  already known (the existing social edges) — `binary[s_idx].multiply(binary[d_idx]).sum(axis=1)`
  computes per-edge intersection counts in one shot via SciPy fancy-row-indexing +
  elementwise multiply, bounded by edge count (not `num_users^2`), with zero Python
  loops over users.

## Decisions made (binding scope boundaries for this sub-project)

- **Loss mechanism**: `social_loss_weight` REPLACES the adaptive `log_vars[2]` scaling
  for the social task only. BPR and Rating-MSE keep their existing adaptive weighting
  exactly as-is. This is a 2-adaptive + 1-static hybrid, not a return to a fully manual
  3-task loss.
- **Rating-MSE task untouched** — it stays in the loss, still adaptively weighted; this
  sub-project's scope is the social term and the social graph only.
- **New default takes effect immediately**, no opt-in flag: `social_loss_weight: float = 0.01`
  is simply the new constructor default. Every future `SocialLightGCNEngine` run
  (including via `grand_arena_runner.py`) uses the new weighting from the next run
  onward. The already-committed `models/grand_arena_results.md` becomes a stale,
  pre-optimization snapshot — expected and correct for an optimization sub-project, not
  a regression to fix.
- **Denoising algorithm**: the edge-targeted vectorized approach (`binary[s_idx].multiply(binary[d_idx]).sum(axis=1)`),
  not `sparse_jaccard.py`'s chunked all-pairs search.
- **Denoising uses `train_csr` only** (train-only interactions), never the full
  pre-split interaction set — matching the binding precedent from sub-project 3 (Jaccard-
  derived trust must never leak test-set co-occurrence information; pruning a real trust
  graph using test-set item overlap is the same category of leakage).
- **`denoise_social_graph=True` becomes the new default for `CIAO_CONFIG` and
  `YELP_CONFIG`** — the two datasets explicitly identified as suffering from social
  noise. `FILMTRUST_CONFIG`, `EPINIONS_CONFIG`, and `DOUBAN_CONFIG` stay at the
  `False` default (zero behavior change for those three).
- **`logging` module used for the "Garbage Edges pruned" announcement specifically** —
  a new pattern for `loader_utils.py`/`explicit_trust_loader.py` (which otherwise use
  plain `print(..., flush=True)` throughout), introduced only for this one new
  announcement, as explicitly requested. Not retrofitted onto any existing print
  statement in either file.

## Architecture

### `DatasetConfig` (modified, in `pipeline/data_loaders/dataset_configs.py`)

Two new fields, both defaulting to a no-op:

```python
denoise_social_graph: bool = False
denoise_jaccard_threshold: float = 0.05
```

`CIAO_CONFIG` and `YELP_CONFIG` are updated to pass `denoise_social_graph=True`
explicitly (keeping `denoise_jaccard_threshold` at the default `0.05`).
`FILMTRUST_CONFIG`/`EPINIONS_CONFIG`/`DOUBAN_CONFIG` are unchanged (inherit the
`False` default).

### `denoise_social_edges` (new, in `pipeline/data_loaders/loader_utils.py`)

```python
def denoise_social_edges(
    df_trust: pd.DataFrame,
    user_map: Dict[str, int],
    train_csr: sp.csr_matrix,
    jaccard_threshold: float,
) -> pd.DataFrame:
    """
    Prune low-homophily trust edges: for each (src, dst) edge, compute the Jaccard
    similarity of src's and dst's TRAIN-ONLY item interaction sets (never test-set
    interactions, to avoid the same category of leakage sub-project 3's binding
    decision forbids for Jaccard-derived trust); drop edges whose similarity is below
    jaccard_threshold. Returns a filtered copy of df_trust (same columns, same dtypes).
    Logs (via the logging module) how many edges were pruned.
    """
```

Implementation outline:
1. Map `src`/`dst` to integer indices via `user_map` (mirrors `_build_social_matrix`'s
   own existing mapping code), drop any unmapped rows (defensive; should be a no-op
   given the union-of-users construction already includes every trust endpoint).
2. Binarize `train_csr` (any nonzero -> 1.0), compute per-user item-degree via
   `.sum(axis=1)`.
3. `intersection = np.asarray(binary[s_idx].multiply(binary[d_idx]).sum(axis=1)).flatten()`
   — one vectorized call, no loop.
4. `union = degrees[s_idx] + degrees[d_idx] - intersection`; `jaccard = intersection / union`
   (zero-guarded).
5. Keep rows where `jaccard >= jaccard_threshold`; return the corresponding subset of
   the ORIGINAL `df_trust` (preserving original columns/dtypes, not the temporary
   index columns).
6. `logging.getLogger(__name__).info(...)` reporting `pruned / total` edge counts and
   the threshold used.

### `ExplicitTrustLoader.load()` (modified, in `pipeline/data_loaders/explicit_trust_loader.py`)

One new conditional step, inserted after `train_csr` is built and immediately before
`_build_social_matrix` is called:

```python
if cfg.denoise_social_graph:
    df_trust = lu.denoise_social_edges(df_trust, user_map, train_csr, cfg.denoise_jaccard_threshold)

social_csr = self._build_social_matrix(df_trust, user_map, num_users)
```

`_build_social_matrix`'s own signature and body are completely unchanged — it still
just receives whatever `df_trust` it's given (pruned or not) and builds the symmetric
matrix the same way it always has, including its existing self-loop drop from the
self-loop bug fix (sub-project 4). This is a no-op for every config with
`denoise_social_graph=False` (the default) — byte-identical behavior to today.

### `SocialLightGCNEngine` (modified, in `pipeline/engines/social_lightgcn_engine.py`)

Constructor gains one new parameter:

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
    ...
    self.social_loss_weight = social_loss_weight
```

In `fit()`, replace the social task's loss computation. Current code:

```python
loss_social = 0.5 * torch.exp(-self.model.log_vars[2]) * F.mse_loss(
    torch.sigmoid(social_preds), social_trust_t
) + 0.5 * self.model.log_vars[2]
```

New code:

```python
loss_social = self.social_loss_weight * F.mse_loss(
    torch.sigmoid(social_preds), social_trust_t
)
```

`self.model.log_vars` shrinks conceptually from "3 adaptive tasks" to "2 adaptive tasks
(BPR, Rating) + 1 statically-weighted task (Social)" — but `log_vars` itself stays a
`nn.Parameter(torch.zeros(3))` (unchanged shape) for minimal diff; `log_vars[2]` becomes
dead weight once the social loss path no longer references it — no loss term depends
on it, so `loss_total.backward()` always produces an exactly-zero gradient for that
slot, and under vanilla Adam (no weight decay configured) a parameter with an
always-zero gradient never moves at all. The epoch's logged `log_vars` tuple still
prints all three values for continuity, with `log_vars[2]` staying frozen at exactly
its init value of `0.0` for the entire run once training starts -- a clear, verifiable
signal that this task is no longer adaptively weighted.

The existing per-epoch print block (BPR / Rating MSE / Social MSE / log_vars) requires
no changes -- it already logs the three loss components separately every 10 epochs (and
epoch 1), satisfying the "log separate loss components" requirement as-is.

## Out of scope

- The other four engine files (`lightgcn_engine.py`, `trust_svd_engine.py`,
  `funk_svd_engine.py`, `sasrec_engine.py`) — untouched, still frozen.
- The three existing arena scripts (`unified_arena/`, `academic_sandbox/`,
  `filmtrust_arena/`) and `pipeline/run_pipeline.py` — untouched, still frozen.
- Re-running `grand_arena_runner.py`'s full `--all` sweep to refresh
  `models/grand_arena_results.md` with the new numbers -- a natural follow-up once this
  lands, but not part of this sub-project's own scope (this sub-project ships the
  capability and verifies it works correctly in isolation; a full benchmark re-run is a
  separate, later activity).
- Any change to `social_csr`'s shape, dtype, or symmetry guarantees -- denoising only
  removes edges (rows from `df_trust` before matrix construction); the resulting
  `social_csr` is still symmetric and zero-diagonal by construction, exactly as before.
- Hyperparameter search/tuning for `denoise_jaccard_threshold` or `social_loss_weight`
  beyond the stated defaults -- this sub-project ships the mechanism, not a tuned value.

## Verification plan

No pytest framework — direct script execution with documented exact expected output,
consistent with every other module in this codebase.

1. **Homophily filter, synthetic correctness check**: construct a small synthetic
   `train_csr` and `df_trust` with hand-computable Jaccard values (a high-overlap pair
   that survives a threshold, a zero-overlap pair that gets dropped), confirm
   `denoise_social_edges`'s output matches the hand-computed expected edge set exactly,
   and confirm the logged prune count matches.
2. **Homophily filter, real-data regression check**: re-run the existing Ciao/Yelp/
   FilmTrust real-data equivalence script (from sub-project 2/3/4, unchanged) -- since
   `denoise_social_graph` is now `True` by default for Ciao/Yelp, this run's `social_csr.nnz`
   for those two datasets is EXPECTED to differ from all prior sub-projects' recorded
   numbers (fewer edges, by design) -- confirm the script still completes with no error
   and report the real before/after edge counts; FilmTrust (denoising still `False`)
   must remain byte-identical to its sub-project-4 baseline.
3. **`SocialLightGCNEngine`, real training smoke test**: train against FilmTrust's real
   data (small, fast) with the new default `social_loss_weight=0.01`, confirm training
   completes with no traceback, the per-epoch log shows `Social MSE` converging
   independently of `BPR`/`Rating MSE`, and `log_vars[2]` stays at exactly `0.0` across
   every logged epoch (confirming it receives no gradient and is no longer adaptively
   driven) while `log_vars[0]`/`log_vars[1]` continue to move as before.
4. **End-to-end via `DatasetFactory`**: `DatasetFactory.create("ciao").load()` ->
   confirm `social_csr.nnz` is strictly less than Ciao's sub-project-4 baseline edge
   count (66,232) -- proof the filter is wired in and load-bearing for the one real
   dataset where its effect is easiest to sanity-check end-to-end.
