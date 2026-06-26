# Late-Fusion with Learnable Attention Gating — Design Spec

Date: 2026-06-26

## Context: sub-project 7, a complete architectural teardown of Social-LightGCN

Sub-project 6 (Graph Denoising & Loss Tuning) shipped a homophily filter and a
fixed `social_loss_weight`, then a full benchmark re-run showed `Social-LightGCN`
still loses to vanilla `LightGCN` on every dataset (Ciao, Yelp, Epinions, FilmTrust,
ml-100k ablation) — on Ciao specifically, retaining *more* (but still mostly
low-quality) social edges made it slightly worse, not better. The diagnosis: the
current architecture's "Early Fusion" propagates messages over the social and
collaborative-filtering (CF) signals jointly at every layer (`alpha * e_u_CF +
(1-alpha) * e_u_Social`, recomputed at each of `num_layers` hops), which lets social
noise pollute the CF signal before either has had a chance to form a clean
representation on its own — signal pollution / over-smoothing.

This sub-project is a complete rewrite of `pipeline/engines/social_lightgcn_engine.py`
to a **Late-Fusion** architecture: train two fully independent embedding spaces (one
per graph), propagate each in isolation, and only combine them once, at the very end,
via a learned per-user gate. The goal is structural: with the two signals never
touching each other during propagation, the model can degrade gracefully to *exactly*
vanilla LightGCN (gate `alpha=1` for every user) if the social signal turns out to be
pure noise for a given dataset — guaranteeing it never structurally underperforms
LightGCN by more than the gate's own training noise.

## Scope note

This is the second sub-project (after sub-project 6) to modify
`pipeline/engines/social_lightgcn_engine.py` — still the only engine file this whole
initiative has authorized changes to. The other four engine files
(`lightgcn_engine.py`, `trust_svd_engine.py`, `funk_svd_engine.py`,
`sasrec_engine.py`), the three existing arena scripts, and `pipeline/run_pipeline.py`
remain frozen and untouched.

Sub-project 6's data-layer work (`denoise_social_graph`/`denoise_jaccard_threshold` in
`DatasetConfig`, `denoise_social_edges` in `loader_utils.py`, the wiring in
`ExplicitTrustLoader.load()`) is **untouched and orthogonal** — it operates on
`social_csr` before this engine ever sees it, and stays exactly as-is regardless of
which engine architecture consumes the result.

## Research findings (verified against real code, not guessed)

- **Read the current full file** (`pipeline/engines/social_lightgcn_engine.py`,
  485 lines post sub-project 6). Confirmed: single `user_embedding`/`item_embedding`
  pair, per-layer early-fusion gate (`W_att` applied inside the `for _ in
  range(num_layers)` loop), 3-parameter `log_vars` (now only `log_vars[0]`/`log_vars[1]`
  live, since sub-project 6 already disconnected `log_vars[2]` from the social loss),
  `user_bias`/`item_bias`/`global_mu` for rating-MSE prediction, `_sample_social_batch`
  for the social-reconstruction loss term.
- **Confirmed `predict_rating` is never called by the actual benchmark evaluation path.**
  Grepped `pipeline/benchmarks/` for `predict_rating` — zero matches.
  `pipeline/benchmarks/evaluation.py::evaluate_model` only calls `recommend_top_n`
  (confirmed during sub-project 5's design). This means removing the rating-MSE
  training signal entirely has zero effect on the benchmarked Recall@10/NDCG@10
  metrics — `predict_rating` only needs to satisfy the `BaseRecommenderEngine`
  interface contract (return *some* valid float), not produce a calibrated rating.
- **Confirmed `model_runner.py`'s actual interface dependency on this engine is narrow.**
  `run_model()` constructs `SocialLightGCNEngine(num_users=..., num_items=...,
  **kwargs)` where `kwargs` is populated by `_scaled_kwargs()`, which only ever sets
  `n_epochs` and `batch_size` (large-dataset auto-scaling from sub-project 5). No other
  constructor parameter (`embedding_dim`, `num_layers`, `lr`, `reg`,
  `social_loss_weight`) is touched by the orchestrator — they're free to change or be
  removed without breaking `model_runner.py`, as long as `num_users`, `num_items`,
  `n_epochs`, `batch_size` remain valid keyword arguments with sensible defaults. `fit()`
  must keep accepting `{"interaction_matrix": ..., "trust_matrix": ...}`.

## Decisions made (binding scope boundaries for this sub-project)

- **L2 weight-decay regularization is kept.** It's standard practice alongside BPR in
  virtually every embedding-based recommender (including every other engine in this
  codebase), not part of the multi-task scaffolding (`log_vars`, social MSE) being torn
  down. The loss becomes `loss_bpr + reg_loss`, not pure unregularized BPR.
- **`social_loss_weight` (added in sub-project 6) is removed entirely.** It weighted a
  social-reconstruction MSE loss that no longer exists in this architecture — the
  social branch is now trained purely through the BPR objective via the gate, with no
  separate social loss term at all. Keeping the parameter would be a misleading no-op.
- **`user_bias`/`item_bias`/`global_mu` (rating-MSE prediction machinery) are removed
  entirely**, along with the rating-MSE loss term. With no rating-MSE training signal,
  these would receive zero gradient and never move from their init values — the same
  dead-weight pattern flagged (non-blocking) for `log_vars[2]` last sub-project, but for
  three parameters instead of one, and this time avoidable outright since
  `predict_rating` isn't benchmarked. `predict_rating` is reimplemented as a clipped
  dot-product of the fused embeddings (`E_user_final` · `E_item_cf`, clipped to
  `[1.0, 5.0]`) — uncalibrated to a true 1-5 rating scale, but that has no effect since
  the method isn't exercised by the benchmark.
- **CF and Social branches share one `num_layers` constructor parameter** (today's
  existing parameter), not independent depth controls per branch. Not requested by the
  spec, and YAGNI — the over-smoothing risk this sub-project targets is already
  structurally avoided by separating the graphs, independent depth isn't needed to
  achieve that.
- **A new diagnostic: mean `alpha` (the gate's per-user attention weight) is logged
  alongside the BPR loss** at the same cadence as the existing epoch print (every 10
  epochs + epoch 1). This directly answers the architecture's own motivating question —
  is the gate actually leaning on the social branch for anyone, or just defaulting to
  `alpha≈1` (pure LightGCN) everywhere — without requiring a separate verification step.
- **No backward compatibility with old saved checkpoints.** `save_model`/`load_model`
  still round-trip via `state_dict()`, but a checkpoint saved by the old early-fusion
  architecture cannot be loaded by the new model (different parameter names/shapes).
  This is acceptable: these are train-fresh-per-benchmark-run engines, no persisted
  production checkpoint depends on the old format.

## Architecture

### `SocialLightGCNModel` (complete rewrite)

```python
class SocialLightGCNModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        self.user_emb_cf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_cf = nn.Embedding(num_items, embedding_dim)
        self.user_emb_social = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb_cf.weight)
        nn.init.xavier_uniform_(self.item_emb_cf.weight)
        nn.init.xavier_uniform_(self.user_emb_social.weight)

        # Late-Fusion Attention Gate (applied once, not per-layer)
        self.W_att = nn.Linear(embedding_dim * 2, 1)
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.zeros_(self.W_att.bias)

    def forward(self, adj_ui, adj_iu, adj_social):
        """
        Returns (E_user_final, E_item_cf, alpha) -- alpha returned for diagnostic
        logging in fit(), shape [num_users, 1].
        """
        # --- CF branch: standard LightGCN bipartite propagation ---
        u_cf = self.user_emb_cf.weight
        i_cf = self.item_emb_cf.weight
        u_cf_list, i_cf_list = [u_cf], [i_cf]
        for _ in range(self.num_layers):
            u_cf = torch.sparse.mm(adj_ui, i_cf)
            i_cf = torch.sparse.mm(adj_iu, u_cf)
            u_cf_list.append(u_cf)
            i_cf_list.append(i_cf)
        E_user_cf = torch.stack(u_cf_list, dim=0).mean(dim=0)
        E_item_cf = torch.stack(i_cf_list, dim=0).mean(dim=0)

        # --- Social branch: pure user-user graph convolution, no items ---
        u_social = self.user_emb_social.weight
        u_social_list = [u_social]
        for _ in range(self.num_layers):
            u_social = torch.sparse.mm(adj_social, u_social)
            u_social_list.append(u_social)
        E_user_social = torch.stack(u_social_list, dim=0).mean(dim=0)

        # --- Late-Fusion Attention Gate ---
        cat_feats = torch.cat([E_user_cf, E_user_social], dim=1)
        alpha = torch.sigmoid(self.W_att(cat_feats))
        E_user_final = alpha * E_user_cf + (1.0 - alpha) * E_user_social

        return E_user_final, E_item_cf, alpha
```

The CF branch's loop body is byte-for-byte the existing LightGCN propagation pattern
(`torch.sparse.mm(adj_ui, i_emb)` / `torch.sparse.mm(adj_iu, u_emb)`), just renamed and
with the social term removed from each step. The social branch is new: a single-matrix
homogeneous-graph convolution (`adj_social @ u_social`, repeated, layer-averaged) — no
analogous existing code to reuse since the social graph never had item nodes.

### `SocialLightGCNEngine` (constructor + `fit()` rewrite)

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
    # social_loss_weight removed -- no social loss term left to weight.
    ...
```

`fit()` keeps its existing `adj_ui`/`adj_iu` construction (bipartite symmetric
normalization, unchanged) and its existing `adj_uv` construction logic, renamed
`adj_social` (user-user symmetric normalization, unchanged math, same as today's
`T_norm` computation) — both adjacency matrices are built once per `fit()` call and
passed to `forward()` every batch, exactly as today. They are never combined into one
matrix, satisfying the spec's explicit requirement.

Per-batch training step:
```python
users, pos_items, neg_items = _sample_training_batch(self._interaction_csr, self.batch_size, rng)
# (no social batch sampling -- _sample_social_batch is deleted)

user_emb, item_emb, alpha = self.model(self.adj_ui, self.adj_iu, self.adj_social)

u_emb = user_emb[users_t]
pos_emb = item_emb[pos_t]
neg_emb = item_emb[neg_t]

pos_scores = (u_emb * pos_emb).sum(dim=1)
neg_scores = (u_emb * neg_emb).sum(dim=1)
loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

reg_loss = self.reg * (
    self.model.user_emb_cf.weight[users_t].norm(2).pow(2)
    + self.model.item_emb_cf.weight[pos_t].norm(2).pow(2)
    + self.model.item_emb_cf.weight[neg_t].norm(2).pow(2)
) / self.batch_size

loss_total = loss_bpr + reg_loss
```

Note `loss_bpr` drops the `log_vars[0]`-based squashed-logits scaling entirely (there is
no `log_vars` left) — it becomes the textbook BPR loss, exactly
`-mean(logsigmoid(pos_scores - neg_scores))`. `reg_loss` is computed only over the CF
embeddings (`user_emb_cf`/`item_emb_cf`), matching today's existing reg term's scope
(it never regularized the social embeddings either, since today's `user_embedding` is
the single shared table the reg term draws `users_t`/`pos_t`/`neg_t` rows from for the
*CF* role; the new architecture's `user_emb_social` is regularized implicitly only
through the L2 norm structure of gradient descent + Adam, not an explicit penalty term
— consistent with not over-specifying beyond what the spec asked for).

Per-epoch logging (every 10 epochs + epoch 1, matching existing cadence):
```python
print(
    f"  SocialGCN Epoch {epoch + 1:2d}/{self.n_epochs} | "
    f"BPR: {avg_bpr:.4f} | Reg: {avg_reg:.4f} | mean(alpha): {avg_alpha:.4f}"
)
```
`avg_alpha` is the mean of `alpha.mean().item()` across the epoch's batches — a single
scalar summarizing how much weight the gate places on the CF branch on average (closer
to 1.0 = behaving like pure LightGCN; closer to 0.0 = leaning on the social branch).

### `predict_rating` / `recommend_top_n` (interface preservation)

```python
def predict_rating(self, user_id: int, item_id: int) -> float:
    if user_id >= self.num_users or item_id >= self.num_items:
        return 3.5
    with torch.no_grad():
        pred = torch.dot(self._user_emb[user_id], self._item_emb[item_id]).item()
    return float(np.clip(pred, 1.0, 5.0))
```
(`self._user_emb` is the cached `E_user_final`, `self._item_emb` is the cached
`E_item_cf` — same caching pattern as today via `_cache_embeddings()`, just sourced from
the new `forward()`'s first two return values.)

`recommend_top_n` is unchanged in shape: same seen-item masking via
`self._interaction_csr[user_id].indices`, same `torch.topk` over
`torch.matmul(self._item_emb, user_vec)`.

## Out of scope

- Sub-project 6's data-layer homophily filter — untouched, orthogonal.
- The other four engine files, the three existing arena scripts, `run_pipeline.py` —
  frozen, untouched.
- Re-running `grand_arena_runner.py --all` to refresh `models/grand_arena_results.md`
  with this new architecture's real numbers — a natural follow-up, but this
  sub-project's own scope is the rewrite + a FilmTrust smoke test, not a full
  benchmark re-run (the user's own Step 4 in the originating request asks only for a
  quick smoke test, not a full sweep).
- Any change to `ArenaDataset`, `DatasetFactory`, `model_runner.py`, or
  `grand_arena_runner.py` — this rewrite is designed to require zero changes to any of
  them (confirmed via the model_runner.py interface-dependency check above).
- Backward-compatible checkpoint loading for old early-fusion `state_dict()` saves.

## Verification plan

No pytest framework — direct script execution with documented exact expected output.

1. **Real training smoke test on FilmTrust** (small, fast, already cached locally):
   construct the engine, call `fit()` with FilmTrust's real `train_csr`/`social_csr`,
   confirm training completes for a few epochs with no traceback (no CUDA/shape
   mismatch errors across the dual-graph forward pass), the per-epoch log shows `BPR`,
   `Reg`, and `mean(alpha)` printing real, finite (non-NaN) numbers, and `mean(alpha)`
   is a valid probability in `[0, 1]`.
2. **Interface compliance check**: after `fit()`, call `recommend_top_n(user_id,
   top_n=10)` and `predict_rating(user_id, item_id)` for a real user/item pair from
   FilmTrust, confirm both return values of the correct type/shape (`List[int]` of
   length <= 10; `float` in `[1.0, 5.0]`) with no exception.
3. **Confirm `model_runner.py` still constructs the engine correctly**: directly call
   `model_runner.run_model("social_lightgcn", dataset)` against the loaded FilmTrust
   `ArenaDataset` (not the full `grand_arena_runner.py --all` sweep — just this one
   model/dataset pair, to prove the orchestrator's existing translation code needs zero
   changes), confirm it returns a fitted engine and a real `train_seconds` float with no
   exception.
