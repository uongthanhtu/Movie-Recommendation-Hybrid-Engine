# Late-Fusion Attention Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely rewrite `pipeline/engines/social_lightgcn_engine.py` to a Late-Fusion architecture: two independent embedding spaces (CF, Social), each propagated over its own graph in isolation, fused exactly once via a learned per-user attention gate, trained purely via BPR + L2.

**Architecture:** One cohesive rewrite (model class + sampling helper + engine class are tightly coupled — the engine's `fit()` depends on the model's new 3-tuple `forward()` return, and `predict_rating`/`recommend_top_n` depend on the new cached embedding shapes). A single task replaces the whole file, verified by a real FilmTrust training smoke test plus a direct `model_runner.py` integration check.

**Tech Stack:** Python, `numpy`, `scipy.sparse`, `torch`, `torch.nn`, `torch.nn.functional`.

## Global Constraints

- This is the only engine file this sub-project (and the only one besides it, sub-project 6) is authorized to touch: `pipeline/engines/social_lightgcn_engine.py`. The other four engine files, the three existing arena scripts, and `pipeline/run_pipeline.py` remain frozen.
- `SocialLightGCNModel` maintains three embedding tables: `user_emb_cf`, `item_emb_cf` (CF branch only), `user_emb_social` (Social branch only, no items). No `log_vars`, no `user_bias`/`item_bias`/`global_mu` — all removed.
- `adj_ui`/`adj_iu` (CF, bipartite) and `adj_social` (Social, user-user) are built independently in `fit()` and are NEVER combined into one matrix.
- CF branch: standard LightGCN bipartite propagation (alternating `adj_ui`/`adj_iu`), layer-averaged readout -> `E_user_cf`, `E_item_cf`. Social branch: pure user-user graph convolution (`adj_social @ u_social`, repeated), layer-averaged readout -> `E_user_social`. Both branches share the existing `num_layers` constructor parameter — no independent per-branch depth controls.
- Attention gate applied exactly once (late fusion, not per-layer): `alpha = sigmoid(W_att(cat([E_user_cf, E_user_social], dim=1)))`; `E_user_final = alpha * E_user_cf + (1 - alpha) * E_user_social`. `forward()` returns `(E_user_final, E_item_cf, alpha)`.
- Loss is `loss_bpr + reg_loss` only — plain BPR (`-mean(logsigmoid(pos_scores - neg_scores))`, no log-variance scaling) plus the existing-style L2 weight decay scoped to `user_emb_cf`/`item_emb_cf` only. No social-reconstruction loss, no `social_loss_weight` parameter (removed — nothing left to weight). `_sample_social_batch` is deleted.
- `predict_rating` is reimplemented as a clipped dot-product of the cached `E_user_final`/`E_item_cf` (`np.clip(pred, 1.0, 5.0)`) — uncalibrated to a true rating scale, but confirmed unused by the actual benchmark evaluation (`pipeline/benchmarks/evaluation.py` only calls `recommend_top_n`), so this has zero effect on Recall/NDCG results.
- `recommend_top_n`, `save_model`, `load_model` keep their exact existing shape (same seen-item masking, same `state_dict()` round-trip) — only the embeddings feeding them change source.
- Constructor keeps `num_users`, `num_items`, `n_epochs`, `batch_size` as valid keyword arguments with sensible defaults (required by `pipeline/benchmarks/model_runner.py`'s `_scaled_kwargs()`, which only ever sets those two). `embedding_dim`, `num_layers`, `lr`, `reg` stay as before; `social_loss_weight` is removed.
- Per-epoch logging (every 10 epochs + epoch 1, matching existing cadence) prints `BPR`, `Reg`, and `mean(alpha)` — the new diagnostic showing how much weight the gate places on the CF branch on average.
- No backward compatibility with old (early-fusion) saved checkpoints required.
- No new third-party dependencies. No pytest framework in this repo — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.` (plain `python` is not on PATH in this environment).

---

### Task 1: Rewrite `SocialLightGCNEngine` to Late-Fusion Attention Gating

**Files:**
- Modify (complete rewrite): `pipeline/engines/social_lightgcn_engine.py`

**Interfaces:**
- Produces: `SocialLightGCNModel.__init__(num_users, num_items, embedding_dim=64, num_layers=3)`, `SocialLightGCNModel.forward(adj_ui, adj_iu, adj_social) -> (E_user_final, E_item_cf, alpha)`. `SocialLightGCNEngine.__init__(num_users, num_items, embedding_dim=64, num_layers=3, lr=1e-3, reg=1e-4, n_epochs=30, batch_size=2048)`. `fit(data: Dict)`, `predict_rating(user_id, item_id) -> float`, `recommend_top_n(user_id, top_n=10) -> List[int]`, `save_model(path)`, `load_model(path)` — same signatures `pipeline/benchmarks/model_runner.py` already calls.

- [ ] **Step 1: Replace the entire file**

Replace the full contents of `pipeline/engines/social_lightgcn_engine.py` with:

```python
"""
Social-LightGCN Engine — Late-Fusion Social-Aware Graph Convolutional Network for
recommendation (PyTorch).

Trains two fully independent embedding spaces -- one propagated over the bipartite
user-item (CF) graph, one propagated over the user-user (Social) graph -- and fuses
them exactly once, after propagation, via a learned per-user Attention Gate
(Late Fusion). This replaces the prior per-layer Early-Fusion architecture (sub-project
6 and earlier), which mixed CF and Social signals at every propagation hop and let
social noise pollute the collaborative-filtering signal before either had formed a
clean representation on its own. With the two graphs never touching during
propagation, the gate can degrade to pure LightGCN (alpha=1 for every user) whenever
the social signal is unhelpful for a given dataset.

Optimized purely via the standard BPR ranking loss plus L2 weight decay -- no
multi-task loss, no learned uncertainty weights, no social-reconstruction term.

Technical contract inherited from BaseRecommenderEngine.
"""
from typing import Any, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.engines.base_engine import BaseRecommenderEngine


# ======================================================================
# PyTorch Neural Network
# ======================================================================

class SocialLightGCNModel(nn.Module):
    """
    Late-Fusion Social-LightGCN PyTorch neural network.
    Propagates collaborative (user-item) and social (user-user) signals over two
    fully independent embedding spaces, then fuses the resulting user representations
    exactly once via a learned per-user Attention Gate.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers

        # CF branch embeddings (bipartite user-item graph)
        self.user_emb_cf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_cf = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb_cf.weight)
        nn.init.xavier_uniform_(self.item_emb_cf.weight)

        # Social branch embeddings (user-user graph only -- no items)
        self.user_emb_social = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb_social.weight)

        # Late-Fusion Attention Gate (applied once, after both branches propagate --
        # NOT per-layer, unlike the prior early-fusion architecture)
        self.W_att = nn.Linear(embedding_dim * 2, 1)
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.zeros_(self.W_att.bias)

    def forward(
        self,
        adj_ui: torch.Tensor,
        adj_iu: torch.Tensor,
        adj_social: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propagate the CF and Social branches independently, then fuse once.

        Args:
            adj_ui: Normalized user-item sparse interaction matrix [U, I]
            adj_iu: Normalized item-user sparse interaction matrix [I, U]
            adj_social: Normalized user-user sparse social trust matrix [U, U]

        Returns:
            (E_user_final, E_item_cf, alpha) -- alpha has shape [U, 1], returned for
            diagnostic logging (mean alpha shows how much weight the gate places on
            the CF branch on average).
        """
        # --- CF branch: standard LightGCN bipartite propagation ---
        u_cf = self.user_emb_cf.weight
        i_cf = self.item_emb_cf.weight
        u_cf_list = [u_cf]
        i_cf_list = [i_cf]
        for _ in range(self.num_layers):
            u_cf = torch.sparse.mm(adj_ui, i_cf)
            i_cf = torch.sparse.mm(adj_iu, u_cf)
            u_cf_list.append(u_cf)
            i_cf_list.append(i_cf)
        E_user_cf = torch.stack(u_cf_list, dim=0).mean(dim=0)
        E_item_cf = torch.stack(i_cf_list, dim=0).mean(dim=0)

        # --- Social branch: pure user-user graph convolution, no items involved ---
        u_social = self.user_emb_social.weight
        u_social_list = [u_social]
        for _ in range(self.num_layers):
            u_social = torch.sparse.mm(adj_social, u_social)
            u_social_list.append(u_social)
        E_user_social = torch.stack(u_social_list, dim=0).mean(dim=0)

        # --- Late-Fusion Attention Gate (applied exactly once) ---
        cat_feats = torch.cat([E_user_cf, E_user_social], dim=1)  # [U, d*2]
        alpha = torch.sigmoid(self.W_att(cat_feats))              # [U, 1]
        E_user_final = alpha * E_user_cf + (1.0 - alpha) * E_user_social

        return E_user_final, E_item_cf, alpha


# ======================================================================
# Memory-Efficient CSR Sampling Helper
# ======================================================================

def _sample_training_batch(
    interaction_csr: sp.csr_matrix,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (user, positive_item, negative_item) BPR triplets.
    Uses direct index-pointer lookups to maximize performance.
    """
    num_users, num_items = interaction_csr.shape
    users = np.zeros(batch_size, dtype=np.int64)
    pos_items = np.zeros(batch_size, dtype=np.int64)
    neg_items = np.zeros(batch_size, dtype=np.int64)

    indices_arr = interaction_csr.indices
    indptr_arr = interaction_csr.indptr

    for idx in range(batch_size):
        u = rng.integers(0, num_users)
        start, end = indptr_arr[u], indptr_arr[u + 1]
        while start == end:
            u = rng.integers(0, num_users)
            start, end = indptr_arr[u], indptr_arr[u + 1]

        pos_idx = rng.integers(start, end)
        pos_i = indices_arr[pos_idx]

        pos_list = indices_arr[start:end]

        neg_j = rng.integers(0, num_items)
        while neg_j in pos_list:
            neg_j = rng.integers(0, num_items)

        users[idx] = u
        pos_items[idx] = pos_i
        neg_items[idx] = neg_j

    return users, pos_items, neg_items


def _sparse_scipy_to_torch(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert a SciPy CSR matrix to a PyTorch sparse COO tensor on specified device."""
    coo = mat.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)


# ======================================================================
# Recommender Engine Implementation
# ======================================================================

class SocialLightGCNEngine(BaseRecommenderEngine):
    """
    Social-LightGCN Engine using Late-Fusion Attention Gating: two independent
    embedding spaces (CF, Social), fused once per forward pass, trained purely via
    BPR + L2 weight decay.
    """

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

        self.model = SocialLightGCNModel(
            num_users, num_items, embedding_dim, num_layers
        ).to(self.device)

        # CF and Social adjacency matrices -- built independently in fit(), never
        # combined into one matrix.
        self.adj_ui: torch.Tensor = torch.empty(0)
        self.adj_iu: torch.Tensor = torch.empty(0)
        self.adj_social: torch.Tensor = torch.empty(0)

        # Raw user-item interaction matrix
        self._interaction_csr: sp.csr_matrix = sp.csr_matrix((0, 0))

        # Latent representations readout cache (E_user_final, E_item_cf)
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def fit(self, data: Any) -> None:
        """
        Train via Late-Fusion: propagate the CF and Social branches independently,
        fuse once per forward pass via the learned gate, optimize purely through the
        BPR ranking objective plus L2 weight decay.

        Args:
            data: Dict containing:
                  - "interaction_matrix": sp.csr_matrix (rating matrix [U, I])
                  - "trust_matrix":       sp.csr_matrix (trust matrix [U, U])
        """
        interaction_mat: sp.csr_matrix = data["interaction_matrix"]
        trust_mat: sp.csr_matrix = data["trust_matrix"]

        self._interaction_csr = interaction_mat.copy()

        # Binarize rating interactions for structural propagation in graph
        R_binary = interaction_mat.copy()
        R_binary.data = np.ones_like(R_binary.data, dtype=np.float32)

        # CF graph: bipartite symmetric normalization
        user_degrees = np.array(R_binary.sum(axis=1)).flatten()
        item_degrees = np.array(R_binary.sum(axis=0)).flatten()

        with np.errstate(divide="ignore"):
            d_u_inv = np.power(user_degrees, -0.5)
            d_i_inv = np.power(item_degrees, -0.5)
        d_u_inv[np.isinf(d_u_inv)] = 0.0
        d_i_inv[np.isinf(d_i_inv)] = 0.0

        D_u_inv = sp.diags(d_u_inv)
        D_i_inv = sp.diags(d_i_inv)

        R_norm = D_u_inv.dot(R_binary).dot(D_i_inv)

        self.adj_ui = _sparse_scipy_to_torch(R_norm.tocsr(), self.device)
        self.adj_iu = _sparse_scipy_to_torch(R_norm.T.tocsr(), self.device)

        # Social graph: user-user symmetric normalization -- kept entirely separate
        # from the CF graph above, never combined into one matrix.
        rowsum_social = np.array(trust_mat.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_s_inv = np.power(rowsum_social, -0.5)
        d_s_inv[np.isinf(d_s_inv)] = 0.0
        D_s_inv = sp.diags(d_s_inv)

        T_norm = D_s_inv.dot(trust_mat).dot(D_s_inv)
        self.adj_social = _sparse_scipy_to_torch(T_norm.tocsr(), self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_interactions = int(self._interaction_csr.nnz)
        n_batches = max(n_interactions // self.batch_size, 1)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_bpr_loss = 0.0
            epoch_reg_loss = 0.0
            epoch_alpha = 0.0

            for _ in range(n_batches):
                users, pos_items, neg_items = _sample_training_batch(
                    self._interaction_csr, self.batch_size, rng
                )
                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)

                # Forward: propagate CF and Social branches independently, fuse once
                user_emb, item_emb, alpha = self.model(
                    self.adj_ui, self.adj_iu, self.adj_social
                )

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

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_bpr_loss += loss_bpr.item()
                epoch_reg_loss += reg_loss.item()
                epoch_alpha += alpha.mean().item()

            avg_bpr = epoch_bpr_loss / n_batches
            avg_reg = epoch_reg_loss / n_batches
            avg_alpha = epoch_alpha / n_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  SocialGCN Epoch {epoch + 1:2d}/{self.n_epochs} | "
                    f"BPR: {avg_bpr:.4f} | Reg: {avg_reg:.4f} | mean(alpha): {avg_alpha:.4f}"
                )

        # Cache representations
        self._cache_embeddings()

    def _cache_embeddings(self) -> None:
        """Pre-compute and cache fused user (E_user_final) and CF item (E_item_cf)
        embeddings for O(1) inference."""
        self.model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb, _ = self.model(
                self.adj_ui, self.adj_iu, self.adj_social
            )

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id >= self.num_users or item_id >= self.num_items:
            return 3.5

        with torch.no_grad():
            user_vec = self._user_emb[user_id]
            item_vec = self._item_emb[item_id]
            pred = torch.dot(user_vec, item_vec).item()

        return float(np.clip(pred, 1.0, 5.0))

    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        if user_id >= self.num_users:
            return list(range(min(top_n, self.num_items)))

        with torch.no_grad():
            user_vec = self._user_emb[user_id]
            scores = torch.matmul(self._item_emb, user_vec)  # [num_items]

            # Exclude watched items
            seen = self._interaction_csr[user_id].indices
            if len(seen) > 0:
                scores[torch.LongTensor(seen).to(self.device)] = float("-inf")

            top_items = torch.topk(scores, min(top_n, self.num_items)).indices

        return top_items.cpu().numpy().tolist()

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._cache_embeddings()
```

- [ ] **Step 2: Real training smoke test on FilmTrust**

FilmTrust data should already be cached locally from prior sub-projects
(`data/filmtrust/`). If not present, this will download fresh (small, fast).

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine
import math

ds = DatasetFactory.create('filmtrust').load()
engine = SocialLightGCNEngine(num_users=ds.num_users, num_items=ds.num_items, n_epochs=5)
engine.fit({'interaction_matrix': ds.train_csr, 'trust_matrix': ds.social_csr})
print('Check 1 (training completed with no traceback): PASS')
"
```
Expected output: FilmTrust's loader progress lines, then `SocialGCN Epoch  1/5 | BPR:
<finite number> | Reg: <finite number> | mean(alpha): <number in [0,1]>` (prints at
epoch 1 per the `epoch == 0` condition; with `n_epochs=5` no other epoch hits the
`% 10 == 0` condition, so only one epoch line is expected -- this is fine, consistent
with the existing logging cadence), then `Check 1 (training completed with no
traceback): PASS`. No traceback, no NaN, `mean(alpha)` must print as a real number
between 0 and 1 inclusive.

- [ ] **Step 3: Interface compliance check**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine

ds = DatasetFactory.create('filmtrust').load()
engine = SocialLightGCNEngine(num_users=ds.num_users, num_items=ds.num_items, n_epochs=2)
engine.fit({'interaction_matrix': ds.train_csr, 'trust_matrix': ds.social_csr})

recs = engine.recommend_top_n(0, top_n=10)
assert isinstance(recs, list) and len(recs) <= 10 and all(isinstance(x, int) for x in recs)
print(f'Check 2 (recommend_top_n returns a valid List[int]): PASS -- sample_recs={recs}')

rating = engine.predict_rating(0, 0)
assert isinstance(rating, float) and 1.0 <= rating <= 5.0
print(f'Check 3 (predict_rating returns a valid float in [1,5]): PASS -- rating={rating}')
"
```
Expected output: training progress, then both `Check 2` and `Check 3` lines printing
`PASS` with real observed values, no exception.

- [ ] **Step 4: Confirm `model_runner.py` needs zero changes**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.benchmarks import model_runner

ds = DatasetFactory.create('filmtrust').load()
engine, train_seconds = model_runner.run_model('social_lightgcn', ds)
assert isinstance(train_seconds, float) and train_seconds > 0
recs = engine.recommend_top_n(0, top_n=10)
assert isinstance(recs, list)
print(f'Check 4 (model_runner.run_model works unmodified): PASS -- train_seconds={train_seconds:.1f}, sample_recs={recs}')
"
```
Expected output: training progress (using `model_runner`'s default `n_epochs=30` for
this small dataset, so this will take roughly as long as the existing FilmTrust
benchmark runs have — about 30-60 seconds), then `Check 4 (model_runner.run_model
works unmodified): PASS` with a real `train_seconds` value, no exception. This proves
the orchestrator's existing translation code requires zero changes for the new
architecture.

- [ ] **Step 5: Commit**

```bash
git add pipeline/engines/social_lightgcn_engine.py
git commit -m "refactor(engines): rewrite Social-LightGCN with Late-Fusion Attention Gating"
```
