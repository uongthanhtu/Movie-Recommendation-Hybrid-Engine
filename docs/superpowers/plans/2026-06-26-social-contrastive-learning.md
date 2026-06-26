# Social Contrastive Learning (SCL) Paradigm Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SocialLightGCNEngine`'s Late-Fusion Attention Gate (sub-project 7) with a Social Contrastive Learning (InfoNCE) paradigm: the social branch becomes a purely auxiliary self-supervised regularizer that pulls `E_user_cf` toward social topology via gradient alone, while the CF embedding is used directly (unfused) for BPR ranking.

**Architecture:** One cohesive rewrite of `pipeline/engines/social_lightgcn_engine.py` (model class + engine class are tightly coupled — `fit()` depends on the model's new 3-tuple return, and the InfoNCE loss depends on both branches' outputs). Single task, verified by a real FilmTrust training smoke test showing both BPR and SSL losses descending.

**Tech Stack:** Python, `numpy`, `scipy.sparse`, `torch`, `torch.nn`, `torch.nn.functional`.

## Global Constraints

- This is the only engine file any sub-project in this initiative is authorized to touch: `pipeline/engines/social_lightgcn_engine.py`. The other four engine files, the three existing arena scripts, and `pipeline/run_pipeline.py` remain frozen.
- **Verified against the actual current file** (post sub-project 7, 374 lines): `SocialLightGCNModel` currently has `user_emb_cf`, `item_emb_cf`, `user_emb_social`, and a gate `W_att`; `forward(adj_ui, adj_iu, adj_social)` returns `(E_user_final, E_item_cf, alpha)`. This plan removes `W_att`/`alpha`/`E_user_final` entirely and changes `forward()`'s return to `(E_user_cf, E_item_cf, E_user_social)`.
- **The Attention Gate is abandoned.** The CF branch's output (`E_user_cf`) is used directly and exclusively for BPR scoring and inference: `y_hat = dot(E_user_cf, E_item_cf)`. The social branch (`E_user_social`) never participates in any prediction computation — it exists solely to receive and supply gradient through the InfoNCE loss.
- **InfoNCE loss, in-batch negatives, no Python loops over users/batch.** For a batch's user IDs, `(E_user_cf[u], E_user_social[u])` is the positive pair; `(E_user_cf[u], E_user_social[v])` for `v != u` (in-batch) are negatives. Cosine similarity, temperature `tau`. Implemented via `F.normalize` + matrix multiply + `F.cross_entropy` against the diagonal — the standard fully-vectorized InfoNCE form, mathematically identical to the spec's `-log(exp(sim(pos)/tau) / sum(exp(sim(all)/tau)))` formula (cross-entropy's internal log-softmax IS that exact ratio).
- **Correctness fix (not a deviation from spec, a fix to an issue the spec's English description doesn't address): in-batch user IDs must be deduplicated via `torch.unique()` before building the similarity matrix.** `_sample_training_batch` samples users *with replacement*; FilmTrust has 1,642 users and the default `batch_size` is 2048, so duplicate user IDs within a batch are guaranteed (pigeonhole). Without deduplication, a duplicate occurrence of user `u` would be wrongly scored as a "negative" against itself, corrupting the loss. `torch.unique` is itself a vectorized op — this fix adds no Python loop.
- New constructor hyperparameters: `temperature: float = 0.2` (the InfoNCE `tau`), `ssl_weight: float = 0.05` (the InfoNCE loss's weight, `lambda_scl`, in the total loss). The existing `reg` parameter continues to serve as the L2 weight (`lambda_l2`) — no rename, matching every other engine's existing convention.
- Total loss: `loss_bpr + reg_loss + ssl_weight * loss_scl`. `reg_loss` stays scoped to `user_emb_cf`/`item_emb_cf` only (unchanged from sub-project 7 — the social embedding is still never L2-regularized directly, consistent prior precedent).
- `predict_rating`/`recommend_top_n`/`save_model`/`load_model` keep their exact existing shape, but now read from `E_user_cf`/`E_item_cf` (cached as `self._user_emb`/`self._item_emb`) instead of the old `E_user_final`/`E_item_cf` — simpler than before, since there's no fusion step left at inference time at all.
- Per-epoch logging (every 10 epochs + epoch 1, matching existing cadence) prints `BPR` and `SSL` losses separately (plus `Reg`, for continuity with the existing convention) — the user explicitly wants to see both descending.
- Constructor keeps `num_users`, `num_items`, `n_epochs`, `batch_size` as valid keyword arguments with sensible defaults (required by `pipeline/benchmarks/model_runner.py`'s `_scaled_kwargs()`, which only ever sets those two — confirmed unchanged from sub-project 7's verification, this rewrite doesn't touch anything `model_runner.py` depends on).
- No backward compatibility with sub-project 7's saved checkpoints required (different parameter set/shapes — `load_state_dict` will fail loudly on an old checkpoint, which is correct/expected, not a landmine).
- No new third-party dependencies — `torch.nn.functional.normalize` and `torch.nn.functional.cross_entropy` are both already-imported-module functions (`F` is already imported). No pytest framework in this repo — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.`.

---

### Task 1: Rewrite `SocialLightGCNEngine` to Social Contrastive Learning

**Files:**
- Modify (complete rewrite): `pipeline/engines/social_lightgcn_engine.py`

**Interfaces:**
- Produces: `SocialLightGCNModel.__init__(num_users, num_items, embedding_dim=64, num_layers=3)`, `SocialLightGCNModel.forward(adj_ui, adj_iu, adj_social) -> (E_user_cf, E_item_cf, E_user_social)`. `SocialLightGCNEngine.__init__(num_users, num_items, embedding_dim=64, num_layers=3, lr=1e-3, reg=1e-4, n_epochs=30, batch_size=2048, temperature=0.2, ssl_weight=0.05)`. `fit(data: Dict)`, `predict_rating(user_id, item_id) -> float`, `recommend_top_n(user_id, top_n=10) -> List[int]`, `save_model(path)`, `load_model(path)` — same signatures `pipeline/benchmarks/model_runner.py` already calls.

- [ ] **Step 1: Replace the entire file**

Replace the full contents of `pipeline/engines/social_lightgcn_engine.py` with:

```python
"""
Social-LightGCN Engine — Social Contrastive Learning (SCL) for recommendation
(PyTorch).

Propagates two fully independent embedding spaces -- one over the bipartite
user-item (CF) graph, one over the user-user (Social) graph -- exactly as the prior
Late-Fusion architecture (sub-project 7) did. The difference is what happens after
propagation: there is no longer any fusion step. The CF embedding (E_user_cf) is used
DIRECTLY and EXCLUSIVELY for BPR ranking and inference; the social embedding
(E_user_social) never participates in any prediction. Instead, the social branch acts
purely as a self-supervised regularizer: an InfoNCE contrastive loss pulls each user's
CF embedding toward their own social embedding (the positive pair) and away from other
users' social embeddings (in-batch negatives), forcing E_user_cf to indirectly absorb
social topology through gradient alone -- never through message-passing, since the
two graphs are never combined or co-propagated.

This sidesteps Late-Fusion's gate entirely (no per-user alpha to learn or interpret)
at the cost of losing the gate's per-user ability to suppress an unhelpful social
signal -- ssl_weight is a single global scalar, not an adaptive one.

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
    Social Contrastive Learning (SCL) PyTorch neural network.
    Propagates collaborative (user-item) and social (user-user) signals over two
    fully independent embedding spaces. Unlike the prior Late-Fusion architecture,
    there is no fusion step in forward() at all -- both branches' outputs are returned
    separately for the engine to combine into BPR + InfoNCE losses during training.
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

        # CF branch embeddings (bipartite user-item graph) -- used directly for BPR
        self.user_emb_cf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_cf = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb_cf.weight)
        nn.init.xavier_uniform_(self.item_emb_cf.weight)

        # Social branch embeddings (user-user graph only -- no items, no prediction
        # role; exists solely as the InfoNCE loss's "social view" of each user)
        self.user_emb_social = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb_social.weight)

    def forward(
        self,
        adj_ui: torch.Tensor,
        adj_iu: torch.Tensor,
        adj_social: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propagate the CF and Social branches independently. No fusion step -- the
        engine combines these via BPR (CF only) and InfoNCE (CF vs Social) losses.

        Args:
            adj_ui: Normalized user-item sparse interaction matrix [U, I]
            adj_iu: Normalized item-user sparse interaction matrix [I, U]
            adj_social: Normalized user-user sparse social trust matrix [U, U]

        Returns:
            (E_user_cf, E_item_cf, E_user_social)
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

        return E_user_cf, E_item_cf, E_user_social


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
    Social-LightGCN Engine using Social Contrastive Learning: the CF branch is used
    directly for BPR ranking; the social branch supplies gradient only, via an
    InfoNCE loss that pulls the CF embedding toward social topology.
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
        temperature: float = 0.2,
        ssl_weight: float = 0.05,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.ssl_weight = ssl_weight

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

        # Latent representations readout cache (E_user_cf, E_item_cf -- the social
        # embedding is never needed at inference time, only during training)
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def fit(self, data: Any) -> None:
        """
        Train via Social Contrastive Learning: propagate the CF and Social branches
        independently; optimize the CF branch directly through BPR + L2, and pull it
        toward social topology indirectly through an InfoNCE loss against the social
        branch (in-batch negatives, cosine similarity, temperature-scaled).

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
            epoch_ssl_loss = 0.0

            for _ in range(n_batches):
                users, pos_items, neg_items = _sample_training_batch(
                    self._interaction_csr, self.batch_size, rng
                )
                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)

                # Forward: propagate CF and Social branches independently
                E_user_cf, E_item_cf, E_user_social = self.model(
                    self.adj_ui, self.adj_iu, self.adj_social
                )

                u_emb = E_user_cf[users_t]
                pos_emb = E_item_cf[pos_t]
                neg_emb = E_item_cf[neg_t]

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

                reg_loss = self.reg * (
                    self.model.user_emb_cf.weight[users_t].norm(2).pow(2)
                    + self.model.item_emb_cf.weight[pos_t].norm(2).pow(2)
                    + self.model.item_emb_cf.weight[neg_t].norm(2).pow(2)
                ) / self.batch_size

                loss_scl = _info_nce_loss(
                    E_user_cf, E_user_social, users_t, self.temperature
                )

                loss_total = loss_bpr + reg_loss + self.ssl_weight * loss_scl

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_bpr_loss += loss_bpr.item()
                epoch_reg_loss += reg_loss.item()
                epoch_ssl_loss += loss_scl.item()

            avg_bpr = epoch_bpr_loss / n_batches
            avg_reg = epoch_reg_loss / n_batches
            avg_ssl = epoch_ssl_loss / n_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  SocialGCN Epoch {epoch + 1:2d}/{self.n_epochs} | "
                    f"BPR: {avg_bpr:.4f} | Reg: {avg_reg:.4f} | SSL: {avg_ssl:.4f}"
                )

        # Cache representations
        self._cache_embeddings()

    def _cache_embeddings(self) -> None:
        """Pre-compute and cache CF user (E_user_cf) and CF item (E_item_cf)
        embeddings for O(1) inference. The social embedding is training-only and is
        discarded here -- it has no role at inference time."""
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

- [ ] **Step 2: Real training smoke test on FilmTrust, both losses logged**

FilmTrust data should already be cached locally from prior sub-projects
(`data/filmtrust/`). If not present, this will download fresh (small, fast).

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine

ds = DatasetFactory.create('filmtrust').load()
engine = SocialLightGCNEngine(num_users=ds.num_users, num_items=ds.num_items, n_epochs=10)
engine.fit({'interaction_matrix': ds.train_csr, 'trust_matrix': ds.social_csr})
print('Check 1 (training completed with no traceback): PASS')
"
```
Expected output: FilmTrust's loader progress lines, then `SocialGCN Epoch  1/10 | BPR:
<finite> | Reg: <finite> | SSL: <finite>` and `SocialGCN Epoch 10/10 | BPR: <finite> |
Reg: <finite> | SSL: <finite>` (prints at epoch 1 and epoch 10 per the existing
cadence), then `Check 1 (training completed with no traceback): PASS`. No traceback,
no NaN. **BPR must be lower at epoch 10 than epoch 1** (confirms BPR is descending).
**SSL must also be lower at epoch 10 than epoch 1**, or close to it — InfoNCE loss on
a small number of unique users per batch has a known floor of `log(N)` where `N` is
the number of unique users in the batch (the loss of a uniform/random alignment), so
don't expect it to approach zero, but it must trend downward from its initial value,
not increase or NaN.

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

- [ ] **Step 4: Verify the in-batch deduplication fix is load-bearing**

This confirms the `torch.unique()` fix actually matters for this codebase's real
defaults (FilmTrust has 1,642 users, default `batch_size` is 2048 — duplicates are
guaranteed) and that `_info_nce_loss` handles it correctly.

Run:
```bash
PYTHONPATH=. py -3 -c "
import numpy as np
from pipeline.engines.social_lightgcn_engine import _sample_training_batch
import scipy.sparse as sp

rng = np.random.default_rng(42)
train_csr = sp.random(1642, 2071, density=0.01, format='csr', random_state=42)
train_csr.data[:] = 1.0
users, _, _ = _sample_training_batch(train_csr, 2048, rng)
n_total = len(users)
n_unique = len(np.unique(users))
assert n_unique < n_total, f'expected duplicate users in a batch of {n_total} drawn from only 1642 users, got {n_unique} unique (no duplicates -- the fix would be a no-op for this dataset)'
print(f'Check 4 (duplicate users confirmed present in a real-sized batch): PASS -- {n_total - n_unique} duplicate draws out of {n_total}, {n_unique} unique users')
"
```
Expected output: `Check 4 (duplicate users confirmed present in a real-sized batch):
PASS` with a real, nonzero duplicate count — confirming the deduplication fix in
`_info_nce_loss` is not a defensive no-op but actually triggers on this codebase's
real data shapes.

- [ ] **Step 5: Confirm `model_runner.py` needs zero changes**

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
print(f'Check 5 (model_runner.run_model works unmodified): PASS -- train_seconds={train_seconds:.1f}, sample_recs={recs}')
"
```
Expected output: training progress (using `model_runner`'s default `n_epochs=30` for
this small dataset; expect roughly 30-90 seconds based on prior runs), then `Check 5
(model_runner.run_model works unmodified): PASS` with a real `train_seconds` value,
no exception.

- [ ] **Step 6: Commit**

```bash
git add pipeline/engines/social_lightgcn_engine.py
git commit -m "refactor(engines): replace Late-Fusion gate with Social Contrastive Learning (InfoNCE)"
```
