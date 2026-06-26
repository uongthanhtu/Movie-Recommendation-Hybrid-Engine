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
        temperature: float = 0.5,
        ssl_weight: float = 0.005,
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
