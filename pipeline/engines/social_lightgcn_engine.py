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
import os
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
    Social-LightGCN PyTorch neural network.
    Combines collaborative signals (from user-item interaction graph)
    with social signals (from user-user trust network) via an Adaptive Attention Gate
    at each propagation layer (Early Fusion).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Core embeddings (layer 0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Self-adaptive multi-task uncertainty log-variances (Kendall et al.)
        self.log_vars = nn.Parameter(torch.zeros(3))

        # Adaptive Attention Gate projection matrix
        self.W_att = nn.Linear(embedding_dim * 2, 1)
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.zeros_(self.W_att.bias)

        # Rating prediction biases and global average
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_mu = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.global_mu.data.fill_(3.5)

    def forward(
        self,
        adj_ui: torch.Tensor,
        adj_iu: torch.Tensor,
        adj_uv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run early-fusion propagation across bipartite and social graphs.

        Args:
            adj_ui: Normalized user-item sparse interaction matrix [U, I]
            adj_iu: Normalized item-user sparse interaction matrix [I, U]
            adj_uv: Normalized user-user sparse social trust matrix [U, U]

        Returns:
            user_embeddings, item_embeddings combined over all layers.
        """
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        u_emb_list = [u_emb]
        i_emb_list = [i_emb]

        for _ in range(self.num_layers):
            # Propagation Flow 1: Collaborative Signal (U <- I)
            e_u_CF = torch.sparse.mm(adj_ui, i_emb)

            # Propagation Flow 2: Social Signal (U <- U)
            e_u_Social = torch.sparse.mm(adj_uv, u_emb)

            # Compute user-specific alpha via Adaptive Attention Gate
            cat_feats = torch.cat([e_u_CF, e_u_Social], dim=1)  # [U, d*2]
            alpha = torch.sigmoid(self.W_att(cat_feats))        # [U, 1]

            # Early Fusion
            u_emb = alpha * e_u_CF + (1.0 - alpha) * e_u_Social

            # Propagation for Items: Standard bipartite graph aggregation
            i_emb = torch.sparse.mm(adj_iu, u_emb)

            u_emb_list.append(u_emb)
            i_emb_list.append(i_emb)

        # Readout Layer Combination (average over all layers)
        final_user_emb = torch.stack(u_emb_list, dim=0).mean(dim=0)
        final_item_emb = torch.stack(i_emb_list, dim=0).mean(dim=0)

        return final_user_emb, final_item_emb


# ======================================================================
# Memory-Efficient CSR Sampling Helpers
# ======================================================================

def _sample_training_batch(
    interaction_csr: sp.csr_matrix,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (user, positive_item, negative_item) triplets and positive rating values.
    Uses direct index-pointer lookups to maximize performance.
    """
    num_users, num_items = interaction_csr.shape
    users = np.zeros(batch_size, dtype=np.int64)
    pos_items = np.zeros(batch_size, dtype=np.int64)
    neg_items = np.zeros(batch_size, dtype=np.int64)
    pos_ratings = np.zeros(batch_size, dtype=np.float32)

    indices_arr = interaction_csr.indices
    indptr_arr = interaction_csr.indptr
    data_arr = interaction_csr.data

    for idx in range(batch_size):
        u = rng.integers(0, num_users)
        start, end = indptr_arr[u], indptr_arr[u+1]
        while start == end:
            u = rng.integers(0, num_users)
            start, end = indptr_arr[u], indptr_arr[u+1]

        # Select positive item
        pos_idx = rng.integers(start, end)
        pos_i = indices_arr[pos_idx]
        r = data_arr[pos_idx]

        pos_list = indices_arr[start:end]

        # Select negative item (not interacted)
        neg_j = rng.integers(0, num_items)
        while neg_j in pos_list:
            neg_j = rng.integers(0, num_items)

        users[idx] = u
        pos_items[idx] = pos_i
        neg_items[idx] = neg_j
        pos_ratings[idx] = r

    return users, pos_items, neg_items, pos_ratings


def _sample_social_batch(
    trust_csr: sp.csr_matrix,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample pairs of users (u, v) and their trust coefficients.
    Returns 50% positive (trusted) links and 50% negative (unlinked) pairs.
    """
    num_users = trust_csr.shape[0]
    users_u = np.zeros(batch_size, dtype=np.int64)
    users_v = np.zeros(batch_size, dtype=np.int64)
    trust_vals = np.zeros(batch_size, dtype=np.float32)

    indices_arr = trust_csr.indices
    indptr_arr = trust_csr.indptr
    data_arr = trust_csr.data

    half_batch = batch_size // 2

    # 1. Sample Positive Pairs
    for idx in range(half_batch):
        u = rng.integers(0, num_users)
        start, end = indptr_arr[u], indptr_arr[u+1]
        while start == end:
            u = rng.integers(0, num_users)
            start, end = indptr_arr[u], indptr_arr[u+1]

        pos_idx = rng.integers(start, end)
        v = indices_arr[pos_idx]
        val = data_arr[pos_idx]

        users_u[idx] = u
        users_v[idx] = v
        trust_vals[idx] = val

    # 2. Sample Negative Pairs (Not linked in trust graph)
    for idx in range(half_batch, batch_size):
        u = rng.integers(0, num_users)
        v = rng.integers(0, num_users)

        start, end = indptr_arr[u], indptr_arr[u+1]
        pos_list = indices_arr[start:end] if start < end else []

        while u == v or v in pos_list:
            u = rng.integers(0, num_users)
            v = rng.integers(0, num_users)
            start, end = indptr_arr[u], indptr_arr[u+1]
            pos_list = indices_arr[start:end] if start < end else []

        users_u[idx] = u
        users_v[idx] = v
        trust_vals[idx] = 0.0

    return users_u, users_v, trust_vals


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
    Social-LightGCN Engine incorporating social trust graph structural signals,
    early feature fusion gates, and adaptive multi-task learning optimization.
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

        self.model = SocialLightGCNModel(
            num_users, num_items, embedding_dim, num_layers
        ).to(self.device)

        # Sparsified adjacency matrices in PyTorch
        self.adj_ui: torch.Tensor = torch.empty(0)
        self.adj_iu: torch.Tensor = torch.empty(0)
        self.adj_uv: torch.Tensor = torch.empty(0)

        # Raw user-item interaction matrix
        self._interaction_csr: sp.csr_matrix = sp.csr_matrix((0, 0))

        # Latent representations readout cache
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def fit(self, data: Any) -> None:
        """
        Train the model using adaptive multi-task learning.

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

        # Bipartite symmetric normalization
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

        # Social trust graph symmetric normalization
        rowsum_social = np.array(trust_mat.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_s_inv = np.power(rowsum_social, -0.5)
        d_s_inv[np.isinf(d_s_inv)] = 0.0
        D_s_inv = sp.diags(d_s_inv)

        T_norm = D_s_inv.dot(trust_mat).dot(D_s_inv)
        self.adj_uv = _sparse_scipy_to_torch(T_norm.tocsr(), self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_interactions = int(self._interaction_csr.nnz)
        n_batches = max(n_interactions // self.batch_size, 1)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_bpr_loss = 0.0
            epoch_rating_loss = 0.0
            epoch_social_loss = 0.0

            for _ in range(n_batches):
                # 1. Sample ratings batch
                users, pos_items, neg_items, pos_ratings = _sample_training_batch(
                    self._interaction_csr, self.batch_size, rng
                )
                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)
                pos_ratings_t = torch.FloatTensor(pos_ratings).to(self.device)

                # 2. Sample social batch
                social_users_u, social_users_v, social_trust = _sample_social_batch(
                    trust_mat, self.batch_size, rng
                )
                social_u_t = torch.LongTensor(social_users_u).to(self.device)
                social_v_t = torch.LongTensor(social_users_v).to(self.device)
                social_trust_t = torch.FloatTensor(social_trust).to(self.device)

                # Forward: Propagate through Early Fusion layers
                user_emb, item_emb = self.model(self.adj_ui, self.adj_iu, self.adj_uv)

                # Extract target embeddings
                u_emb = user_emb[users_t]
                pos_emb = item_emb[pos_t]
                neg_emb = item_emb[neg_t]

                # --- Task 1: BPR Ranking Loss with Squashed Logits Scale ---
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)

                loss_bpr = -torch.mean(
                    F.logsigmoid((pos_scores - neg_scores) * torch.exp(-self.model.log_vars[0]))
                ) + 0.5 * self.model.log_vars[0]

                # --- Task 2: Rating MSE Loss ---
                rating_preds = (
                    self.model.global_mu
                    + self.model.user_bias(users_t).squeeze()
                    + self.model.item_bias(pos_t).squeeze()
                    + (u_emb * pos_emb).sum(dim=1)
                )
                loss_rating = 0.5 * torch.exp(-self.model.log_vars[1]) * F.mse_loss(rating_preds, pos_ratings_t) + 0.5 * self.model.log_vars[1]

                # --- Task 3: Social Graph Reconstruction MSE Loss ---
                u_emb_social_u = user_emb[social_u_t]
                u_emb_social_v = user_emb[social_v_t]
                social_preds = (u_emb_social_u * u_emb_social_v).sum(dim=1)

                loss_social = self.social_loss_weight * F.mse_loss(
                    torch.sigmoid(social_preds), social_trust_t
                )

                # Weight decay (L2 Regularization) on base embeddings
                reg_loss = self.reg * (
                    self.model.user_embedding.weight[users_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[pos_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[neg_t].norm(2).pow(2)
                ) / self.batch_size

                # Unified Adaptive Multi-Task Loss
                loss_total = loss_bpr + loss_rating + loss_social + reg_loss

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # Squashing / clipping parameter values for uncertainty scales
                with torch.no_grad():
                    self.model.log_vars.clamp_(min=-5.0, max=5.0)

                epoch_loss += loss_total.item()
                epoch_bpr_loss += loss_bpr.item()
                epoch_rating_loss += loss_rating.item()
                epoch_social_loss += loss_social.item()

            avg_loss = epoch_loss / n_batches
            avg_bpr = epoch_bpr_loss / n_batches
            avg_rating = epoch_rating_loss / n_batches
            avg_social = epoch_social_loss / n_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  SocialGCN Epoch {epoch + 1:2d}/{self.n_epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"BPR: {avg_bpr:.4f} | Rating MSE: {avg_rating:.4f} | Social MSE: {avg_social:.4f} | "
                    f"log_vars: [{self.model.log_vars[0].item():.2f}, {self.model.log_vars[1].item():.2f}, {self.model.log_vars[2].item():.2f}]"
                )

        # Cache representations
        self._cache_embeddings()

    def _cache_embeddings(self) -> None:
        """Pre-compute and cache user and item embeddings for O(1) inference."""
        self.model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self.model(self.adj_ui, self.adj_iu, self.adj_uv)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id >= self.num_users or item_id >= self.num_items:
            return 3.5

        with torch.no_grad():
            user_vec = self._user_emb[user_id]
            item_vec = self._item_emb[item_id]

            pred = (
                self.model.global_mu.item()
                + self.model.user_bias.weight[user_id].item()
                + self.model.item_bias.weight[item_id].item()
                + torch.dot(user_vec, item_vec).item()
            )

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
