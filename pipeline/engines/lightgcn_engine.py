"""
LightGCN Engine — Graph Convolutional Network for recommendation (PyTorch).

Core insight: Strip all nonlinear activations and feature transformations
from GCN. Only neighborhood aggregation matters for CF.

Embedding Propagation (layer k+1):
  e_u^{k+1} = sum_{i in N_u} (1 / sqrt(|N_u| * |N_i|)) * e_i^{k}
  e_i^{k+1} = sum_{u in N_i} (1 / sqrt(|N_i| * |N_u|)) * e_u^{k}

Layer Combination:
  e_u = (1 / (K+1)) * sum_{k=0}^{K} e_u^{k}

Loss (BPR — Bayesian Personalized Ranking):
  L = -sum_{u} sum_{i in N_u} sum_{j not in N_u}
      ln sigma(y_ui - y_uj) + lambda * ||E^{0}||^2

Reference:
  He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
  LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation. SIGIR 2020.
"""
import os
from typing import Any, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from pipeline.engines.base_engine import BaseRecommenderEngine


# ======================================================================
# PyTorch Model
# ======================================================================

class LightGCNModel(nn.Module):
    """Pure LightGCN model — linear embedding propagation + layer combination."""

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

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate embeddings through the bipartite graph.

        Args:
            adj_matrix: Sparse symmetric normalized adjacency
                        shape (num_users + num_items, num_users + num_items)

        Returns:
            (user_embeddings, item_embeddings) after layer combination.
        """
        all_emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        emb_list = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            emb_list.append(all_emb)

        # Layer combination: mean of all layers (0..K)
        stacked = torch.stack(emb_list, dim=0)  # (K+1, N+M, d)
        final = stacked.mean(dim=0)             # (N+M, d)

        user_emb = final[: self.num_users]
        item_emb = final[self.num_users :]
        return user_emb, item_emb


# ======================================================================
# BPR Training Utilities
# ======================================================================

def _sample_bpr_triplets(
    interaction_csr: sp.csr_matrix,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (user, positive_item, negative_item) triplets for BPR loss.
    """
    num_users, num_items = interaction_csr.shape
    users = np.zeros(num_samples, dtype=np.int64)
    pos_items = np.zeros(num_samples, dtype=np.int64)
    neg_items = np.zeros(num_samples, dtype=np.int64)

    indices_arr = interaction_csr.indices
    indptr_arr = interaction_csr.indptr

    for idx in range(num_samples):
        u = rng.integers(0, num_users)
        start, end = indptr_arr[u], indptr_arr[u+1]
        while start == end:
            u = rng.integers(0, num_users)
            start, end = indptr_arr[u], indptr_arr[u+1]

        pos_list = indices_arr[start:end]
        pos_i = pos_list[rng.integers(0, len(pos_list))]

        # Sample a negative item (not interacted)
        neg_j = rng.integers(0, num_items)
        while neg_j in pos_list:
            neg_j = rng.integers(0, num_items)

        users[idx] = u
        pos_items[idx] = pos_i
        neg_items[idx] = neg_j

    return users, pos_items, neg_items


def _sparse_scipy_to_torch(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert scipy sparse CSR to PyTorch sparse COO tensor."""
    coo = mat.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)


# ======================================================================
# Engine
# ======================================================================

class LightGCNEngine(BaseRecommenderEngine):
    """LightGCN with full BPR training loop."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 50,
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

        self.model = LightGCNModel(
            num_users, num_items, embedding_dim, num_layers
        ).to(self.device)

        self.adj_matrix: torch.Tensor = torch.empty(0)
        self._interaction_csr: sp.csr_matrix = sp.csr_matrix((0, 0))

        # Cached final embeddings for fast inference
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    # ------------------------------------------------------------------
    # fit — full BPR training
    # ------------------------------------------------------------------
    def fit(self, data: Any) -> None:
        """
        Train LightGCN with BPR loss.

        Args:
            data: dict with keys:
                  - "sym_adj_mat":        sp.csr_matrix (normalized bipartite adjacency)
                  - "interaction_matrix": sp.csr_matrix (num_users x num_items, binary)
        """
        sym_adj_mat: sp.csr_matrix = data["sym_adj_mat"]
        interaction_mat: sp.csr_matrix = data["interaction_matrix"]

        # Binarize interaction matrix for negative sampling
        self._interaction_csr = interaction_mat.copy()
        self._interaction_csr.data = np.ones_like(self._interaction_csr.data)

        # Convert adjacency to PyTorch sparse
        self.adj_matrix = _sparse_scipy_to_torch(sym_adj_mat, self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_interactions = int(self._interaction_csr.nnz)
        n_batches = max(n_interactions // self.batch_size, 1)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for _ in range(n_batches):
                # Sample BPR triplets
                users, pos_items, neg_items = _sample_bpr_triplets(
                    self._interaction_csr, self.batch_size, rng
                )

                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)

                # Forward pass: propagate all embeddings
                user_emb, item_emb = self.model(self.adj_matrix)

                u_emb = user_emb[users_t]        # (B, d)
                pos_emb = item_emb[pos_t]         # (B, d)
                neg_emb = item_emb[neg_t]         # (B, d)

                # BPR loss: -ln sigma(y_ui - y_uj)
                pos_scores = (u_emb * pos_emb).sum(dim=1)  # (B,)
                neg_scores = (u_emb * neg_emb).sum(dim=1)  # (B,)
                bpr_loss = -torch.log(
                    torch.sigmoid(pos_scores - neg_scores) + 1e-8
                ).mean()

                # L2 regularization on initial embeddings
                reg_loss = self.reg * (
                    self.model.user_embedding.weight[users_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[pos_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[neg_t].norm(2).pow(2)
                ) / self.batch_size

                loss = bpr_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  LightGCN Epoch {epoch + 1:3d}/{self.n_epochs} "
                    f"| BPR Loss: {avg_loss:.4f}"
                )

        # Cache final embeddings for inference
        self._cache_embeddings()

    def _cache_embeddings(self) -> None:
        """Pre-compute and cache final user/item embeddings."""
        self.model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self.model(self.adj_matrix)

    # ------------------------------------------------------------------
    # predict_rating
    # ------------------------------------------------------------------
    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id >= self.num_users or item_id >= self.num_items:
            return 0.0

        with torch.no_grad():
            score = torch.dot(
                self._user_emb[user_id], self._item_emb[item_id]
            ).item()
        return float(score)

    # ------------------------------------------------------------------
    # recommend_top_n
    # ------------------------------------------------------------------
    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        if user_id >= self.num_users:
            return list(range(min(top_n, self.num_items)))

        with torch.no_grad():
            user_vec = self._user_emb[user_id]                    # (d,)
            scores = torch.matmul(self._item_emb, user_vec)       # (M,)

            # Mask seen items with -inf
            seen = self._interaction_csr[user_id].indices
            if len(seen) > 0:
                scores[torch.LongTensor(seen).to(self.device)] = float("-inf")

            top_items = torch.topk(scores, min(top_n, self.num_items)).indices
        return top_items.cpu().numpy().tolist()

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._cache_embeddings()
