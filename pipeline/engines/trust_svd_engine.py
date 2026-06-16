"""
TrustSVD Engine — Social-aware matrix factorization (PyTorch optimized).

Extends SVD++ by incorporating an implicit trust network derived from
co-interaction Jaccard similarity, allowing the model to regularize
latent vectors of cold-start users via their trusted neighbors.

Prediction:
  r_hat(u,i) = mu + b_u + b_i
               + ( p_u
                   + |I_u|^{-0.5} * sum_{j in I_u} y_j
                   + |T_u|^{-0.5} * sum_{v in T_u} w_v
                 )^T * q_i

Loss:
  L = 0.5 * sum_{u,i}(r_ui - r_hat)^2
      + (lambda_t / 2) * sum_{u,v}(t_uv - p_u^T w_v)^2
      + (lambda / 2) * (||p||^2 + ||q||^2 + ||y||^2 + ||w||^2 + b_u^2 + b_i^2)

Reference:
  Guo, G., Zhang, J., & Yorke-Smith, N. (2015).
  TrustSVD: Collaborative Filtering with Both the Explicit and Implicit
  Influence of User Trust and of Item Ratings. AAAI 2015.
"""
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.engines.base_engine import BaseRecommenderEngine


# ======================================================================
# PyTorch Model
# ======================================================================

class TrustSVDModel(nn.Module):
    """PyTorch implementation of TrustSVD model."""
    def __init__(self, num_users: int, num_items: int, n_factors: int = 50, reg: float = 0.02, reg_trust: float = 0.01):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_factors = n_factors
        self.reg = reg
        self.reg_trust = reg_trust

        self.u_bias = nn.Embedding(num_users, 1)
        self.i_bias = nn.Embedding(num_items, 1)
        self.p = nn.Embedding(num_users, n_factors)
        self.q = nn.Embedding(num_items, n_factors)
        self.y = nn.Embedding(num_items, n_factors)
        self.w = nn.Embedding(num_users, n_factors)

        # Initialize
        scale = 0.01
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)
        nn.init.normal_(self.p.weight, std=scale)
        nn.init.normal_(self.q.weight, std=scale)
        nn.init.normal_(self.y.weight, std=scale)
        nn.init.normal_(self.w.weight, std=scale)

    def forward(self, users: torch.Tensor, items: torch.Tensor, S_I: torch.Tensor, S_T: torch.Tensor, mu: float) -> torch.Tensor:
        # Compute implicit item sum: S_I is (num_users, num_items) sparse tensor
        imp_item_sum = torch.sparse.mm(S_I, self.y.weight)  # (num_users, K)

        # Compute trust sum: S_T is (num_users, num_users) sparse tensor
        trust_sum = torch.sparse.mm(S_T, self.w.weight)  # (num_users, K)

        # Effective user representations
        p_star = self.p.weight + imp_item_sum + trust_sum  # (num_users, K)

        # Predict ratings
        user_bias = self.u_bias(users).squeeze()
        item_bias = self.i_bias(items).squeeze()
        dot = (p_star[users] * self.q(items)).sum(dim=1)
        predictions = mu + user_bias + item_bias + dot

        return predictions

    def compute_trust_predictions(self, u_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        return (self.p(u_t) * self.w(v_t)).sum(dim=1)


# ======================================================================
# Engine
# ======================================================================

class TrustSVDEngine(BaseRecommenderEngine):
    """TrustSVD recommendation engine powered by PyTorch for speed and efficiency."""

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 30,
        lr: float = 0.005,
        reg: float = 0.02,
        reg_trust: float = 0.01,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.reg_trust = reg_trust

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mu = 0.0
        self.num_users = 0
        self.num_items = 0

        # Lookup structures
        self._user_items = {}
        self._user_trustees = {}
        self._trust_pairs = []

        # Sparse tensors
        self.S_I = None
        self.S_T = None

        # Cached user embeddings for recommendation/prediction
        self._user_emb_cached = None

    def fit(self, data: Any) -> None:
        """
        Train TrustSVD via PyTorch optimization.

        Args:
            data: dict with keys:
                  - "interaction_matrix": sp.csr_matrix (num_users x num_items)
                  - "trust_matrix":       sp.csr_matrix (num_users x num_users)
        """
        interaction_mat: sp.csr_matrix = data["interaction_matrix"]
        trust_mat: sp.csr_matrix = data["trust_matrix"]

        self.num_users, self.num_items = interaction_mat.shape

        # Extract explicit ratings
        coo = interaction_mat.tocoo()
        row = coo.row
        col = coo.col
        ratings = coo.data.astype(np.float32)
        self.mu = float(ratings.mean())

        # Build user_items dictionary
        self._user_items = {}
        for u, i in zip(row, col):
            self._user_items.setdefault(int(u), []).append(int(i))

        # Build trust lookup
        coo_trust = trust_mat.tocoo()
        t_row = coo_trust.row
        t_col = coo_trust.col
        t_data = coo_trust.data

        self._user_trustees = {}
        self._trust_pairs = []
        for u, v, t_val in zip(t_row, t_col, t_data):
            u, v = int(u), int(v)
            self._user_trustees.setdefault(u, []).append(v)
            self._trust_pairs.append((u, v, float(t_val)))

        # Build sparse normalization matrices S_I and S_T
        user_interaction_counts = np.array((interaction_mat > 0).sum(axis=1)).flatten()
        s_i_values = np.zeros(len(ratings), dtype=np.float32)
        for idx, u in enumerate(row):
            count = user_interaction_counts[u]
            s_i_values[idx] = 1.0 / np.sqrt(count) if count > 0 else 0.0

        i_idx = torch.LongTensor(np.vstack((row, col)))
        i_val = torch.FloatTensor(s_i_values)
        self.S_I = torch.sparse_coo_tensor(i_idx, i_val, torch.Size([self.num_users, self.num_items])).coalesce().to(self.device)

        if len(t_row) > 0:
            user_trust_counts = np.array((trust_mat > 0).sum(axis=1)).flatten()
            s_t_values = np.zeros(len(t_data), dtype=np.float32)
            for idx, u in enumerate(t_row):
                count = user_trust_counts[u]
                s_t_values[idx] = 1.0 / np.sqrt(count) if count > 0 else 0.0

            t_idx = torch.LongTensor(np.vstack((t_row, t_col)))
            t_val = torch.FloatTensor(s_t_values)
            self.S_T = torch.sparse_coo_tensor(t_idx, t_val, torch.Size([self.num_users, self.num_users])).coalesce().to(self.device)
        else:
            # Empty sparse matrix
            self.S_T = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.Size([self.num_users, self.num_users])
            ).coalesce().to(self.device)

        # Convert arrays to PyTorch tensors
        users_t = torch.LongTensor(row).to(self.device)
        items_t = torch.LongTensor(col).to(self.device)
        ratings_t = torch.FloatTensor(ratings).to(self.device)

        if len(t_row) > 0:
            u_t = torch.LongTensor(t_row).to(self.device)
            v_t = torch.LongTensor(t_col).to(self.device)
            t_val_t = torch.FloatTensor(t_data).to(self.device)

        # Initialize PyTorch model
        self.model = TrustSVDModel(
            self.num_users, self.num_items, self.n_factors, self.reg, self.reg_trust
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Rating prediction and loss
            predictions = self.model(users_t, items_t, self.S_I, self.S_T, self.mu)
            rating_loss = F.mse_loss(predictions, ratings_t)

            # Trust prediction and loss
            if len(t_row) > 0:
                trust_preds = self.model.compute_trust_predictions(u_t, v_t)
                trust_loss = self.model.reg_trust * F.mse_loss(trust_preds, t_val_t)
            else:
                trust_loss = 0.0

            # L2 Regularization
            reg_loss = 0.5 * self.model.reg * (
                self.model.p.weight.pow(2).sum()
                + self.model.q.weight.pow(2).sum()
                + self.model.y.weight.pow(2).sum()
                + self.model.w.weight.pow(2).sum()
                + self.model.u_bias.weight.pow(2).sum()
                + self.model.i_bias.weight.pow(2).sum()
            )

            loss = rating_loss + trust_loss + reg_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  TrustSVD Epoch {epoch + 1:3d}/{self.n_epochs} "
                    f"| Loss: {loss.item():.4f} "
                    f"| Rating MSE: {rating_loss.item():.4f} "
                    f"| Trust MSE: {trust_loss.item() if len(t_row) > 0 else 0.0:.4f}"
                )

        # Cache effective user embeddings
        self._cache_embeddings()

    def _cache_embeddings(self) -> None:
        self.model.eval()
        with torch.no_grad():
            imp_item_sum = torch.sparse.mm(self.S_I, self.model.y.weight)
            trust_sum = torch.sparse.mm(self.S_T, self.model.w.weight)
            self._user_emb_cached = (self.model.p.weight + imp_item_sum + trust_sum).cpu()

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id >= self.num_users or item_id >= self.num_items:
            return float(self.mu)

        self.model.eval()
        with torch.no_grad():
            p_star = self._user_emb_cached[user_id]
            q_i = self.model.q.weight[item_id].cpu()
            pred = (
                self.mu
                + self.model.u_bias.weight[user_id].item()
                + self.model.i_bias.weight[item_id].item()
                + torch.dot(p_star, q_i).item()
            )
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        Predict ratings for a batch of item IDs for a given user.
        Vectorized in PyTorch/NumPy.
        """
        if user_id >= self.num_users or not item_ids:
            return np.full(len(item_ids), float(self.mu), dtype=np.float32)

        # Handle out of range item IDs safely
        valid_item_ids = [min(item_id, self.num_items - 1) for item_id in item_ids]

        self.model.eval()
        with torch.no_grad():
            p_star = self._user_emb_cached[user_id].to(self.device)  # (K,)
            q_items = self.model.q.weight[valid_item_ids]          # (B, K)

            user_bias = self.model.u_bias.weight[user_id].item()
            item_biases = self.model.i_bias.weight[valid_item_ids].view(-1)  # (B,)

            dot = torch.matmul(q_items, p_star)  # (B,)
            preds = self.mu + user_bias + item_biases + dot  # (B,)

            preds = torch.clamp(preds, 1.0, 5.0)
            preds_np = preds.cpu().numpy()

            # Handle fallback for items that were out of range
            for idx, item_id in enumerate(item_ids):
                if item_id >= self.num_items:
                    preds_np[idx] = float(self.mu)

        return preds_np

    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        if user_id >= self.num_users:
            return list(range(min(top_n, self.num_items)))

        self.model.eval()
        with torch.no_grad():
            p_star = self._user_emb_cached[user_id].to(self.device)  # (K,)
            # Vectorized scoring for all items
            scores = (
                self.mu
                + self.model.u_bias.weight[user_id].item()
                + self.model.i_bias.weight.squeeze()
                + torch.matmul(self.model.q.weight, p_star)
            )

            # Mask seen items
            seen = self._user_items.get(user_id, [])
            if len(seen) > 0:
                scores[torch.LongTensor(seen).to(self.device)] = float("-inf")

            top_items = torch.topk(scores, min(top_n, self.num_items)).indices
        return top_items.cpu().numpy().tolist()

    def save_model(self, path: str) -> None:
        state = {
            "model_state": self.model.state_dict(),
            "mu": self.mu,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "user_items": self._user_items,
            "user_trustees": self._user_trustees,
            "trust_pairs": self._trust_pairs,
        }
        torch.save(state, path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.num_users = state["num_users"]
        self.num_items = state["num_items"]
        self.mu = state["mu"]
        self._user_items = state["user_items"]
        self._user_trustees = state["user_trustees"]
        self._trust_pairs = state["trust_pairs"]

        self.model = TrustSVDModel(
            self.num_users, self.num_items, self.n_factors
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])

        # Reconstruct sparse tensors
        self._reconstruct_sparse_tensors()
        self._cache_embeddings()

    def _reconstruct_sparse_tensors(self) -> None:
        # Reconstruct S_I
        row, col, ratings = [], [], []
        for u, items in self._user_items.items():
            for i in items:
                row.append(u)
                col.append(i)
                ratings.append(1.0) # binary/placeholder

        # Compute counts
        user_interaction_counts = {}
        for u, items in self._user_items.items():
            user_interaction_counts[u] = len(items)

        s_i_values = []
        for u in row:
            count = user_interaction_counts[u]
            s_i_values.append(1.0 / np.sqrt(count) if count > 0 else 0.0)

        i_idx = torch.LongTensor(np.vstack((row, col)))
        i_val = torch.FloatTensor(s_i_values)
        self.S_I = torch.sparse_coo_tensor(i_idx, i_val, torch.Size([self.num_users, self.num_items])).coalesce().to(self.device)

        # Reconstruct S_T
        if len(self._trust_pairs) > 0:
            t_row = [p[0] for p in self._trust_pairs]
            t_col = [p[1] for p in self._trust_pairs]
            t_data = [p[2] for p in self._trust_pairs]

            user_trust_counts = {}
            for u, trustees in self._user_trustees.items():
                user_trust_counts[u] = len(trustees)

            s_t_values = []
            for u in t_row:
                count = user_trust_counts.get(u, 0)
                s_t_values.append(1.0 / np.sqrt(count) if count > 0 else 0.0)

            t_idx = torch.LongTensor(np.vstack((t_row, t_col)))
            t_val = torch.FloatTensor(s_t_values)
            self.S_T = torch.sparse_coo_tensor(t_idx, t_val, torch.Size([self.num_users, self.num_users])).coalesce().to(self.device)
        else:
            self.S_T = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.Size([self.num_users, self.num_users])
            ).coalesce().to(self.device)
