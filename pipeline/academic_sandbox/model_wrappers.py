"""
Model Wrappers — Isolated wrappers for the Yelp Benchmark Sandbox.

Re-uses the core SocialLightGCNModel from the production engine but adapts
it for the Yelp evaluation pipeline.  Also provides a "vanilla LightGCN" 
baseline wrapper (ablation: identical architecture but with the Attention Gate
disabled, i.e., alpha = 1.0 always — pure collaborative signal, no social).

These wrappers are STANDALONE and do NOT touch the production pipeline.
"""
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure parent paths are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.engines.social_lightgcn_engine import (
    SocialLightGCNModel,
    _sample_training_batch,
    _sample_social_batch,
    _sparse_scipy_to_torch,
)


# ======================================================================
# Social-LightGCN Wrapper for Yelp Sandbox
# ======================================================================

class SocialLightGCNWrapper:
    """
    Thin wrapper around the production SocialLightGCNModel that:
    - Takes pre-built CSR matrices directly (no need for BaseRecommenderEngine)
    - Provides ranking evaluation helpers (recommend_top_n, recommend_all_scores)
    - Reports per-epoch loss breakdown
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
        batch_size: int = 4096,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = SocialLightGCNModel(
            num_users, num_items, embedding_dim, num_layers
        ).to(self.device)

        # PyTorch sparse adjacency tensors
        self.adj_ui: torch.Tensor = torch.empty(0)
        self.adj_iu: torch.Tensor = torch.empty(0)
        self.adj_uv: torch.Tensor = torch.empty(0)

        # CSR for sampling
        self._interaction_csr: sp.csr_matrix = sp.csr_matrix((0, 0))

        # Cached embeddings after training
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

        self.name = "Social-LightGCN"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        interaction_csr: sp.csr_matrix,
        trust_csr: sp.csr_matrix,
    ) -> None:
        """Train using Adaptive MTL (BPR + optional Rating MSE + Social Reconstruction).

        Auto-detects implicit feedback datasets (all ratings ~ 1.0) and disables
        the Rating MSE task which would be degenerate (global_mu=3.5 vs target=1.0).
        """
        self._interaction_csr = interaction_csr.copy()

        # Detect implicit vs explicit feedback
        unique_ratings = np.unique(interaction_csr.data)
        is_implicit = len(unique_ratings) <= 2 and np.max(unique_ratings) <= 1.0
        if is_implicit:
            print(f"    [{self.name}] Detected IMPLICIT feedback -> disabling Rating MSE loss", flush=True)
            # Reset global_mu to 0 so it doesn't interfere via bias terms
            with torch.no_grad():
                self.model.global_mu.fill_(0.0)

        # Symmetric normalization of bipartite graph
        R_binary = interaction_csr.copy()
        R_binary.data = np.ones_like(R_binary.data, dtype=np.float32)

        user_deg = np.array(R_binary.sum(axis=1)).flatten()
        item_deg = np.array(R_binary.sum(axis=0)).flatten()

        with np.errstate(divide="ignore"):
            d_u = np.power(user_deg, -0.5)
            d_i = np.power(item_deg, -0.5)
        d_u[np.isinf(d_u)] = 0.0
        d_i[np.isinf(d_i)] = 0.0

        R_norm = sp.diags(d_u).dot(R_binary).dot(sp.diags(d_i))
        self.adj_ui = _sparse_scipy_to_torch(R_norm.tocsr(), self.device)
        self.adj_iu = _sparse_scipy_to_torch(R_norm.T.tocsr(), self.device)

        # Trust graph normalization
        rowsum = np.array(trust_csr.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_s = np.power(rowsum, -0.5)
        d_s[np.isinf(d_s)] = 0.0
        T_norm = sp.diags(d_s).dot(trust_csr).dot(sp.diags(d_s))
        self.adj_uv = _sparse_scipy_to_torch(T_norm.tocsr(), self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_batches_raw = max(self._interaction_csr.nnz // self.batch_size, 1)
        # Cap batches per epoch for CPU feasibility on large datasets
        n_batches = min(n_batches_raw, 30)
        loss_mode = "BPR + Social" if is_implicit else "BPR + Rating MSE + Social"
        print(f"    [{self.name}] Training: {self.n_epochs} epochs x {n_batches} batches | Loss: {loss_mode}", flush=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for _ in range(n_batches):
                users, pos_items, neg_items, pos_ratings = _sample_training_batch(
                    self._interaction_csr, self.batch_size, rng
                )
                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)

                social_u, social_v, social_trust = _sample_social_batch(
                    trust_csr, self.batch_size, rng
                )
                social_u_t = torch.LongTensor(social_u).to(self.device)
                social_v_t = torch.LongTensor(social_v).to(self.device)
                social_trust_t = torch.FloatTensor(social_trust).to(self.device)

                user_emb, item_emb = self.model(self.adj_ui, self.adj_iu, self.adj_uv)

                u_e = user_emb[users_t]
                pos_e = item_emb[pos_t]
                neg_e = item_emb[neg_t]

                # Task 1: BPR Ranking Loss (always active)
                pos_scores = (u_e * pos_e).sum(dim=1)
                neg_scores = (u_e * neg_e).sum(dim=1)
                loss_bpr = -torch.mean(
                    F.logsigmoid((pos_scores - neg_scores) * torch.exp(-self.model.log_vars[0]))
                ) + 0.5 * self.model.log_vars[0]

                # Task 2: Rating MSE (only for explicit feedback)
                if not is_implicit:
                    pos_ratings_t = torch.FloatTensor(pos_ratings).to(self.device)
                    rating_preds = (
                        self.model.global_mu
                        + self.model.user_bias(users_t).squeeze()
                        + self.model.item_bias(pos_t).squeeze()
                        + (u_e * pos_e).sum(dim=1)
                    )
                    loss_rating = (
                        0.5 * torch.exp(-self.model.log_vars[1])
                        * F.mse_loss(rating_preds, pos_ratings_t)
                        + 0.5 * self.model.log_vars[1]
                    )
                else:
                    loss_rating = torch.tensor(0.0, device=self.device)

                # Task 3: Social Graph Reconstruction
                su = user_emb[social_u_t]
                sv = user_emb[social_v_t]
                social_preds = (su * sv).sum(dim=1)
                loss_social = (
                    0.5 * torch.exp(-self.model.log_vars[2])
                    * F.mse_loss(torch.sigmoid(social_preds), social_trust_t)
                    + 0.5 * self.model.log_vars[2]
                )

                # L2 Regularization
                reg_loss = self.reg * (
                    self.model.user_embedding.weight[users_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[pos_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[neg_t].norm(2).pow(2)
                ) / self.batch_size

                loss = loss_bpr + loss_rating + loss_social + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.model.log_vars.clamp_(min=-5.0, max=5.0)

                epoch_loss += loss.item()

            avg = epoch_loss / n_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                lv = self.model.log_vars.detach().cpu().numpy()
                print(
                    f"    [{self.name}] Epoch {epoch+1:2d}/{self.n_epochs} | "
                    f"Loss: {avg:.4f} | log_vars: [{lv[0]:.2f}, {lv[1]:.2f}, {lv[2]:.2f}]",
                    flush=True,
                )

        self._cache()

    def _cache(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self.model(self.adj_ui, self.adj_iu, self.adj_uv)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend_top_n(self, user_id: int, top_n: int = 20, exclude: Set[int] = None) -> List[int]:
        with torch.no_grad():
            scores = torch.matmul(self._item_emb, self._user_emb[user_id])
            if exclude:
                idx = torch.LongTensor(list(exclude)).to(self.device)
                scores[idx] = float("-inf")
            return torch.topk(scores, min(top_n, self.num_items)).indices.cpu().tolist()

    def get_all_scores(self, user_id: int) -> torch.Tensor:
        """Return raw dot-product scores for all items for a single user."""
        with torch.no_grad():
            return torch.matmul(self._item_emb, self._user_emb[user_id])


# ======================================================================
# Vanilla LightGCN Wrapper (Ablation — no social graph)
# ======================================================================

class VanillaLightGCNModel(nn.Module):
    """
    Standard LightGCN (He et al., SIGIR'20) without any social graph integration.
    Used as an ablation baseline to quantify the contribution of social trust.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_ui: torch.Tensor, adj_iu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        u_list = [u_emb]
        i_list = [i_emb]

        for _ in range(self.num_layers):
            u_emb_new = torch.sparse.mm(adj_ui, i_emb)
            i_emb_new = torch.sparse.mm(adj_iu, u_emb)
            u_emb = u_emb_new
            i_emb = i_emb_new
            u_list.append(u_emb)
            i_list.append(i_emb)

        return torch.stack(u_list, 0).mean(0), torch.stack(i_list, 0).mean(0)


class VanillaLightGCNWrapper:
    """Wrapper for standard LightGCN — BPR-only, no social loss."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 4096,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = num_users
        self.num_items = num_items
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = VanillaLightGCNModel(
            num_users, num_items, embedding_dim, num_layers
        ).to(self.device)

        self.adj_ui: torch.Tensor = torch.empty(0)
        self.adj_iu: torch.Tensor = torch.empty(0)
        self._interaction_csr: sp.csr_matrix = sp.csr_matrix((0, 0))
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

        self.name = "LightGCN (Vanilla)"

    def fit(self, interaction_csr: sp.csr_matrix, trust_csr: sp.csr_matrix = None) -> None:
        """Train with BPR loss only (ignores trust_csr)."""
        self._interaction_csr = interaction_csr.copy()

        R_binary = interaction_csr.copy()
        R_binary.data = np.ones_like(R_binary.data, dtype=np.float32)

        user_deg = np.array(R_binary.sum(axis=1)).flatten()
        item_deg = np.array(R_binary.sum(axis=0)).flatten()

        with np.errstate(divide="ignore"):
            d_u = np.power(user_deg, -0.5)
            d_i = np.power(item_deg, -0.5)
        d_u[np.isinf(d_u)] = 0.0
        d_i[np.isinf(d_i)] = 0.0

        R_norm = sp.diags(d_u).dot(R_binary).dot(sp.diags(d_i))
        self.adj_ui = _sparse_scipy_to_torch(R_norm.tocsr(), self.device)
        self.adj_iu = _sparse_scipy_to_torch(R_norm.T.tocsr(), self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_batches_raw = max(self._interaction_csr.nnz // self.batch_size, 1)
        # Cap batches per epoch for CPU feasibility on large datasets
        n_batches = min(n_batches_raw, 20)
        print(f"    [{self.name}] Training: {self.n_epochs} epochs x {n_batches} batches (batch_size={self.batch_size})", flush=True)

        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0

            for _ in range(n_batches):
                users, pos_items, neg_items, _ = _sample_training_batch(
                    self._interaction_csr, self.batch_size, rng
                )
                users_t = torch.LongTensor(users).to(self.device)
                pos_t = torch.LongTensor(pos_items).to(self.device)
                neg_t = torch.LongTensor(neg_items).to(self.device)

                user_emb, item_emb = self.model(self.adj_ui, self.adj_iu)

                u_e = user_emb[users_t]
                pos_e = item_emb[pos_t]
                neg_e = item_emb[neg_t]

                pos_scores = (u_e * pos_e).sum(dim=1)
                neg_scores = (u_e * neg_e).sum(dim=1)
                loss_bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

                reg_loss = self.reg * (
                    self.model.user_embedding.weight[users_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[pos_t].norm(2).pow(2)
                    + self.model.item_embedding.weight[neg_t].norm(2).pow(2)
                ) / self.batch_size

                loss = loss_bpr + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg = epoch_loss / n_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [{self.name}] Epoch {epoch+1:2d}/{self.n_epochs} | Loss: {avg:.4f}", flush=True)

        self._cache()

    def _cache(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self.model(self.adj_ui, self.adj_iu)

    def recommend_top_n(self, user_id: int, top_n: int = 20, exclude: Set[int] = None) -> List[int]:
        with torch.no_grad():
            scores = torch.matmul(self._item_emb, self._user_emb[user_id])
            if exclude:
                idx = torch.LongTensor(list(exclude)).to(self.device)
                scores[idx] = float("-inf")
            return torch.topk(scores, min(top_n, self.num_items)).indices.cpu().tolist()

    def get_all_scores(self, user_id: int) -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(self._item_emb, self._user_emb[user_id])
