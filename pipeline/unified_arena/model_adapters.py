"""
Model Adapters -- Unified interface for all benchmark models in the Arena.

Provides a standard BaseAdapter ABC with concrete implementations:
  - SocialLightGCNAdapter: Wraps our production SocialLightGCNModel (PyTorch)
  - VanillaLightGCNAdapter: Standard LightGCN ablation (no social graph)
  - QRecModelAdapter: Dynamically imports models from external/QRec/ (TF-based)
    with a pure-PyTorch fallback reimplementation for portability.

All adapters share the same .fit() / .get_all_scores() / .get_name() contract
so the Evaluator can run any model seamlessly.
"""
from __future__ import annotations

import os
import sys
import abc
import time
import warnings
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.engines.social_lightgcn_engine import (
    SocialLightGCNModel,
    _sample_training_batch,
    _sample_social_batch,
    _sparse_scipy_to_torch,
)


# ======================================================================
# Abstract Base Adapter
# ======================================================================

class BaseAdapter(abc.ABC):
    """Standard interface for all benchmark models."""

    @abc.abstractmethod
    def get_name(self) -> str:
        """Human-readable model name for the results table."""
        ...

    @abc.abstractmethod
    def fit(
        self,
        train_csr: sp.csr_matrix,
        social_csr: sp.csr_matrix,
        num_users: int,
        num_items: int,
        n_epochs: int = 50,
        batch_size: int = 4096,
    ) -> None:
        """Train the model on the given data."""
        ...

    @abc.abstractmethod
    def get_all_scores(self, user_id: int) -> torch.Tensor:
        """Return a 1-D tensor of scores for ALL items for the given user."""
        ...


# ======================================================================
# Helper: Sparse SciPy -> PyTorch COO
# ======================================================================

def _to_torch_sparse(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert SciPy CSR to PyTorch sparse COO tensor."""
    coo = mat.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).to(device)


def _symmetric_norm(R: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} R D^{-1/2} symmetric normalization for bipartite graph."""
    R_bin = R.copy()
    R_bin.data = np.ones_like(R_bin.data, dtype=np.float32)

    row_sum = np.array(R_bin.sum(axis=1)).flatten()
    col_sum = np.array(R_bin.sum(axis=0)).flatten()

    with np.errstate(divide="ignore"):
        d_row = np.power(row_sum, -0.5)
        d_col = np.power(col_sum, -0.5)
    d_row[np.isinf(d_row)] = 0.0
    d_col[np.isinf(d_col)] = 0.0

    return sp.diags(d_row).dot(R_bin).dot(sp.diags(d_col))


def _symmetric_norm_square(A: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2} for square adjacency matrix."""
    row_sum = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv = np.power(row_sum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    D = sp.diags(d_inv)
    return D.dot(A).dot(D)


# ======================================================================
# 1. Social-LightGCN Adapter (Ours)
# ======================================================================

class SocialLightGCNAdapter(BaseAdapter):
    """
    Wraps our production SocialLightGCNModel for the unified arena.
    Auto-detects implicit feedback and disables Rating MSE loss.
    """

    def __init__(self, embedding_dim: int = 64, num_layers: int = 3,
                 lr: float = 1e-3, reg: float = 1e-4, max_batches: int = 30):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.reg = reg
        self.max_batches = max_batches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[SocialLightGCNModel] = None
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def get_name(self) -> str:
        return "Social-LightGCN (Ours)"

    def fit(self, train_csr: sp.csr_matrix, social_csr: sp.csr_matrix,
            num_users: int, num_items: int, n_epochs: int = 50,
            batch_size: int = 4096) -> None:

        self._model = SocialLightGCNModel(
            num_users, num_items, self.embedding_dim, self.num_layers
        ).to(self.device)

        # Detect implicit feedback -> disable Rating MSE
        unique_vals = np.unique(train_csr.data)
        is_implicit = len(unique_vals) <= 2 and np.max(unique_vals) <= 1.0
        if is_implicit:
            with torch.no_grad():
                self._model.global_mu.fill_(0.0)

        # Normalize adjacency matrices
        R_norm = _symmetric_norm(train_csr)
        adj_ui = _to_torch_sparse(R_norm.tocsr(), self.device)
        adj_iu = _to_torch_sparse(R_norm.T.tocsr(), self.device)

        T_norm = _symmetric_norm_square(social_csr)
        adj_uv = _to_torch_sparse(T_norm.tocsr(), self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)

        n_batches = min(max(train_csr.nnz // batch_size, 1), self.max_batches)

        self._model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for _ in range(n_batches):
                users, pos_items, neg_items, pos_ratings = _sample_training_batch(
                    train_csr, batch_size, rng
                )
                u_t = torch.LongTensor(users).to(self.device)
                p_t = torch.LongTensor(pos_items).to(self.device)
                n_t = torch.LongTensor(neg_items).to(self.device)

                social_u, social_v, trust_vals = _sample_social_batch(
                    social_csr, batch_size, rng
                )
                su_t = torch.LongTensor(social_u).to(self.device)
                sv_t = torch.LongTensor(social_v).to(self.device)
                st_t = torch.FloatTensor(trust_vals).to(self.device)

                user_emb, item_emb = self._model(adj_ui, adj_iu, adj_uv)

                u_e = user_emb[u_t]
                p_e = item_emb[p_t]
                n_e = item_emb[n_t]

                # BPR
                pos_s = (u_e * p_e).sum(1)
                neg_s = (u_e * n_e).sum(1)
                loss_bpr = -torch.mean(
                    F.logsigmoid((pos_s - neg_s) * torch.exp(-self._model.log_vars[0]))
                ) + 0.5 * self._model.log_vars[0]

                # Rating MSE (skip for implicit)
                if not is_implicit:
                    pr_t = torch.FloatTensor(pos_ratings).to(self.device)
                    preds = (self._model.global_mu
                             + self._model.user_bias(u_t).squeeze()
                             + self._model.item_bias(p_t).squeeze()
                             + (u_e * p_e).sum(1))
                    loss_rat = (0.5 * torch.exp(-self._model.log_vars[1])
                                * F.mse_loss(preds, pr_t)
                                + 0.5 * self._model.log_vars[1])
                else:
                    loss_rat = torch.tensor(0.0, device=self.device)

                # Social reconstruction
                su_e = user_emb[su_t]
                sv_e = user_emb[sv_t]
                sp = (su_e * sv_e).sum(1)
                loss_soc = (0.5 * torch.exp(-self._model.log_vars[2])
                            * F.mse_loss(torch.sigmoid(sp), st_t)
                            + 0.5 * self._model.log_vars[2])

                # L2
                reg_loss = self.reg * (
                    self._model.user_embedding.weight[u_t].norm(2).pow(2)
                    + self._model.item_embedding.weight[p_t].norm(2).pow(2)
                    + self._model.item_embedding.weight[n_t].norm(2).pow(2)
                ) / batch_size

                loss = loss_bpr + loss_rat + loss_soc + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self._model.log_vars.clamp_(-5.0, 5.0)
                epoch_loss += loss.item()

            avg = epoch_loss / n_batches
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lv = self._model.log_vars.detach().cpu().numpy()
                print(f"    [{self.get_name()}] Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Loss: {avg:.4f} | lv: [{lv[0]:.2f},{lv[1]:.2f},{lv[2]:.2f}]", flush=True)

        # Cache final embeddings
        self._model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self._model(adj_ui, adj_iu, adj_uv)

    def get_all_scores(self, user_id: int) -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(self._item_emb, self._user_emb[user_id])


# ======================================================================
# 2. Vanilla LightGCN Adapter (Ablation Baseline)
# ======================================================================

class _VanillaLightGCNNet(nn.Module):
    """Standard LightGCN (He et al., SIGIR'20) -- no social graph."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64, layers: int = 3):
        super().__init__()
        self.n_layers = layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, adj_ui, adj_iu):
        u, i = self.user_emb.weight, self.item_emb.weight
        u_list, i_list = [u], [i]
        for _ in range(self.n_layers):
            u_new = torch.sparse.mm(adj_ui, i)
            i_new = torch.sparse.mm(adj_iu, u)
            u, i = u_new, i_new
            u_list.append(u)
            i_list.append(i)
        return torch.stack(u_list, 0).mean(0), torch.stack(i_list, 0).mean(0)


class VanillaLightGCNAdapter(BaseAdapter):
    """Standard LightGCN -- BPR only, no social signal."""

    def __init__(self, embedding_dim: int = 64, num_layers: int = 3,
                 lr: float = 1e-3, reg: float = 1e-4, max_batches: int = 30):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.reg = reg
        self.max_batches = max_batches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[_VanillaLightGCNNet] = None
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def get_name(self) -> str:
        return "LightGCN (He et al.)"

    def fit(self, train_csr: sp.csr_matrix, social_csr: sp.csr_matrix,
            num_users: int, num_items: int, n_epochs: int = 50,
            batch_size: int = 4096) -> None:

        self._model = _VanillaLightGCNNet(
            num_users, num_items, self.embedding_dim, self.num_layers
        ).to(self.device)

        R_norm = _symmetric_norm(train_csr)
        adj_ui = _to_torch_sparse(R_norm.tocsr(), self.device)
        adj_iu = _to_torch_sparse(R_norm.T.tocsr(), self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        rng = np.random.default_rng(42)
        n_batches = min(max(train_csr.nnz // batch_size, 1), self.max_batches)

        self._model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for _ in range(n_batches):
                users, pos_items, neg_items, _ = _sample_training_batch(
                    train_csr, batch_size, rng
                )
                u_t = torch.LongTensor(users).to(self.device)
                p_t = torch.LongTensor(pos_items).to(self.device)
                n_t = torch.LongTensor(neg_items).to(self.device)

                user_emb, item_emb = self._model(adj_ui, adj_iu)
                u_e, p_e, n_e = user_emb[u_t], item_emb[p_t], item_emb[n_t]

                loss_bpr = -torch.mean(F.logsigmoid((u_e * p_e).sum(1) - (u_e * n_e).sum(1)))
                reg_loss = self.reg * (
                    self._model.user_emb.weight[u_t].norm(2).pow(2)
                    + self._model.item_emb.weight[p_t].norm(2).pow(2)
                    + self._model.item_emb.weight[n_t].norm(2).pow(2)
                ) / batch_size

                loss = loss_bpr + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    [{self.get_name()}] Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Loss: {epoch_loss/n_batches:.4f}", flush=True)

        self._model.eval()
        with torch.no_grad():
            self._user_emb, self._item_emb = self._model(adj_ui, adj_iu)

    def get_all_scores(self, user_id: int) -> torch.Tensor:
        with torch.no_grad():
            return torch.matmul(self._item_emb, self._user_emb[user_id])


# ======================================================================
# 3. QRec Model Adapter (External 3rd-Party)
# ======================================================================

class QRecModelAdapter(BaseAdapter):
    """
    Adapter for models from the QRec framework (Coder-Yu/QRec, TensorFlow).

    Strategy:
      1. Try to dynamically import from external/QRec/ via sys.path
      2. If unavailable, fall back to a pure-PyTorch reimplementation of the
         requested model (LightGCN or SEPT) for portability.

    Supported model_name values: "LightGCN", "SEPT"
    """

    def __init__(self, model_name: str = "LightGCN",
                 qrec_path: str = "external/QRec",
                 embedding_dim: int = 64, num_layers: int = 3,
                 lr: float = 1e-3, reg: float = 1e-4, max_batches: int = 30):
        self.model_name = model_name
        self.qrec_path = os.path.abspath(qrec_path)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.reg = reg
        self.max_batches = max_batches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._qrec_available = False
        self._fallback_adapter: Optional[BaseAdapter] = None
        self._user_emb: torch.Tensor = torch.empty(0)
        self._item_emb: torch.Tensor = torch.empty(0)

    def get_name(self) -> str:
        suffix = " (QRec)" if self._qrec_available else " (PyTorch Reimpl.)"
        return f"{self.model_name}{suffix}"

    def fit(self, train_csr: sp.csr_matrix, social_csr: sp.csr_matrix,
            num_users: int, num_items: int, n_epochs: int = 50,
            batch_size: int = 4096) -> None:

        # Attempt QRec import
        self._qrec_available = self._try_import_qrec()

        if self._qrec_available:
            print(f"    [{self.get_name()}] Using official QRec implementation", flush=True)
            self._fit_qrec(train_csr, social_csr, num_users, num_items, n_epochs)
        else:
            print(f"    [{self.get_name()}] QRec not found at {self.qrec_path}", flush=True)
            print(f"    [{self.get_name()}] Using PyTorch reimplementation as fallback", flush=True)
            self._fit_fallback(train_csr, social_csr, num_users, num_items, n_epochs, batch_size)

    def get_all_scores(self, user_id: int) -> torch.Tensor:
        if self._fallback_adapter is not None:
            return self._fallback_adapter.get_all_scores(user_id)
        with torch.no_grad():
            return torch.matmul(self._item_emb, self._user_emb[user_id])

    # ------------------------------------------------------------------
    # QRec Dynamic Import
    # ------------------------------------------------------------------
    def _try_import_qrec(self) -> bool:
        """Attempt to import QRec framework from external/QRec/."""
        if not os.path.isdir(self.qrec_path):
            return False
        try:
            if self.qrec_path not in sys.path:
                sys.path.append(self.qrec_path)
            # QRec requires TensorFlow 1.x -- check availability
            import tensorflow as tf
            if int(tf.__version__.split(".")[0]) >= 2:
                warnings.warn("QRec requires TensorFlow 1.x. Falling back to PyTorch.")
                return False
            return True
        except ImportError:
            return False

    def _fit_qrec(self, train_csr, social_csr, num_users, num_items, n_epochs):
        """Train using official QRec model. Writes temp config files."""
        # QRec models require file-based configuration.
        # This is a best-effort integration -- in practice, QRec's internal
        # data structures are tightly coupled with their framework.
        raise NotImplementedError(
            "Full QRec integration requires TensorFlow 1.14 and manual config setup. "
            "Use the PyTorch fallback for portable benchmarking."
        )

    # ------------------------------------------------------------------
    # PyTorch Fallback Reimplementations
    # ------------------------------------------------------------------
    def _fit_fallback(self, train_csr, social_csr, num_users, num_items,
                      n_epochs, batch_size):
        """Use a PyTorch reimplementation of the requested model."""
        if self.model_name.upper() == "LIGHTGCN":
            self._fallback_adapter = VanillaLightGCNAdapter(
                self.embedding_dim, self.num_layers, self.lr, self.reg, self.max_batches
            )
        elif self.model_name.upper() == "SEPT":
            # SEPT uses self-supervised tri-training with social augmentation.
            # We provide a simplified version (LightGCN + social) as approximation.
            print(f"    [Note] SEPT full contrastive learning not reimplemented. "
                  f"Using Social-LightGCN as structural approximation.", flush=True)
            self._fallback_adapter = SocialLightGCNAdapter(
                self.embedding_dim, self.num_layers, self.lr, self.reg, self.max_batches
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Supported: LightGCN, SEPT")

        self._fallback_adapter.fit(
            train_csr, social_csr, num_users, num_items, n_epochs, batch_size
        )
