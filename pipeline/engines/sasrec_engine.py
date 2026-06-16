"""
SASRec Engine — Self-Attentive Sequential Recommendation (PyTorch).

Models user behavior as a time-ordered sequence and uses a Transformer
decoder (with causal masking) to predict the next item.

Embedding:
  E = [M_{x_1} + P_1, ..., M_{x_N} + P_N]

Causal Self-Attention:
  S = Softmax( (E W^Q)(E W^K)^T / sqrt(d) + Omega ) * (E W^V)
  where Omega_{ij} = 0 if i >= j, -inf otherwise

Loss (Binary Cross Entropy per position):
  L = -sum_t [ log sigma(e_t^T * e_{pos}) + log(1 - sigma(e_t^T * e_{neg})) ]

Reference:
  Kang, W.-C. & McAuley, J. (2018).
  Self-Attentive Sequential Recommendation. ICDM 2018.
"""
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.engines.base_engine import BaseRecommenderEngine


# ======================================================================
# PyTorch Model
# ======================================================================

class SASRecModel(nn.Module):
    """Transformer-based sequential recommendation model with causal masking."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 50,
        num_heads: int = 1,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # +1 for padding index 0
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Stacked self-attention blocks
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout_rate),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(hidden_dim))
            self.norm2_layers.append(nn.LayerNorm(hidden_dim))

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_seqs: (B, L) integer item IDs (0 = padding)

        Returns:
            hidden_states: (B, L, d) — latent representation at each position.
        """
        B, L = input_seqs.shape
        device = input_seqs.device

        # Positional encoding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        seq_emb = self.item_emb(input_seqs) + self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=device) * float("-inf"), diagonal=1
        )

        # Padding mask: ignore positions where input is 0
        key_padding_mask = (input_seqs == 0)  # (B, L)
        # Prevent NaNs by ensuring key_padding_mask is never all-True
        key_padding_mask[:, 0] = False

        for attn, ffn, norm1, norm2 in zip(
            self.attention_layers,
            self.ffn_layers,
            self.norm1_layers,
            self.norm2_layers,
        ):
            # Self-attention with residual
            attn_out, _ = attn(
                seq_emb, seq_emb, seq_emb,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )
            seq_emb = norm1(seq_emb + attn_out)

            # Feed-forward with residual
            ffn_out = ffn(seq_emb)
            seq_emb = norm2(seq_emb + ffn_out)

        return self.final_norm(seq_emb)


# ======================================================================
# Engine
# ======================================================================

class SASRecEngine(BaseRecommenderEngine):
    """SASRec with full BCE training loop on sequential user histories."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 50,
        num_heads: int = 1,
        num_blocks: int = 2,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 128,
        dropout_rate: float = 0.2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = SASRecModel(
            num_items, max_seq_len, hidden_dim, num_heads, num_blocks, dropout_rate
        ).to(self.device)

        self.user_sequences: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # fit — full sequential training
    # ------------------------------------------------------------------
    def fit(self, data: Any) -> None:
        """
        Train SASRec with Binary Cross-Entropy loss.

        For each user sequence (x_1, ..., x_T):
          Input:  (x_1, x_2, ..., x_{T-1})
          Target: (x_2, x_3, ..., x_T)

        At each position t, compute:
          pos_score = dot(hidden_t, emb(target_t))   -> label=1
          neg_score = dot(hidden_t, emb(neg_sample))  -> label=0

        Args:
            data: Dict[int, np.ndarray] mapping user_idx -> item sequence.
        """
        self.user_sequences = data
        user_ids = list(data.keys())

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        bce_loss = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(42)

        self.model.train()
        for epoch in range(self.n_epochs):
            rng.shuffle(user_ids)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(user_ids), self.batch_size):
                batch_uids = user_ids[start : start + self.batch_size]

                # Build input / target / negative arrays
                input_seqs_list = []
                target_seqs_list = []
                neg_seqs_list = []

                for uid in batch_uids:
                    seq = data[uid]  # shape (max_seq_len,)

                    # Input: shift right by 1 (drop last)
                    input_seq = seq[:-1]
                    # Target: shift left by 1 (drop first)
                    target_seq = seq[1:]

                    # Negative samples: random items not in this user's history
                    user_set = set(seq[seq > 0].tolist())
                    neg_seq = np.zeros_like(target_seq)
                    for t in range(len(target_seq)):
                        if target_seq[t] == 0:
                            neg_seq[t] = 0
                            continue
                        neg = rng.integers(1, self.num_items + 1)
                        while neg in user_set:
                            neg = rng.integers(1, self.num_items + 1)
                        neg_seq[t] = neg

                    input_seqs_list.append(input_seq)
                    target_seqs_list.append(target_seq)
                    neg_seqs_list.append(neg_seq)

                input_t = torch.LongTensor(np.array(input_seqs_list)).to(self.device)
                target_t = torch.LongTensor(np.array(target_seqs_list)).to(self.device)
                neg_t = torch.LongTensor(np.array(neg_seqs_list)).to(self.device)

                # Forward
                hidden = self.model(input_t)  # (B, L-1, d)

                # Positive & negative item embeddings
                pos_emb = self.model.item_emb(target_t)  # (B, L-1, d)
                neg_emb = self.model.item_emb(neg_t)     # (B, L-1, d)

                # Dot-product scores
                pos_logits = (hidden * pos_emb).sum(dim=-1)  # (B, L-1)
                neg_logits = (hidden * neg_emb).sum(dim=-1)  # (B, L-1)

                # Mask out padding positions
                mask = (target_t != 0).float()  # (B, L-1)

                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)

                loss = (
                    bce_loss(pos_logits * mask, pos_labels * mask)
                    + bce_loss(neg_logits * mask, neg_labels * mask)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  SASRec Epoch {epoch + 1:3d}/{self.n_epochs} "
                    f"| BCE Loss: {avg_loss:.4f}"
                )

    # ------------------------------------------------------------------
    # predict_rating
    # ------------------------------------------------------------------
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict score for (user, item). item_id is 0-indexed."""
        self.model.eval()
        seq = self.user_sequences.get(user_id)
        if seq is None:
            return 0.0

        # Convert 0-indexed item_id to 1-indexed for embedding lookup
        internal_item_id = item_id + 1

        with torch.no_grad():
            input_seq = torch.LongTensor(seq[:-1]).unsqueeze(0).to(self.device)
            hidden = self.model(input_seq)               # (1, L-1, d)
            last_hidden = hidden[:, -1, :]                # (1, d)
            item_emb = self.model.item_emb(
                torch.tensor([internal_item_id], device=self.device)
            )                                             # (1, d)
            score = torch.matmul(last_hidden, item_emb.T).squeeze()
        return float(score.item())

    # ------------------------------------------------------------------
    # recommend_top_n
    # ------------------------------------------------------------------
    def recommend_top_n(self, user_id: int, top_n: int = 10) -> List[int]:
        """Return top-N item IDs (0-indexed, consistent with other engines)."""
        self.model.eval()
        seq = self.user_sequences.get(user_id)
        if seq is None:
            return list(range(min(top_n, self.num_items)))

        with torch.no_grad():
            input_seq = torch.LongTensor(seq[:-1]).unsqueeze(0).to(self.device)
            hidden = self.model(input_seq)
            last_hidden = hidden[:, -1, :]  # (1, d)

            # Score all items (skip padding index 0)
            all_item_emb = self.model.item_emb.weight[1:]  # (num_items, d)
            scores = torch.matmul(last_hidden, all_item_emb.T).squeeze()  # (num_items,)

            # Mask items already in sequence (seq uses 1-indexed)
            seen = set(seq[seq > 0].tolist())
            for s in seen:
                if 1 <= s <= self.num_items:
                    scores[s - 1] = float("-inf")

            top_items = torch.topk(scores, min(top_n, self.num_items)).indices
            # Return 0-indexed: indices are already 0-based relative to weight[1:]
            result = top_items.cpu().numpy().tolist()
        return result

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "user_sequences": self.user_sequences,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.user_sequences = checkpoint["user_sequences"]
