"""
Evaluator -- Rigorous Top-K ranking evaluation engine for the Unified Arena.

Implements the All-Ranking protocol used in SEPT (KDD'21), DRSoRec (AAAI'26),
and other SOTA social recommendation papers:

  For each test user:
    1. Compute scores for ALL items
    2. Exclude items the user interacted with during training
    3. Rank remaining items by predicted score
    4. Compute Recall@K and NDCG@K against ground-truth test items

Metrics:
  - Recall@K: |{relevant items in top-K}| / |{relevant items}|
  - NDCG@K:   Normalized Discounted Cumulative Gain at K
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Set

import numpy as np
import torch


class ArenaEvaluator:
    """
    All-Ranking evaluation engine.

    Usage:
        evaluator = ArenaEvaluator(k_list=[10, 20])
        metrics = evaluator.evaluate(model, test_dict, train_dict, num_items)
    """

    def __init__(
        self,
        k_list: List[int] = None,
        max_eval_users: int = 5000,
        seed: int = 42,
    ):
        self.k_list = k_list or [10, 20]
        self.max_eval_users = max_eval_users
        self.seed = seed

    def evaluate(
        self,
        model: Any,
        test_dict: Dict[int, Set[int]],
        train_dict: Dict[int, Set[int]],
        num_items: int,
    ) -> Dict[str, float]:
        """
        Run full All-Ranking evaluation.

        Args:
            model:       Object with ``get_all_scores(user_id) -> Tensor``
            test_dict:   {user_idx: set(item_idx)} ground-truth test interactions
            train_dict:  {user_idx: set(item_idx)} to mask during ranking
            num_items:   Total number of items

        Returns:
            Dict mapping metric names (e.g. "Recall@10") to float values.
        """
        eval_users = [u for u in test_dict if len(test_dict[u]) > 0]

        if len(eval_users) > self.max_eval_users:
            rng = np.random.default_rng(self.seed)
            eval_users = rng.choice(
                eval_users, size=self.max_eval_users, replace=False
            ).tolist()
            print(f"    [Evaluator] Sampled {self.max_eval_users} / "
                  f"{len(test_dict)} test users", flush=True)

        max_k = max(self.k_list)
        recalls = {k: [] for k in self.k_list}
        ndcgs = {k: [] for k in self.k_list}
        precisions = {k: [] for k in self.k_list}

        t0 = time.time()
        for idx, u in enumerate(eval_users):
            ground_truth = test_dict[u]
            if not ground_truth:
                continue

            # Get predicted scores for all items
            scores = model.get_all_scores(u)

            # Mask training items (set to -inf)
            train_items = train_dict.get(u, set())
            if train_items:
                mask_idx = torch.LongTensor(list(train_items)).to(scores.device)
                scores[mask_idx] = float("-inf")

            # Top-K selection
            _, topk_indices = torch.topk(scores, min(max_k, num_items))
            topk_list = topk_indices.cpu().tolist()

            # Compute metrics for each K
            for k in self.k_list:
                top = topk_list[:k]
                hits = set(top) & ground_truth
                n_hits = len(hits)

                # Recall@K = |hits| / min(|ground_truth|, K)
                recalls[k].append(n_hits / min(len(ground_truth), k))

                # Precision@K = |hits| / K
                precisions[k].append(n_hits / k)

                # NDCG@K
                dcg = sum(
                    1.0 / np.log2(pos + 2)
                    for pos, item in enumerate(top)
                    if item in ground_truth
                )
                ideal_n = min(len(ground_truth), k)
                idcg = sum(1.0 / np.log2(pos + 2) for pos in range(ideal_n))
                ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

            # Progress reporting
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - t0
                speed = (idx + 1) / elapsed
                eta = (len(eval_users) - idx - 1) / speed
                print(f"    [Evaluator] {idx+1}/{len(eval_users)} users "
                      f"({speed:.0f} users/s, ETA: {eta:.0f}s)", flush=True)

        # Aggregate results
        results: Dict[str, float] = {}
        for k in self.k_list:
            results[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
            results[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
            results[f"Precision@{k}"] = float(np.mean(precisions[k])) if precisions[k] else 0.0

        elapsed = time.time() - t0
        print(f"    [Evaluator] Done in {elapsed:.1f}s ({len(eval_users)} users)", flush=True)

        return results


# ======================================================================
# Utility: Pretty-print results table
# ======================================================================

def print_results_table(
    results_list: List[Dict[str, Any]],
    k_list: List[int] = None,
    title: str = "BENCHMARK RESULTS",
) -> None:
    """
    Print a beautifully formatted ASCII comparison table.

    Args:
        results_list: List of dicts, each with "Model", "Train (s)", metric keys
        k_list: K values to display
        title: Table title
    """
    if not results_list:
        return
    if k_list is None:
        k_list = [10, 20]

    # Build column definitions
    columns = ["Model", "Train (s)"]
    for k in k_list:
        columns.extend([f"Recall@{k}", f"NDCG@{k}"])

    # Calculate column widths
    widths = {}
    for col in columns:
        widths[col] = max(len(col), 12)
        for row in results_list:
            val = row.get(col, "")
            widths[col] = max(widths[col], len(str(val)))

    # Separator
    total_width = sum(widths.values()) + 3 * (len(columns) - 1) + 4
    sep = "=" * total_width

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # Header
    header = " | ".join(f"{col:>{widths[col]}}" for col in columns)
    print(f"  {header}")
    print(f"  {'-' * (total_width - 4)}")

    # Data rows
    for row in results_list:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(f"{val:>{widths[col]}.4f}")
            else:
                cells.append(f"{str(val):>{widths[col]}}")
        print(f"  {' | '.join(cells)}")

    print(sep)

    # Improvement analysis (compare last model vs first)
    if len(results_list) >= 2:
        baseline = results_list[0]
        print(f"\n  Improvement Analysis (vs. {baseline['Model']}):")
        for row in results_list[1:]:
            print(f"    {row['Model']}:")
            for k in k_list:
                for metric in [f"Recall@{k}", f"NDCG@{k}"]:
                    b_val = baseline.get(metric, 0)
                    o_val = row.get(metric, 0)
                    if b_val > 0:
                        delta = ((o_val - b_val) / b_val) * 100
                        sign = "+" if delta > 0 else ""
                        indicator = " ***" if delta > 5 else (" **" if delta > 2 else "")
                        print(f"      {metric}: {b_val:.4f} -> {o_val:.4f} "
                              f"({sign}{delta:.2f}%){indicator}")
                    else:
                        print(f"      {metric}: {b_val:.4f} -> {o_val:.4f}")
        print()
