"""
Evaluation -- Recall@K / NDCG@K / serving-latency measurement against the
ArenaDataset contract (pipeline/data_loaders/base_loader.py).

Built fresh rather than reusing any of the three pre-existing, mutually divergent
evaluators (pipeline/unified_arena/evaluator.py::ArenaEvaluator,
pipeline/engines/benchmark_arena.py's utility functions,
pipeline/academic_sandbox/run_yelp_benchmark.py::evaluate_ranking) -- none of them
operate against ArenaDataset, and every BaseRecommenderEngine.recommend_top_n
implementation already masks seen/training items internally (confirmed by reading
all five engine implementations during design), so this evaluator needs no masking
logic of its own.
"""
from __future__ import annotations

import math
import time
from typing import Dict

from pipeline.data_loaders.base_loader import ArenaDataset
from pipeline.engines.base_engine import BaseRecommenderEngine


def evaluate_model(engine: BaseRecommenderEngine, dataset: ArenaDataset, k: int = 10) -> Dict[str, float]:
    """
    For every user_id in dataset.test_dict with a non-empty test item set, calls
    engine.recommend_top_n(user_id, top_n=k), times the call, and accumulates
    Recall@k and NDCG@k (binary relevance) against dataset.test_dict[user_id].
    Users with an empty test item set are skipped entirely (not counted in the mean).

    Returns {f"recall@{k}": float, f"ndcg@{k}": float, "latency_ms": float} --
    latency_ms is the mean per-call wall-clock time in milliseconds across all
    evaluated users. Returns all-zero values if no user had a non-empty test set.

    Does NOT re-mask dataset.train_dict items -- every engine's recommend_top_n
    already excludes training items internally.
    """
    recalls = []
    ndcgs = []
    latencies_ms = []

    for user_id, test_items in dataset.test_dict.items():
        if not test_items:
            continue

        start = time.perf_counter()
        recommended = engine.recommend_top_n(user_id, top_n=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        hits = [1 if item in test_items else 0 for item in recommended[:k]]
        n_hits = sum(hits)
        recalls.append(n_hits / len(test_items))

        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        n_ideal = min(k, len(test_items))
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    n_evaluated = len(recalls)
    return {
        f"recall@{k}": sum(recalls) / n_evaluated if n_evaluated else 0.0,
        f"ndcg@{k}": sum(ndcgs) / n_evaluated if n_evaluated else 0.0,
        "latency_ms": sum(latencies_ms) / n_evaluated if n_evaluated else 0.0,
    }
