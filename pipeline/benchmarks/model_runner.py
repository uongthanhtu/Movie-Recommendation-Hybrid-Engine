"""
Model Runner -- translates an ArenaDataset into the exact fit() input shape each
BaseRecommenderEngine expects, applies large-dataset auto-scaling, and measures
training time.

Each of the four models below has a different fit() input shape (confirmed by
reading all five engine implementations during design -- see
docs/superpowers/specs/2026-06-25-grand-arena-runner-design.md):
  - LightGCNEngine.fit(data):        dict with "sym_adj_mat", "interaction_matrix"
  - SocialLightGCNEngine.fit(data):  dict with "interaction_matrix", "trust_matrix"
  - TrustSVDEngine.fit(data):        dict with "interaction_matrix", "trust_matrix"
  - FunkSVDEngine.fit(data):         pd.DataFrame with columns [user_idx, item_idx, rating]
This module owns that translation so grand_arena_runner.py's orchestration loop stays
uniform across all four.

SASRec is intentionally absent: it needs chronologically-ordered per-user item
sequences, but no loader in pipeline/data_loaders/ preserves timestamps (deferred
until that data path exists, per the design spec).
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import pandas as pd

from pipeline.data_loaders.base_loader import ArenaDataset
from pipeline.engines.base_engine import BaseRecommenderEngine
from pipeline.engines.funk_svd_engine import FunkSVDEngine
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine
from pipeline.engines.trust_svd_engine import TrustSVDEngine

MODE_A_MODELS = ["lightgcn", "trustsvd", "social_lightgcn"]
MODE_B_MODELS = ["funksvd", "lightgcn", "trustsvd", "social_lightgcn"]

LARGE_DATASET_USER_THRESHOLD = 10_000
LARGE_DATASET_MAX_EPOCHS = 15
LARGE_DATASET_MIN_BATCH_SIZE = 16384


def _scaled_kwargs(dataset: ArenaDataset, supports_batch_size: bool) -> Dict[str, int]:
    """
    Auto-scaling for large datasets, matching the existing precedent in
    pipeline/academic_sandbox/run_yelp_benchmark.py:175-179 exactly. Returns an empty
    dict (falls back to each engine's own constructor defaults) when dataset.num_users
    is at or below the threshold.
    """
    kwargs: Dict[str, int] = {}
    if dataset.num_users > LARGE_DATASET_USER_THRESHOLD:
        kwargs["n_epochs"] = LARGE_DATASET_MAX_EPOCHS
        if supports_batch_size:
            kwargs["batch_size"] = LARGE_DATASET_MIN_BATCH_SIZE
    return kwargs


def run_model(model_name: str, dataset: ArenaDataset) -> Tuple[BaseRecommenderEngine, float]:
    """
    Construct the engine named by model_name, build its fit() input from `dataset`,
    apply large-dataset auto-scaling if applicable, call fit(), and return
    (fitted_engine, train_seconds).

    Raises ValueError for an unrecognized model_name, or whatever the underlying
    engine's __init__/fit() raises -- the caller is responsible for catching and
    logging, not this function.
    """
    if model_name == "lightgcn":
        kwargs = _scaled_kwargs(dataset, supports_batch_size=True)
        engine: BaseRecommenderEngine = LightGCNEngine(
            num_users=dataset.num_users, num_items=dataset.num_items, **kwargs
        )
        fit_data = {"sym_adj_mat": dataset.get_sym_adj_mat(), "interaction_matrix": dataset.train_csr}
    elif model_name == "social_lightgcn":
        kwargs = _scaled_kwargs(dataset, supports_batch_size=True)
        engine = SocialLightGCNEngine(
            num_users=dataset.num_users, num_items=dataset.num_items, **kwargs
        )
        fit_data = {"interaction_matrix": dataset.train_csr, "trust_matrix": dataset.social_csr}
    elif model_name == "trustsvd":
        kwargs = _scaled_kwargs(dataset, supports_batch_size=False)
        engine = TrustSVDEngine(**kwargs)
        fit_data = {"interaction_matrix": dataset.train_csr, "trust_matrix": dataset.social_csr}
    elif model_name == "funksvd":
        engine = FunkSVDEngine()
        coo = dataset.train_csr.tocoo()
        fit_data = pd.DataFrame({"user_idx": coo.row, "item_idx": coo.col, "rating": coo.data})
    else:
        all_models = sorted(set(MODE_A_MODELS) | set(MODE_B_MODELS))
        raise ValueError(f"Unknown model_name '{model_name}'. Expected one of: {all_models}")

    start = time.perf_counter()
    engine.fit(fit_data)
    train_seconds = time.perf_counter() - start

    return engine, train_seconds
