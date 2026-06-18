"""
Unified Academic Benchmark Arena -- Main Orchestrator (CLI).

Runs a rigorous, apples-to-apples comparison of Social-LightGCN against
SOTA baselines on the standard Ciao dataset using identical evaluation protocol.

Pipeline:
  1. Download & preprocess Ciao dataset (5-core filtering)
  2. Initialize all competing models
  3. Train each model on the same training data
  4. Evaluate each model with the same All-Ranking protocol
  5. Print comparative results table

Usage:
    py pipeline/unified_arena/run_arena.py
    py pipeline/unified_arena/run_arena.py --epochs 30 --dim 64
    py -u pipeline/unified_arena/run_arena.py --help
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Any, Dict, List

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.unified_arena.academic_data_loader import AcademicDataLoader, ArenaDataset
from pipeline.unified_arena.model_adapters import (
    BaseAdapter,
    SocialLightGCNAdapter,
    VanillaLightGCNAdapter,
    QRecModelAdapter,
)
from pipeline.unified_arena.evaluator import ArenaEvaluator, print_results_table


# ======================================================================
# ASCII Banner
# ======================================================================

_BANNER = r"""
================================================================================
    __  __      _ _____              __    ___
   / / / /___  (_) __(_)__  ____   /  |  / (_)__  ____  ____ _
  / / / / __ \/ / /_/ / _ \/ __ \ / /| | / / / _ \/ __ \/ __ `/
 / /_/ / / / / / __/ /  __/ /_/ // ___ |/ / /  __/ / / / /_/ /
 \____/_/ /_/_/_/ /_/\___/\____//_/  |_/_/_/\___/_/ /_/\__,_/

           Unified Academic Benchmark Arena
           Social-LightGCN vs. SOTA Baselines
           Dataset: CiaoDVD | Protocol: All-Ranking
================================================================================
"""


# ======================================================================
# Main Orchestrator
# ======================================================================

def run_arena(
    data_dir: str = "data/ciao",
    n_epochs: int = 50,
    embedding_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-3,
    reg: float = 1e-4,
    k_core: int = 5,
    k_list: List[int] = None,
    max_eval_users: int = 5000,
    qrec_path: str = "external/QRec",
    models_to_run: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute the full benchmark arena.

    Args:
        data_dir:        Directory for Ciao dataset files
        n_epochs:        Training epochs per model
        embedding_dim:   GCN embedding dimension
        num_layers:      Number of GCN propagation layers
        batch_size:      Training batch size
        lr:              Learning rate (Adam)
        reg:             L2 regularization weight
        k_core:          K-core filtering threshold
        k_list:          Top-K values for evaluation
        max_eval_users:  Cap for evaluation speed
        qrec_path:       Path to cloned QRec repository
        models_to_run:   Which models to include (default: all)

    Returns:
        List of result dicts for each model.
    """
    if k_list is None:
        k_list = [10, 20]
    if models_to_run is None:
        models_to_run = ["lightgcn", "qrec_lightgcn", "social_lightgcn"]

    print(_BANNER, flush=True)
    print("  Initializing Unified Academic Benchmark Arena...\n", flush=True)

    # ==================================================================
    # Step 1: Load & Preprocess Dataset
    # ==================================================================
    print("[Step 1/4] Loading CiaoDVD dataset ...", flush=True)
    t_data = time.time()

    loader = AcademicDataLoader(
        data_dir=data_dir,
        k_core=k_core,
        test_ratio=0.2,
        seed=42,
        rating_threshold=3.0,
    )
    dataset: ArenaDataset = loader.load()
    data_time = time.time() - t_data

    # Print dataset summary
    print(f"\n  {'='*60}", flush=True)
    print(f"  DATASET SUMMARY -- CiaoDVD ({k_core}-core filtered)", flush=True)
    print(f"  {'='*60}", flush=True)
    print(f"    Raw Interactions  : {dataset.n_raw_interactions:>10,}", flush=True)
    print(f"    Raw Users/Items   : {dataset.n_raw_users:>10,} / {dataset.n_raw_items:>7,}", flush=True)
    print(f"    {k_core}-core Rounds      : {dataset.filtering_rounds:>10}", flush=True)
    print(f"    Final Users       : {dataset.num_users:>10,}", flush=True)
    print(f"    Final Items       : {dataset.num_items:>10,}", flush=True)
    print(f"    Train Interactions: {dataset.n_train_interactions:>10,}", flush=True)
    print(f"    Test Interactions : {dataset.n_test_interactions:>10,}", flush=True)
    print(f"    Social Links      : {dataset.n_trust_links:>10,} (symmetric)", flush=True)

    density_r = dataset.train_csr.nnz / (dataset.num_users * dataset.num_items) * 100
    density_s = dataset.social_csr.nnz / (dataset.num_users ** 2) * 100
    print(f"    Interaction Density: {density_r:>9.4f}%", flush=True)
    print(f"    Social Density     : {density_s:>9.4f}%", flush=True)
    print(f"    Test Users        : {len(dataset.test_dict):>10,}", flush=True)
    print(f"    Data Load Time    : {data_time:>10.1f}s", flush=True)
    print(f"  {'='*60}\n", flush=True)

    # ==================================================================
    # Step 2: Initialize Models
    # ==================================================================
    print(f"[Step 2/4] Initializing models (dim={embedding_dim}, layers={num_layers}) ...\n", flush=True)

    all_models: Dict[str, BaseAdapter] = {}

    if "lightgcn" in models_to_run:
        all_models["lightgcn"] = VanillaLightGCNAdapter(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            lr=lr,
            reg=reg,
        )

    if "qrec_lightgcn" in models_to_run:
        all_models["qrec_lightgcn"] = QRecModelAdapter(
            model_name="LightGCN",
            qrec_path=qrec_path,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            lr=lr,
            reg=reg,
        )

    if "social_lightgcn" in models_to_run:
        all_models["social_lightgcn"] = SocialLightGCNAdapter(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            lr=lr,
            reg=reg,
        )

    model_names = [m.get_name() for m in all_models.values()]
    print(f"  Models registered: {', '.join(model_names)}\n", flush=True)

    # ==================================================================
    # Step 3: Train Models
    # ==================================================================
    print(f"[Step 3/4] Training ({n_epochs} epochs, batch_size={batch_size}) ...\n", flush=True)

    evaluator = ArenaEvaluator(k_list=k_list, max_eval_users=max_eval_users)
    results_list: List[Dict[str, Any]] = []

    for key, model in all_models.items():
        name = model.get_name()
        print(f"  {'~'*60}", flush=True)
        print(f"  Training: {name}", flush=True)
        print(f"  {'~'*60}", flush=True)

        t0 = time.time()
        model.fit(
            train_csr=dataset.train_csr,
            social_csr=dataset.social_csr,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        train_time = time.time() - t0
        print(f"  -> Training completed in {train_time:.1f}s\n", flush=True)

        # Evaluate
        print(f"  Evaluating: {name}", flush=True)
        t1 = time.time()
        metrics = evaluator.evaluate(
            model=model,
            test_dict=dataset.test_dict,
            train_dict=dataset.train_dict,
            num_items=dataset.num_items,
        )
        eval_time = time.time() - t1

        row: Dict[str, Any] = {
            "Model": name,
            "Train (s)": round(train_time, 1),
        }
        for metric_name, metric_val in metrics.items():
            row[metric_name] = round(metric_val, 4)
        results_list.append(row)

        print(f"  -> Evaluation completed in {eval_time:.1f}s", flush=True)
        for mk, mv in metrics.items():
            if "Precision" not in mk:  # skip precision in summary (detailed in table)
                print(f"     {mk}: {mv:.4f}", flush=True)
        print(flush=True)

    # ==================================================================
    # Step 4: Print Results
    # ==================================================================
    print(f"\n[Step 4/4] Results\n", flush=True)
    print_results_table(results_list, k_list=k_list, title="UNIFIED ARENA RESULTS -- CiaoDVD")

    print("  Arena completed successfully!", flush=True)
    return results_list


# ======================================================================
# CLI Entry Point
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Academic Benchmark Arena: Social-LightGCN vs. SOTA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data/ciao",
                        help="Directory for Ciao dataset files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per model")
    parser.add_argument("--dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--reg", type=float, default=1e-4,
                        help="L2 regularization weight")
    parser.add_argument("--k_core", type=int, default=5,
                        help="K-core filtering threshold")
    parser.add_argument("--max_eval_users", type=int, default=5000,
                        help="Max users for evaluation sampling")
    parser.add_argument("--qrec_path", type=str, default="external/QRec",
                        help="Path to cloned QRec repository")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["lightgcn", "qrec_lightgcn", "social_lightgcn"],
                        choices=["lightgcn", "qrec_lightgcn", "social_lightgcn"],
                        help="Which models to benchmark")

    args = parser.parse_args()

    run_arena(
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        k_core=args.k_core,
        max_eval_users=args.max_eval_users,
        qrec_path=args.qrec_path,
        models_to_run=args.models,
    )
