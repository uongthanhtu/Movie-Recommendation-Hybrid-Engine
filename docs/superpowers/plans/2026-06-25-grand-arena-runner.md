# Grand Arena Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pipeline/benchmarks/grand_arena_runner.py`, a CLI orchestrator that loads datasets via `DatasetFactory`, trains and evaluates the correct model set per dataset mode, and renders a publication-ready Markdown/CSV summary table.

**Architecture:** Three new, independently-testable modules under a new `pipeline/benchmarks/` package: `evaluation.py` (Recall@10/NDCG@10/latency against `ArenaDataset`), `model_runner.py` (per-model `ArenaDataset -> fit()` input translation + large-dataset auto-scaling), and `grand_arena_runner.py` (CLI, orchestration loop, error containment, table rendering, persistence). SASRec is deferred entirely (no chronological-sequence data path exists in any loader yet).

**Tech Stack:** Python, `argparse`, `pandas`, `scipy.sparse`, the existing five `pipeline/engines/*_engine.py` model implementations (unmodified), `pipeline/data_loaders/` (unmodified except one docstring).

## Global Constraints

- Mode A models: `["lightgcn", "trustsvd", "social_lightgcn"]`. Mode B models: `["funksvd", "lightgcn", "trustsvd", "social_lightgcn"]`. SASRec is in neither list — no code, no config, no registry entry for it in this sub-project.
- Every `BaseRecommenderEngine.recommend_top_n` implementation already masks seen/training items internally — the new evaluator must NOT re-mask `train_dict` items itself.
- A fresh, `ArenaDataset`-native evaluator is written — do not import or adapt `pipeline/unified_arena/evaluator.py::ArenaEvaluator`, `pipeline/engines/benchmark_arena.py`'s functions, or `pipeline/academic_sandbox/run_yelp_benchmark.py::evaluate_ranking`.
- Large-dataset auto-scaling exact values (matching `pipeline/academic_sandbox/run_yelp_benchmark.py:175-179`'s existing precedent): `if dataset.num_users > 10_000: n_epochs = min(n_epochs, 15)`, and `batch_size = max(batch_size, 16384)` for the two engines that accept a `batch_size` kwarg (`lightgcn`, `social_lightgcn`).
- Per-dataset `ManualDownloadRequiredError` (from `pipeline.data_loaders.loader_utils`) is caught separately from per-model failures: logged as `WARNING`, dataset recorded `SKIPPED`, sweep continues. Any other exception during a single model's `run_model()`/`evaluate_model()` call is caught, logged as `ERROR`, that cell recorded `FAILED`, sweep continues — no single failure ever aborts the whole run.
- Results are saved to `models/grand_arena_results.csv` (gitignored — matches the existing `models/*.csv` rule) and `models/grand_arena_results.md` (tracked in git — the human-facing deliverable), in addition to printing the Markdown table to stdout.
- No new third-party dependencies. No pytest framework in this repo — verification is direct script execution with documented exact expected output, run via `py -3` with `PYTHONPATH=.` (plain `python` is not on PATH in this environment).
- None of the five existing `pipeline/engines/*_engine.py` files, the three existing arena scripts (`unified_arena/`, `academic_sandbox/`, `filmtrust_arena/`), `pipeline/run_pipeline.py`, or `pipeline/engines/unified_data_loader.py` are modified by this plan.

---

### Task 1: `evaluation.py` + package init + `base_loader.py` docstring fix

**Files:**
- Create: `pipeline/benchmarks/__init__.py`
- Create: `pipeline/benchmarks/evaluation.py`
- Modify: `pipeline/data_loaders/base_loader.py:36-37` (one-line docstring correction, no functional change)

**Interfaces:**
- Produces: `evaluate_model(engine: BaseRecommenderEngine, dataset: ArenaDataset, k: int = 10) -> Dict[str, float]` returning exactly the keys `f"recall@{k}"`, `f"ndcg@{k}"`, `"latency_ms"`. Consumed by Task 3's orchestration loop (always called with `k=10`, so the live keys are literally `"recall@10"`, `"ndcg@10"`, `"latency_ms"`).

- [ ] **Step 1: Create the package marker**

Create `pipeline/benchmarks/__init__.py` (empty file — matches `pipeline/data_loaders/__init__.py` and `pipeline/utils/__init__.py`'s existing convention):

```python
```

- [ ] **Step 2: Fix the stale docstring in `base_loader.py`**

In `pipeline/data_loaders/base_loader.py`, replace:

```python
        mode: "explicit" (real trust data) or "implicit" (Jaccard-derived, Mode B --
            not produced by any loader yet; reserved for a future sub-project).
```

With:

```python
        mode: "explicit" (real trust data, produced by ExplicitTrustLoader) or
            "implicit" (Jaccard-derived, Mode B ablation study, produced by
            ImplicitTrustLoader).
```

- [ ] **Step 3: Write `evaluation.py`**

Create `pipeline/benchmarks/evaluation.py`:

```python
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
```

- [ ] **Step 4: Write and run a verification script with a synthetic fake engine**

This tests the metric arithmetic deterministically, without needing a real trained
model. Run:

```bash
PYTHONPATH=. py -3 -c "
import math
from typing import Dict, List

from pipeline.benchmarks.evaluation import evaluate_model
from pipeline.engines.base_engine import BaseRecommenderEngine


class _FakeEngine(BaseRecommenderEngine):
    def __init__(self, fixed_recommendations: Dict[int, List[int]]):
        self._fixed = fixed_recommendations

    def fit(self, data):
        pass

    def predict_rating(self, user_id, item_id):
        return 0.0

    def recommend_top_n(self, user_id, top_n=10):
        return self._fixed.get(user_id, [])[:top_n]

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


class _FakeDataset:
    def __init__(self, test_dict):
        self.test_dict = test_dict


fixed = {
    0: [1, 9, 2, 8, 7, 6, 5, 4, 3, 0],
    1: [5, 1, 2, 3, 4, 6, 7, 8, 9, 0],
    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
test_dict = {0: {1, 2}, 1: {5}, 2: {99}, 3: set()}

engine = _FakeEngine(fixed)
dataset = _FakeDataset(test_dict)
metrics = evaluate_model(engine, dataset, k=10)
print('metrics:', metrics)

# Independently recompute expected recall/ndcg (only users 0,1,2 count -- user 3 is skipped)
cases = [
    ({1, 2}, fixed[0]),
    ({5}, fixed[1]),
    ({99}, fixed[2]),
]
recalls, ndcgs = [], []
for test_items, recommended in cases:
    hits = [1 if it in test_items else 0 for it in recommended[:10]]
    recalls.append(sum(hits) / len(test_items))
    dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
    n_ideal = min(10, len(test_items))
    idcg = sum(1.0 / math.log2(r + 2) for r in range(n_ideal))
    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
expected_recall = sum(recalls) / len(recalls)
expected_ndcg = sum(ndcgs) / len(ndcgs)
print(f'expected recall@10={expected_recall}, expected ndcg@10={expected_ndcg}')

assert math.isclose(metrics['recall@10'], expected_recall, rel_tol=1e-9)
assert math.isclose(metrics['ndcg@10'], expected_ndcg, rel_tol=1e-9)
assert metrics['latency_ms'] >= 0.0
print('Check 1 (recall@10 matches independently-computed expected value): PASS')
print('Check 2 (ndcg@10 matches independently-computed expected value): PASS')
print('Check 3 (latency_ms is non-negative): PASS')
print('Check 4 (user with empty test set was skipped, not counted): PASS' if abs(metrics['recall@10'] - 2/3) < 1e-9 else 'Check 4: FAIL')
"
```

Expected output (the printed `metrics` dict's exact float values will match
`expected recall@10=0.6666666666666666, expected ndcg@10=0.6399069297160626` since
both lines compute the same formula independently):
```
metrics: {'recall@10': 0.6666666666666666, 'ndcg@10': 0.6399069297160626, 'latency_ms': ...}
expected recall@10=0.6666666666666666, expected ndcg@10=0.6399069297160626
Check 1 (recall@10 matches independently-computed expected value): PASS
Check 2 (ndcg@10 matches independently-computed expected value): PASS
Check 3 (latency_ms is non-negative): PASS
Check 4 (user with empty test set was skipped, not counted): PASS
```
(`latency_ms`'s printed value will be some small positive float — that part of the
output is expected to vary run to run; everything else must match exactly.)

- [ ] **Step 5: Verify the docstring fix landed correctly**

Run:
```bash
grep -n "not produced by any loader yet" pipeline/data_loaders/base_loader.py
grep -n "produced by ImplicitTrustLoader" pipeline/data_loaders/base_loader.py
```
Expected: the first command prints nothing (no output, exit code 1 -- the stale phrase
is gone); the second command prints one matching line (the corrected text is present).

- [ ] **Step 6: Commit**

```bash
git add pipeline/benchmarks/__init__.py pipeline/benchmarks/evaluation.py pipeline/data_loaders/base_loader.py
git commit -m "feat(benchmarks): add ArenaDataset-native Recall@K/NDCG@K/latency evaluator"
```

---

### Task 2: `model_runner.py`

**Files:**
- Create: `pipeline/benchmarks/model_runner.py`

**Interfaces:**
- Consumes: `ArenaDataset` (`pipeline.data_loaders.base_loader`), `BaseRecommenderEngine` (`pipeline.engines.base_engine`).
- Produces: `MODE_A_MODELS: List[str]`, `MODE_B_MODELS: List[str]`,
  `run_model(model_name: str, dataset: ArenaDataset) -> Tuple[BaseRecommenderEngine, float]`
  (the float is training wall-clock seconds). Consumed by Task 3's orchestration loop.

- [ ] **Step 1: Write `model_runner.py`**

Create `pipeline/benchmarks/model_runner.py`:

```python
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
```

- [ ] **Step 2: Verify the auto-scaling logic in isolation (fast, no real training)**

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.benchmarks.model_runner import _scaled_kwargs, LARGE_DATASET_USER_THRESHOLD, LARGE_DATASET_MAX_EPOCHS, LARGE_DATASET_MIN_BATCH_SIZE


class _FakeDataset:
    def __init__(self, num_users):
        self.num_users = num_users


small = _FakeDataset(LARGE_DATASET_USER_THRESHOLD)  # exactly at threshold, not above
assert _scaled_kwargs(small, supports_batch_size=True) == {}
print('Check 1 (no scaling at/under threshold): PASS')

large = _FakeDataset(LARGE_DATASET_USER_THRESHOLD + 1)
assert _scaled_kwargs(large, supports_batch_size=True) == {'n_epochs': LARGE_DATASET_MAX_EPOCHS, 'batch_size': LARGE_DATASET_MIN_BATCH_SIZE}
assert _scaled_kwargs(large, supports_batch_size=False) == {'n_epochs': LARGE_DATASET_MAX_EPOCHS}
print('Check 2 (scaling above threshold, with/without batch_size support): PASS')
"
```
Expected output:
```
Check 1 (no scaling at/under threshold): PASS
Check 2 (scaling above threshold, with/without batch_size support): PASS
```

- [ ] **Step 3: Verify real end-to-end training against ML-100K (covers all 4 models)**

ML-100K already has a real, populated `social_csr` (synthesized via Jaccard by
`ImplicitTrustLoader`, sub-project 3), so this single dataset load exercises all four
`run_model` translation branches in one pass. If `data/ml-100k/` is not already
present in this worktree, this will trigger a real download (small, ~4.7MB, already
proven reliable in prior sub-projects).

Run:
```bash
PYTHONPATH=. py -3 -c "
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.benchmarks.model_runner import run_model, MODE_B_MODELS

dataset = DatasetFactory.create('ml-100k').load()

for model_name in MODE_B_MODELS:
    engine, train_seconds = run_model(model_name, dataset)
    assert train_seconds > 0.0, f'{model_name}: train_seconds must be positive, got {train_seconds}'
    recs = engine.recommend_top_n(0, top_n=10)
    assert isinstance(recs, list) and len(recs) <= 10, f'{model_name}: recommend_top_n returned {recs!r}'
    print(f'Check ({model_name}): PASS -- train_seconds={train_seconds:.2f}, sample_recs={recs}')
"
```
Expected output: the four engines' own training console logs (LightGCN/Social-LightGCN
print per-epoch loss lines; FunkSVD and TrustSVD train silently), followed by four
`Check (<model_name>): PASS -- train_seconds=..., sample_recs=[...]` lines (the exact
`train_seconds` values and recommended item IDs are real, non-deterministic outputs --
report what you observe, just confirm all four lines print with no traceback).

- [ ] **Step 4: Commit**

```bash
git add pipeline/benchmarks/model_runner.py
git commit -m "feat(benchmarks): add per-model ArenaDataset translation + large-dataset auto-scaling"
```

---

### Task 3: `grand_arena_runner.py`

**Files:**
- Create: `pipeline/benchmarks/grand_arena_runner.py`

**Interfaces:**
- Consumes: `evaluation.evaluate_model` (Task 1), `model_runner.run_model`,
  `model_runner.MODE_A_MODELS`, `model_runner.MODE_B_MODELS` (Task 2),
  `DatasetFactory.create(name).load() -> ArenaDataset`,
  `DATASET_REGISTRY`, `IMPLICIT_DATASET_REGISTRY` (`pipeline.data_loaders.dataset_configs`),
  `ManualDownloadRequiredError` (`pipeline.data_loaders.loader_utils`).
- Produces: `main(argv: Optional[List[str]] = None) -> _Results` (the CLI entry point;
  accepting an explicit `argv` makes it directly callable from a verification script
  without spawning a subprocess).

- [ ] **Step 1: Write `grand_arena_runner.py`**

Create `pipeline/benchmarks/grand_arena_runner.py`:

```python
"""
Grand Arena Runner -- config-driven CLI orchestrator for the Grand Unified Benchmark
Arena. Iterates over a selected list of datasets (or every registered dataset), loads
each via DatasetFactory, routes to the correct model set based on ArenaDataset.mode,
trains + evaluates each model, and renders a publication-ready Markdown summary table
(also saved alongside a CSV).

Per-dataset ManualDownloadRequiredError failures (e.g. Douban, which has no working
automated source -- see
docs/superpowers/specs/2026-06-24-epinions-douban-flixster-design.md) are caught and
logged as a graceful SKIP. Per-(dataset, model) failures during training or evaluation
are caught independently and logged as FAILED, so one bad combination never aborts the
rest of the sweep.

Usage:
    py -3 -m pipeline.benchmarks.grand_arena_runner --datasets filmtrust ciao ml-100k
    py -3 -m pipeline.benchmarks.grand_arena_runner --all
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pipeline.benchmarks import evaluation, model_runner
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY, IMPLICIT_DATASET_REGISTRY
from pipeline.data_loaders.dataset_factory import DatasetFactory
from pipeline.data_loaders.loader_utils import ManualDownloadRequiredError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("grand_arena_runner")

RESULTS_DIR = "models"
CSV_PATH = os.path.join(RESULTS_DIR, "grand_arena_results.csv")
MD_PATH = os.path.join(RESULTS_DIR, "grand_arena_results.md")


@dataclass
class _Row:
    dataset: str
    mode: str
    model: str
    status: str  # "success" | "failed"
    recall_at_10: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    train_seconds: Optional[float] = None
    latency_ms: Optional[float] = None
    note: str = ""


@dataclass
class _Results:
    rows: List[_Row] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (dataset, message)

    def record_success(self, dataset: str, mode: str, model: str, train_seconds: float, metrics: dict) -> None:
        self.rows.append(_Row(
            dataset=dataset, mode=mode, model=model, status="success",
            recall_at_10=metrics["recall@10"], ndcg_at_10=metrics["ndcg@10"],
            train_seconds=train_seconds, latency_ms=metrics["latency_ms"],
        ))

    def record_failed(self, dataset: str, mode: str, model: str, message: str) -> None:
        self.rows.append(_Row(dataset=dataset, mode=mode, model=model, status="failed", note=message))

    def record_skipped(self, dataset: str, message: str) -> None:
        self.skipped.append((dataset, message))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grand Unified Benchmark Arena orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datasets", nargs="+", metavar="NAME", help="Dataset names, e.g. --datasets filmtrust ciao ml-100k")
    group.add_argument("--all", action="store_true", help="Run every dataset registered in either DatasetFactory registry")
    return parser.parse_args(argv)


def _resolve_dataset_names(args: argparse.Namespace) -> List[str]:
    if args.all:
        return sorted(set(DATASET_REGISTRY.keys()) | set(IMPLICIT_DATASET_REGISTRY.keys()))
    return list(args.datasets)


def _print_metadata_banner(name: str, dataset) -> None:
    density_pct = 100.0 * dataset.train_csr.nnz / (dataset.num_users * dataset.num_items)
    mode_label = "Mode A: Explicit Trust" if dataset.mode == "explicit" else "Mode B: Implicit Trust (ABLATION STUDY)"
    edge_label = "Explicit trust edges" if dataset.mode == "explicit" else "Synthetic (Jaccard) edges"
    print(f"\n{'=' * 70}")
    print(f"[{name}] {mode_label}")
    print(f"  Users={dataset.num_users:,}  Items={dataset.num_items:,}  Density={density_pct:.4f}%")
    print(f"  {edge_label}: {dataset.social_csr.nnz:,}")
    print(f"{'=' * 70}")


def _run_sweep(dataset_names: List[str], results: _Results) -> None:
    for name in dataset_names:
        try:
            dataset = DatasetFactory.create(name).load()
        except ManualDownloadRequiredError as e:
            log.warning("[%s] SKIPPED -- manual download required:\n%s", name, e)
            results.record_skipped(name, str(e))
            continue

        _print_metadata_banner(name, dataset)

        models = model_runner.MODE_A_MODELS if dataset.mode == "explicit" else model_runner.MODE_B_MODELS
        for model_name in models:
            try:
                engine, train_seconds = model_runner.run_model(model_name, dataset)
                metrics = evaluation.evaluate_model(engine, dataset, k=10)
                results.record_success(name, dataset.mode, model_name, train_seconds, metrics)
                print(
                    f"  [{name}/{model_name}] recall@10={metrics['recall@10']:.4f} "
                    f"ndcg@10={metrics['ndcg@10']:.4f} train={train_seconds:.1f}s "
                    f"latency={metrics['latency_ms']:.2f}ms"
                )
            except Exception as e:
                log.error("[%s/%s] FAILED: %s", name, model_name, e)
                results.record_failed(name, dataset.mode, model_name, str(e))


def _render_markdown(results: _Results) -> str:
    lines = ["# Grand Arena Results", ""]
    datasets_seen = sorted({row.dataset for row in results.rows})

    for dataset in datasets_seen:
        dataset_rows = [r for r in results.rows if r.dataset == dataset]
        mode_label = "Mode A: Explicit Trust" if dataset_rows[0].mode == "explicit" else "Mode B: Implicit Trust (ABLATION STUDY)"
        lines.append(f"## {dataset} -- {mode_label}")
        lines.append("")
        lines.append("| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |")
        lines.append("|---|---|---|---|---|")
        for row in dataset_rows:
            if row.status == "success":
                lines.append(
                    f"| {row.model} | {row.recall_at_10:.4f} | {row.ndcg_at_10:.4f} "
                    f"| {row.train_seconds:.1f} | {row.latency_ms:.2f} |"
                )
            else:
                lines.append(f"| {row.model} | FAILED | FAILED | FAILED | FAILED ({row.note}) |")
        lines.append("")

    if results.skipped:
        lines.append("## Skipped Datasets")
        lines.append("")
        for name, message in results.skipped:
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"```\n{message}\n```")
            lines.append("")

    return "\n".join(lines)


def _write_csv(results: _Results, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mode", "model", "status", "recall@10", "ndcg@10", "train_seconds", "latency_ms", "note"])
        for row in results.rows:
            writer.writerow([
                row.dataset, row.mode, row.model, row.status,
                row.recall_at_10 if row.recall_at_10 is not None else "",
                row.ndcg_at_10 if row.ndcg_at_10 is not None else "",
                row.train_seconds if row.train_seconds is not None else "",
                row.latency_ms if row.latency_ms is not None else "",
                row.note,
            ])
        for name, message in results.skipped:
            writer.writerow([name, "", "", "skipped", "", "", "", "", message])


def main(argv: Optional[List[str]] = None) -> _Results:
    args = _parse_args(argv)
    dataset_names = _resolve_dataset_names(args)
    results = _Results()
    _run_sweep(dataset_names, results)

    markdown = _render_markdown(results)
    print("\n" + markdown)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    _write_csv(results, CSV_PATH)

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify a single-dataset Mode A smoke test (FilmTrust -- small, fast)**

Run:
```bash
PYTHONPATH=. py -3 -m pipeline.benchmarks.grand_arena_runner --datasets filmtrust
```
Expected: the metadata banner for `filmtrust` (`Mode A: Explicit Trust`, real
`Users=1,642 Items=2,071` -- matching sub-project 4's regression numbers), three
`[filmtrust/<model>] recall@10=... ndcg@10=... train=...s latency=...ms` progress lines
(one per `lightgcn`, `trustsvd`, `social_lightgcn`), then the rendered Markdown printed
to stdout with one `## filmtrust -- Mode A: Explicit Trust` section containing exactly
3 model rows, no `FAILED`/`SKIPPED` anywhere, no traceback. Confirm
`models/grand_arena_results.md` and `models/grand_arena_results.csv` were written:
```bash
ls -la models/grand_arena_results.md models/grand_arena_results.csv
```
Expected: both files exist with a non-zero size.

- [ ] **Step 3: Verify a single-dataset Mode B smoke test (ML-100K)**

Run:
```bash
PYTHONPATH=. py -3 -m pipeline.benchmarks.grand_arena_runner --datasets ml-100k
```
Expected: the metadata banner shows `Mode B: Implicit Trust (ABLATION STUDY)` and
`Synthetic (Jaccard) edges: ...`, four progress lines (one per `funksvd`, `lightgcn`,
`trustsvd`, `social_lightgcn`), and the rendered Markdown's `## ml-100k -- Mode B:
Implicit Trust (ABLATION STUDY)` section has exactly 4 model rows, no traceback.

- [ ] **Step 4: Verify the graceful-skip path (Douban)**

Run (ensure `data/douban/` does not exist first, so the manual-download failure path is
genuinely exercised, not a stale cached file):
```bash
rm -rf data/douban
PYTHONPATH=. py -3 -m pipeline.benchmarks.grand_arena_runner --datasets douban
```
Expected: a `WARNING: [douban] SKIPPED -- manual download required:` log line
containing the full manual-instructions text (the `113333244@qq.com` contact and the
dead-source citations), no model banner or progress lines for `douban` at all, and the
rendered Markdown has a `## Skipped Datasets` section with a `### douban` block
containing that same message -- no traceback, no `## douban -- Mode ...` model table.

- [ ] **Step 5: Verify failure containment with a deliberately broken model name**

This proves a single bad combination doesn't abort the sweep, using a direct call to
`_run_sweep` with a temporarily-corrupted model list (restored immediately after) rather
than modifying any shipped file:

```bash
PYTHONPATH=. py -3 -c "
from pipeline.benchmarks import grand_arena_runner as gar
from pipeline.benchmarks import model_runner

original_mode_a = model_runner.MODE_A_MODELS
model_runner.MODE_A_MODELS = ['lightgcn', 'definitely_not_a_real_model', 'trustsvd']
try:
    results = gar._Results()
    gar._run_sweep(['filmtrust'], results)
finally:
    model_runner.MODE_A_MODELS = original_mode_a

statuses = {row.model: row.status for row in results.rows}
print('statuses:', statuses)
assert statuses['lightgcn'] == 'success', statuses
assert statuses['definitely_not_a_real_model'] == 'failed', statuses
assert statuses['trustsvd'] == 'success', statuses
print('Check 1 (bad model name failed but did not abort lightgcn/trustsvd): PASS')
"
```
Expected output: a `lightgcn` progress line, an `ERROR: [filmtrust/definitely_not_a_real_model] FAILED: ...` log line, a `trustsvd` progress line, then:
```
statuses: {'lightgcn': 'success', 'definitely_not_a_real_model': 'failed', 'trustsvd': 'success'}
Check 1 (bad model name failed but did not abort lightgcn/trustsvd): PASS
```

- [ ] **Step 6: Verify large-dataset auto-scaling actually fires for Epinions**

Epinions has ~49k users (well over the 10,000 threshold) and will trigger a real
network download if `data/epinions/` is not already present in this worktree (already
proven reliable in sub-project 4). This step also serves as the full Mode A sweep
against the largest registered dataset.

Run:
```bash
PYTHONPATH=. py -3 -m pipeline.benchmarks.grand_arena_runner --datasets epinions
```
Expected: the metadata banner shows `Users=49,289` (or the real current count),
`Mode A: Explicit Trust`; LightGCN's and Social-LightGCN's own per-epoch console logs
show no more than 15 epochs printed (their code prints every 10th epoch plus epoch 1,
so at most 2 logged lines per model: epoch 1 and epoch 10 -- confirm no epoch number
above 15 ever appears); all three Mode A models complete with a `recall@10=...`
progress line and no traceback. Report the actual wall-clock time this step took in
your task report.

- [ ] **Step 7: Verify the full `--all` sweep mixes SUCCESS, SKIPPED, and (if any) FAILED**

Run:
```bash
PYTHONPATH=. py -3 -m pipeline.benchmarks.grand_arena_runner --all
```
Expected: banners and progress lines for `ciao`, `yelp`, `filmtrust`, `epinions`
(Mode A) and `ml-100k` (Mode B), a `WARNING` skip line for `douban`, the final rendered
Markdown contains 5 dataset sections (`ciao`, `yelp`, `filmtrust`, `epinions`,
`ml-100k`) plus one `## Skipped Datasets` section with `### douban`, and the process
exits with no unhandled traceback. Report the real per-cell metric values you observe
in your task report -- they are genuine, data-dependent outputs, not pre-determined.

- [ ] **Step 8: Commit the code**

```bash
git add pipeline/benchmarks/grand_arena_runner.py
git commit -m "feat(benchmarks): add grand_arena_runner CLI orchestrator"
```

Confirm `git status` shows `models/grand_arena_results.csv` as ignored/untracked (NOT
staged) -- it must never be committed, per the existing `models/*.csv` gitignore rule.

- [ ] **Step 9: Commit the benchmark deliverable from the `--all` run**

Step 7's `--all` run is the most complete sweep performed in this task (every
registered dataset). Commit the `models/grand_arena_results.md` it produced as the
project's benchmark snapshot -- this file is intentionally tracked (not gitignored),
per the design's "human-facing deliverable" decision:

```bash
git add models/grand_arena_results.md
git commit -m "docs(benchmarks): add Grand Arena results snapshot from --all sweep"
```
