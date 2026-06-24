# Grand Arena Runner — Design Spec

Date: 2026-06-25

## Context: this is sub-project 5 of 5 (the finale)

Part of the "Grand Unified Benchmark Arena" initiative:

1. ~~`pipeline/utils/sparse_jaccard.py`~~ — DONE, merged.
2. ~~`dataset_factory.py` + consolidation of Ciao/Yelp/FilmTrust~~ — DONE, merged.
3. ~~`ImplicitTrustLoader` (Mode B)~~ — DONE, merged.
4. ~~New explicit-trust datasets (Epinions; Douban manual-only; Flixster deferred)~~ — DONE, merged.
5. **`grand_arena_runner.py` — config-driven orchestrator tying 1-4 together** (THIS SPEC)

## Research findings (verified against real code, not guessed)

Before designing anything, the actual model/evaluation landscape was inventoried directly:

- **Five model engines already exist** in `pipeline/engines/`, each implementing the
  shared `BaseRecommenderEngine` ABC (`fit`, `predict_rating`, `recommend_top_n`,
  `save_model`, `load_model`): `LightGCNEngine`, `TrustSVDEngine`,
  `SocialLightGCNEngine`, `FunkSVDEngine`, `SASRecEngine`.
- **Their `fit()` inputs are NOT uniform.** Each expects a different shape:
  - `LightGCNEngine.fit(data)`: dict with `"sym_adj_mat"`, `"interaction_matrix"`.
  - `SocialLightGCNEngine.fit(data)` / `TrustSVDEngine.fit(data)`: dict with
    `"interaction_matrix"`, `"trust_matrix"`.
  - `FunkSVDEngine.fit(data)`: a `pd.DataFrame` with columns exactly
    `[user_idx, item_idx, rating]` (built internally via `surprise.Dataset.load_from_df`
    with a hardcoded `Reader(rating_scale=(1, 5))`).
  - `SASRecEngine.fit(data)`: `Dict[user_idx -> chronologically-ordered item sequence]`.
- **`recommend_top_n(user_id, top_n)` already masks seen/training items internally in
  every engine** (confirmed by reading all five implementations) — e.g. LightGCN masks
  via `self._interaction_csr[user_id].indices`; FunkSVD via Surprise's trainset `ur`;
  TrustSVD via its own `_user_items`. This means a single evaluator can call
  `recommend_top_n` uniformly across all engines without re-implementing per-engine
  masking logic.
- **SASRec is not buildable today.** It needs chronologically-ordered per-user item
  sequences, but `pipeline/data_loaders/loader_utils.py::parse_rows` discards
  timestamps entirely — no loader in this initiative preserves temporal order. Per the
  same "no guessing" discipline used for Flixster/Douban-real-files/ML-1M, **SASRec is
  deferred entirely from this sub-project**, not approximated with a fake ordering.
- **Three pre-existing, mutually divergent Recall@K/NDCG@K evaluators already exist**
  (`pipeline/unified_arena/evaluator.py::ArenaEvaluator`, utility functions in
  `pipeline/engines/benchmark_arena.py`, `evaluate_ranking()` in
  `pipeline/academic_sandbox/run_yelp_benchmark.py`) — none of them operate against the
  `ArenaDataset` contract this whole initiative has built. A fresh, `ArenaDataset`-native
  evaluator is written rather than reusing or modifying any of the three (consistent
  with how every prior sub-project built new parallel code instead of touching the
  frozen arenas).
- **`ArenaDataset.mode` already exists and is already produced by both loaders**
  (`"explicit"` from `ExplicitTrustLoader`, `"implicit"` from `ImplicitTrustLoader`,
  shipped in sub-project 3) — `pipeline/data_loaders/base_loader.py`'s docstring is
  stale (still says implicit mode is "not produced by any loader yet"); this sub-project
  corrects that one-line docstring as a housekeeping fix while it's already reading this
  file closely, no functional change.
- **An existing auto-scaling precedent for large datasets** already exists in
  `pipeline/academic_sandbox/run_yelp_benchmark.py:175-179`:
  `if num_users > 10000: n_epochs = min(n_epochs, 15); batch_size = max(batch_size, 16384)`.
  This sub-project replicates that exact threshold and values rather than inventing new
  ones, since Epinions (~49k users) is now in scope and LightGCN's BPR sampler is an
  unvectorized Python loop.
- **`.gitignore` already ignores `models/*.csv`** (the existing convention — FilmTrust
  Social Arena already saves its results there). No new ignore rule is needed for the
  CSV output; the Markdown output is deliberately left tracked (see Output section).

## Decisions made (binding scope boundaries for this sub-project)

- **SASRec is deferred entirely** — no code, no config, no registry entry for it in this
  sub-project. Mode B therefore runs 4 models (Funk-SVD, LightGCN, TrustSVD,
  Social-LightGCN), not 5.
- **A fresh, `ArenaDataset`-native evaluator is written** (`pipeline/benchmarks/evaluation.py`),
  not a reuse/adaptation of any of the three existing evaluators.
- **Per-(dataset, model) failures are caught and logged, never abort the sweep.** Any
  exception during a single model's run (training or evaluation) is caught, logged as
  `ERROR`, recorded as a `FAILED` cell in that dataset's results block, and the sweep
  continues to the next model/dataset.
- **`ManualDownloadRequiredError` is caught separately and more specifically**, at the
  dataset-load level (before any model loop starts): logged as `WARNING` with the
  exception's full manual-instructions text, the dataset is recorded as `SKIPPED`, and
  the sweep continues to the next dataset.
- **Large-dataset auto-scaling**: `if dataset.num_users > 10000`, cap `n_epochs` at 15
  for every epoch-based engine (LightGCN, Social-LightGCN, TrustSVD) and raise
  `batch_size` to at least 16384 for the two engines that accept it (LightGCN,
  Social-LightGCN) — exact values from the existing Yelp benchmark precedent.
- **Results are saved to both `models/grand_arena_results.csv` (gitignored, matches the
  existing `models/*.csv` rule) and `models/grand_arena_results.md` (tracked in git —
  the human-facing deliverable meant for direct copy-paste into a README or slides)**, in
  addition to printing the Markdown table to stdout.
- **`base_loader.py`'s `ArenaDataset.mode` docstring is corrected** (one line, no
  functional change) since it's now factually wrong after sub-project 3 shipped.

## Architecture

### File structure

```
pipeline/benchmarks/
├── __init__.py            # NEW -- empty marker, matches data_loaders/, utils/
├── model_runner.py        # NEW -- per-model ArenaDataset -> engine.fit() translation,
│                           #        auto-scaling, train-time measurement
├── evaluation.py           # NEW -- Recall@10/NDCG@10/serving-latency
└── grand_arena_runner.py  # NEW -- CLI, orchestration loop, error containment,
                            #        table rendering, persistence
```

`model_runner.py` and `evaluation.py` are split out because each is independently usable
and testable without the CLI (e.g. a future script could import `evaluation.py` directly).
`grand_arena_runner.py` stays focused on orchestration/presentation, matching this whole
initiative's "smaller, focused files" pattern.

### `pipeline/benchmarks/model_runner.py`

```python
MODE_A_MODELS = ["lightgcn", "trustsvd", "social_lightgcn"]
MODE_B_MODELS = ["funksvd", "lightgcn", "trustsvd", "social_lightgcn"]

LARGE_DATASET_USER_THRESHOLD = 10_000  # matches run_yelp_benchmark.py precedent
LARGE_DATASET_MAX_EPOCHS = 15
LARGE_DATASET_MIN_BATCH_SIZE = 16384


def run_model(model_name: str, dataset: ArenaDataset) -> Tuple[BaseRecommenderEngine, float]:
    """
    Construct the engine named by model_name, build its fit() input from `dataset`,
    apply large-dataset auto-scaling if applicable, call fit(), and return
    (fitted_engine, train_seconds).

    Raises whatever the underlying engine's __init__/fit() raises -- the caller
    (grand_arena_runner.py) is responsible for catching and logging, not this function.
    """
```

Per-model `fit()` input construction (the actual translation logic this function owns):

- **`"lightgcn"`** -> `LightGCNEngine(num_users=dataset.num_users, num_items=dataset.num_items, **scaled_kwargs)`,
  `fit({"sym_adj_mat": dataset.get_sym_adj_mat(), "interaction_matrix": dataset.train_csr})`.
- **`"social_lightgcn"`** -> `SocialLightGCNEngine(num_users=dataset.num_users, num_items=dataset.num_items, **scaled_kwargs)`,
  `fit({"interaction_matrix": dataset.train_csr, "trust_matrix": dataset.social_csr})`.
- **`"trustsvd"`** -> `TrustSVDEngine(**scaled_kwargs)`,
  `fit({"interaction_matrix": dataset.train_csr, "trust_matrix": dataset.social_csr})`.
- **`"funksvd"`** -> `FunkSVDEngine()` (defaults; never auto-scaled -- it has no epoch/batch
  knobs that matter here), `fit(df)` where `df` is built via:
  ```python
  coo = dataset.train_csr.tocoo()
  df = pd.DataFrame({"user_idx": coo.row, "item_idx": coo.col, "rating": coo.data})
  ```

`scaled_kwargs` is computed once per dataset:
```python
def _scaled_kwargs(dataset: ArenaDataset, supports_batch_size: bool) -> dict:
    kwargs = {}
    if dataset.num_users > LARGE_DATASET_USER_THRESHOLD:
        kwargs["n_epochs"] = LARGE_DATASET_MAX_EPOCHS
        if supports_batch_size:
            kwargs["batch_size"] = LARGE_DATASET_MIN_BATCH_SIZE
    return kwargs
```
(Passed as `**kwargs` so unset keys fall back to each engine's own constructor defaults.)

Train time is measured by wrapping the `fit()` call in `time.perf_counter()`.

### `pipeline/benchmarks/evaluation.py`

```python
def evaluate_model(engine: BaseRecommenderEngine, dataset: ArenaDataset, k: int = 10) -> Dict[str, float]:
    """
    For every user_id in dataset.test_dict with a non-empty test item set:
      - time engine.recommend_top_n(user_id, top_n=k) via time.perf_counter()
      - compute hits against dataset.test_dict[user_id] (binary relevance)
      - accumulate Recall@k and NDCG@k
    Returns {"recall@10": float, "ndcg@10": float, "latency_ms": float} -- latency_ms
    is the mean per-call wall-clock time across all evaluated users.

    Relies on every BaseRecommenderEngine.recommend_top_n implementation already
    masking seen/training items internally (verified by reading all five engines
    during design) -- this function does NOT re-mask train_dict items itself.
    """
```

NDCG@k uses standard binary-relevance DCG (`1 / log2(rank + 1)` for each hit, divided by
the ideal DCG for `min(k, len(test_items))` relevant items) — no graded relevance exists
in this dataset family, so this is the correct, simplest formulation.

### `pipeline/benchmarks/grand_arena_runner.py`

CLI (`argparse`):
```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--datasets", nargs="+", help="Dataset names, e.g. --datasets filmtrust ciao ml-100k")
group.add_argument("--all", action="store_true", help="Run every dataset registered in either DatasetFactory registry")
```
`--all` resolves to `sorted(list(DATASET_REGISTRY.keys()) + list(IMPLICIT_DATASET_REGISTRY.keys()))`.

Orchestration loop (pseudocode, real control flow):
```python
for name in dataset_names:
    try:
        dataset = DatasetFactory.create(name).load()
    except ManualDownloadRequiredError as e:
        log.warning(f"[{name}] SKIPPED -- manual download required:\n{e}")
        results.record_skipped(name, str(e))
        continue

    print_dataset_metadata_banner(name, dataset)  # users, items, density%, social edges, mode

    models = MODE_A_MODELS if dataset.mode == "explicit" else MODE_B_MODELS
    for model_name in models:
        try:
            engine, train_seconds = model_runner.run_model(model_name, dataset)
            metrics = evaluation.evaluate_model(engine, dataset, k=10)
            results.record_success(name, model_name, train_seconds, metrics)
        except Exception as e:
            log.error(f"[{name}/{model_name}] FAILED: {e}")
            results.record_failed(name, model_name, str(e))
```

Dataset metadata banner prints: `num_users`, `num_items`,
`density% = 100 * train_csr.nnz / (num_users * num_items)`, `social_csr.nnz`
("Explicit trust edges" or "Synthetic (Jaccard) edges" depending on `dataset.mode`), and
a clear `Mode A: Explicit Trust` / `Mode B: Implicit Trust (ABLATION STUDY)` label.

### Output

One Markdown block per processed dataset (header states the dataset name + its Mode
label), with a table of Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) rows.
`FAILED` model rows show `FAILED` in the metric columns with the exception message
underneath. `SKIPPED` datasets get their own short block (dataset name + the manual
instructions text) instead of a model table. The full rendered Markdown is:
1. printed to stdout,
2. written to `models/grand_arena_results.md` (tracked in git -- the deliverable),
3. and the same per-cell data is written to `models/grand_arena_results.csv` (gitignored,
   matches the existing `models/*.csv` rule) with columns
   `dataset,mode,model,status,recall@10,ndcg@10,train_seconds,latency_ms,note`.

### `base_loader.py` housekeeping fix

One-line docstring correction in `ArenaDataset`'s `mode` field description (currently
says implicit mode is "not produced by any loader yet" -- false since sub-project 3).
No functional change.

## Out of scope

- SASRec — no working chronological-sequence data path exists anywhere in this
  initiative's loaders; deferred until timestamps are preserved by a future sub-project.
- Modifying any of the five existing `pipeline/engines/*_engine.py` files, or any of the
  three existing arena scripts (`unified_arena/`, `academic_sandbox/`, `filmtrust_arena/`)
  and their own evaluators -- all frozen, all untouched.
- `pipeline/run_pipeline.py` and `pipeline/engines/unified_data_loader.py` -- remain
  frozen production code, unrelated to this sub-project.
- Hyperparameter tuning or per-dataset model configuration beyond the large-dataset
  auto-scaling rule -- every engine uses its own constructor defaults otherwise.
- A `--quick`/reduced-epoch CLI flag -- the auto-scaling rule already handles the
  practical runtime concern for the one large dataset (Epinions) currently registered.

## Verification plan

No pytest framework — direct script execution with documented exact expected output,
consistent with every other module in this codebase.

1. **Single dataset, single mode, smoke test**: `--datasets filmtrust` (Mode A, small,
   fast) -- confirm all 3 Mode A models run to completion, the metadata banner prints
   correct real numbers, the Markdown table has 3 model rows with plausible (non-NaN,
   non-negative, Recall/NDCG in [0,1]) metrics, and both output files are written.
2. **Mode B smoke test**: `--datasets ml-100k` -- confirm all 4 Mode B models run
   (Funk-SVD, LightGCN, TrustSVD, Social-LightGCN against the synthetic Jaccard
   `social_csr`), the Mode B ablation label appears in the banner, metrics are plausible.
3. **Graceful skip**: `--datasets douban` (alone) -- confirm `ManualDownloadRequiredError`
   is caught, logged as WARNING with the full manual-instructions text, the dataset is
   recorded SKIPPED in the output (not crashing, not silently omitted), and the run exits
   cleanly with no model table for Douban.
4. **Large-dataset auto-scaling, observed not assumed**: `--datasets epinions` -- confirm
   the auto-scaling log line fires (`num_users` ~49k > 10,000), `n_epochs`/`batch_size`
   actually used by LightGCN/Social-LightGCN/TrustSVD match the scaled values (verified
   by inspecting the engine instances or their logged training output, not just trusting
   the kwarg was passed), and the full Mode A sweep completes in a practically verifiable
   time budget.
5. **Failure containment**: deliberately trigger a single model failure (e.g. monkeypatch
   or pass a malformed dataset to one engine in an isolated script) and confirm the
   sweep records a FAILED cell and continues to the next model/dataset rather than
   crashing the whole process.
6. **`--all` end-to-end**: run the full sweep once across every registered dataset
   (ciao, yelp, filmtrust, epinions, douban, ml-100k) and confirm the final Markdown/CSV
   correctly mixes SUCCESS, SKIPPED (douban), and any FAILED cells with no crash,
   matching the per-dataset/per-model counts expected from the registries' current
   contents.
7. **`base_loader.py` docstring fix**: confirm the corrected docstring text and that no
   other code references the old wording.
