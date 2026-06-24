# FilmTrust Social Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "Social Arena" benchmark on the FilmTrust dataset (real, explicit trust network) to evaluate TrustSVD and Social-LightGCN against a no-social LightGCN baseline, and trim the existing Classic Arena (MovieLens) down to non-social models only — resolving the academic-validity problem of benchmarking Social-Aware models against a Jaccard-fabricated trust graph.

**Architecture:** New self-contained package `pipeline/filmtrust_arena/` (mirroring the existing `pipeline/unified_arena/` and `pipeline/academic_sandbox/` packages) bundles a `FilmTrustLoader` (download/parse/split into scipy CSR matrices) and a `run_filmtrust.py` CLI orchestrator. The orchestrator reuses the **existing, unmodified** production engines (`LightGCNEngine`, `TrustSVDEngine`, `SocialLightGCNEngine`) and the existing `recall_at_k`/`ndcg_at_k` helpers from `benchmark_arena.py` — no new model code. `pipeline/engines/benchmark_arena.py` is trimmed to drop the two social engines. Spec: `docs/superpowers/specs/2026-06-24-filmtrust-social-arena-design.md`.

**Tech Stack:** Python, PyTorch, scipy.sparse, pandas, urllib.request (no new dependencies).

## Global Constraints

- No new third-party dependencies. Use `urllib.request` (not `requests`), `scipy.sparse`, `pandas`, `torch` — all already in `requirements.txt`.
- `pipeline/run_pipeline.py` and `app/main.py` (production serving) are out of scope — do not modify.
- `pipeline/engines/unified_data_loader.py` is out of scope — do not modify. `build_implicit_trust_matrix` (Jaccard) stays, still used by `run_pipeline.py`.
- New code lives under `pipeline/filmtrust_arena/`, mirroring the structure of `pipeline/unified_arena/` and `pipeline/academic_sandbox/`.
- No fabricated benchmark numbers anywhere (code output or README) — every metric printed or written to README must come from an actually-executed run.
- FilmTrust download URLs (verified against the `guoguibing/librec` GitHub repo — do not substitute a different mirror):
  - `https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/ratings.txt` — space-delimited, 3 columns `userId itemId rating`.
  - `https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/trust.txt` — space-delimited, 3 columns `trustorId trusteeId trustValue` (trustValue always `1`, directed).
- This codebase has no unit test framework (confirmed: no pytest config, no `tests/` directory). Verification in this plan uses direct script execution with documented expected output, matching how every other arena script (`run_yelp_benchmark.py`, `run_arena.py`, `benchmark_arena.py`) is verified in this repo.

---

### Task 1: FilmTrustLoader

**Files:**
- Create: `pipeline/filmtrust_arena/__init__.py`
- Create: `pipeline/filmtrust_arena/filmtrust_loader.py`

**Interfaces:**
- Produces: `FilmTrustLoader` class with:
  - `__init__(self, data_dir: str = "data/filmtrust", test_ratio: float = 0.2, seed: int = 42)`
  - `download(self, force: bool = False) -> None`
  - `load_data(self) -> None` (must be called after `download()`; populates `self.num_users`, `self.num_items`)
  - `get_train_interaction_matrix(self) -> scipy.sparse.csr_matrix` — shape `(num_users, num_items)`, explicit rating values
  - `get_sym_adj_mat(self) -> scipy.sparse.csr_matrix` — shape `(num_users+num_items, num_users+num_items)`, symmetric-normalized bipartite adjacency from train interactions
  - `get_trust_matrix(self) -> scipy.sparse.csr_matrix` — shape `(num_users, num_users)`, raw (unnormalized) symmetric trust weights
  - `get_test_dict(self) -> Dict[int, Set[int]]` — `{user_idx: set(item_idx)}` of held-out test ratings

- [ ] **Step 1: Create the package `__init__.py`**

```python
# FilmTrust Social Arena — Explicit-trust benchmark for TrustSVD / Social-LightGCN.
```

- [ ] **Step 2: Write `pipeline/filmtrust_arena/filmtrust_loader.py`**

```python
"""
FilmTrust Data Loader -- Isolated downloader & parser for the LibRec FilmTrust dataset.

Downloads the FilmTrust dataset (Guo, Zhang & Yorke-Smith, IJCAI 2013) directly from
the official LibRec GitHub repository, then parses ratings.txt and trust.txt into
contiguous-index scipy sparse matrices ready for TrustSVD and Social-LightGCN training.

Unlike pipeline/engines/unified_data_loader.py::build_implicit_trust_matrix(), which
fabricates a "trust" network via Jaccard similarity on co-interacted items, this loader
parses FilmTrust's real, explicit, user-asserted trust network -- the entire point of
the Social Arena is to evaluate Social-Aware models against actual trust data instead of
a collaborative-filtering proxy.

Official FilmTrust format (space-delimited):
    ratings.txt:  userId  itemId  rating       (rating in 0.5 increments, e.g. "1 3 3.5")
    trust.txt:    trustorId  trusteeId  trustValue   (trustValue always 1, directed)

FilmTrust has no timestamp column, so unlike the Classic Arena's leave-last-one-out
split, this loader performs a stratified 80/20 per-user split (same approach as
pipeline/academic_sandbox/yelp_data_loader.py).

Reference:
    Guo, G., Zhang, J., & Yorke-Smith, N. (2013). A Novel Bayesian Similarity Measure
    for Recommender Systems. IJCAI 2013.
    Repository: https://github.com/guoguibing/librec
"""
import os
import urllib.request
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


RATINGS_URL = (
    "https://raw.githubusercontent.com/guoguibing/librec/master/"
    "librec/demo/Datasets/FilmTrust/ratings.txt"
)
TRUST_URL = (
    "https://raw.githubusercontent.com/guoguibing/librec/master/"
    "librec/demo/Datasets/FilmTrust/trust.txt"
)


class FilmTrustLoader:
    """
    End-to-end FilmTrust dataset handler for the Social Arena.

    Lifecycle:
        1. download()    -- fetch ratings.txt / trust.txt into *data_dir*
        2. load_data()   -- parse, build ID mappings, stratified 80/20 split
        3. get_train_interaction_matrix() / get_sym_adj_mat() / get_trust_matrix()
        4. get_test_dict() -- {user_idx: set(item_idx)} for ranking evaluation
    """

    def __init__(self, data_dir: str = "data/filmtrust", test_ratio: float = 0.2, seed: int = 42):
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.seed = seed

        self.ratings_path: str = os.path.join(data_dir, "ratings.txt")
        self.trust_path: str = os.path.join(data_dir, "trust.txt")

        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.num_users: int = 0
        self.num_items: int = 0

        self._df_train: pd.DataFrame = pd.DataFrame()
        self._df_test: pd.DataFrame = pd.DataFrame()
        self._df_trust: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> None:
        """Download ratings.txt and trust.txt from the LibRec GitHub repo."""
        os.makedirs(self.data_dir, exist_ok=True)

        if not force and os.path.exists(self.ratings_path) and os.path.exists(self.trust_path):
            print(f"  [FilmTrustLoader] Dataset files already present in {self.data_dir}")
            return

        for url, dest, label in (
            (RATINGS_URL, self.ratings_path, "ratings.txt"),
            (TRUST_URL, self.trust_path, "trust.txt"),
        ):
            print(f"  [FilmTrustLoader] Downloading {label} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; FilmTrustLoader/1.0)"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download {label} from {url}: {exc}\n"
                    f"Manual fallback: download the file yourself and place it at {dest}"
                ) from exc

            with open(dest, "wb") as f:
                f.write(data)
            print(f"  [FilmTrustLoader] Saved {len(data):,} bytes -> {dest}")

    # ------------------------------------------------------------------
    # 2. Parse & Split
    # ------------------------------------------------------------------
    def load_data(self) -> None:
        """Parse text files, build contiguous ID mappings, stratified 80/20 split."""
        df_all = self._read_space_delimited(self.ratings_path, ["user", "item", "rating"])
        print(f"  [FilmTrustLoader] Loaded {len(df_all):,} ratings from {self.ratings_path}")

        self._df_trust = self._read_space_delimited(self.trust_path, ["src", "dst", "weight"])
        print(f"  [FilmTrustLoader] Loaded {len(self._df_trust):,} trust links from {self.trust_path}")

        all_users: Set[str] = set(df_all["user"].unique())
        all_users.update(self._df_trust["src"].unique())
        all_users.update(self._df_trust["dst"].unique())
        all_items: Set[str] = set(df_all["item"].unique())

        sorted_users = sorted(all_users, key=lambda x: int(x))
        sorted_items = sorted(all_items, key=lambda x: int(x))
        self.user_map = {uid: idx for idx, uid in enumerate(sorted_users)}
        self.item_map = {iid: idx for idx, iid in enumerate(sorted_items)}
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)

        df_all["u_idx"] = df_all["user"].map(self.user_map)
        df_all["i_idx"] = df_all["item"].map(self.item_map)

        self._df_train, self._df_test = self._stratified_split(df_all, self.test_ratio, self.seed)

        self._df_trust["s_idx"] = self._df_trust["src"].map(self.user_map)
        self._df_trust["d_idx"] = self._df_trust["dst"].map(self.user_map)

        print(f"\n  [FilmTrustLoader] Dataset statistics:")
        print(f"    Users : {self.num_users:,}")
        print(f"    Items : {self.num_items:,}")
        print(f"    Train : {len(self._df_train):,} ratings")
        print(f"    Test  : {len(self._df_test):,} ratings")
        print(f"    Trust : {len(self._df_trust):,} directed trust edges")

    @staticmethod
    def _read_space_delimited(path: str, columns: List[str]) -> pd.DataFrame:
        """Read a 3-column space-delimited FilmTrust file."""
        rows: List[Tuple[str, str, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    rows.append((parts[0], parts[1], float(parts[2])))
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _stratified_split(
        df: pd.DataFrame, test_ratio: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Leave-N-out split: for each user, hold out test_ratio of their ratings."""
        rng = np.random.default_rng(seed)
        train_indices: List[int] = []
        test_indices: List[int] = []

        for _, group in df.groupby("u_idx"):
            indices = group.index.tolist()
            n_test = max(1, int(len(indices) * test_ratio))

            if len(indices) < 2:
                train_indices.extend(indices)
            else:
                rng.shuffle(indices)
                test_indices.extend(indices[:n_test])
                train_indices.extend(indices[n_test:])

        return (
            df.loc[train_indices].reset_index(drop=True),
            df.loc[test_indices].reset_index(drop=True),
        )

    # ------------------------------------------------------------------
    # 3. Build Sparse Matrices
    # ------------------------------------------------------------------
    def get_train_interaction_matrix(self) -> sp.csr_matrix:
        """Build a (num_users x num_items) CSR matrix of explicit train ratings."""
        rows = self._df_train["u_idx"].values.astype(np.int64)
        cols = self._df_train["i_idx"].values.astype(np.int64)
        vals = self._df_train["rating"].values.astype(np.float32)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_items))

    def get_sym_adj_mat(self) -> sp.csr_matrix:
        """
        Build the symmetric-normalized bipartite adjacency matrix required by
        LightGCNEngine.fit(), built from TRAIN interactions only.

        Returns:
            sp.csr_matrix of shape (num_users + num_items, num_users + num_items)
        """
        rows = self._df_train["u_idx"].values
        cols = self._df_train["i_idx"].values

        R = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.num_users, self.num_items),
        )
        adj_mat = sp.bmat([[None, R], [R.T, None]], format="csr")

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        return D_inv_sqrt.dot(adj_mat).dot(D_inv_sqrt).tocsr()

    def get_trust_matrix(self) -> sp.csr_matrix:
        """
        Build a symmetric undirected trust matrix (num_users x num_users) from the
        directed FilmTrust trust.txt edges: A_social = A_trust + A_trust^T, clipped
        to binary. Returns RAW (unnormalized) weights -- TrustSVDEngine and
        SocialLightGCNEngine normalize internally during fit().
        """
        if len(self._df_trust) == 0:
            return sp.csr_matrix((self.num_users, self.num_users), dtype=np.float32)

        rows = self._df_trust["s_idx"].values.astype(np.int64)
        cols = self._df_trust["d_idx"].values.astype(np.int64)
        vals = self._df_trust["weight"].values.astype(np.float32)

        A_trust = sp.coo_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_users))
        A_social = (A_trust + A_trust.T).tocsr()
        A_social.data = np.minimum(A_social.data, 1.0)
        return A_social

    # ------------------------------------------------------------------
    # 4. Test Dictionary for Ranking Evaluation
    # ------------------------------------------------------------------
    def get_test_dict(self) -> Dict[int, Set[int]]:
        """Return {user_idx: set(item_idx)} of held-out test ratings."""
        test_dict: Dict[int, Set[int]] = {}
        for _, row in self._df_test.iterrows():
            test_dict.setdefault(int(row["u_idx"]), set()).add(int(row["i_idx"]))
        return test_dict
```

Note: a `get_train_dict()` was mentioned in the design spec for symmetry with `YelpDataLoader`, but it is intentionally omitted here — `LightGCNEngine`, `TrustSVDEngine`, and `SocialLightGCNEngine` all already mask previously-seen (train) items internally inside `recommend_top_n()` using the `interaction_matrix` passed to `fit()`, so a separate train-exclusion dict would be unused dead code (verified by reading all three engines' `recommend_top_n` in `pipeline/engines/*.py`).

- [ ] **Step 3: Verify by running a real download + shape check**

Run:
```bash
python -c "
from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader
loader = FilmTrustLoader(data_dir='data/filmtrust')
loader.download()
loader.load_data()
inter = loader.get_train_interaction_matrix()
adj = loader.get_sym_adj_mat()
trust = loader.get_trust_matrix()
test_dict = loader.get_test_dict()
assert inter.shape == (loader.num_users, loader.num_items)
assert adj.shape == (loader.num_users + loader.num_items, loader.num_users + loader.num_items)
assert trust.shape == (loader.num_users, loader.num_users)
assert (trust != trust.T).nnz == 0, 'trust matrix must be symmetric'
assert len(test_dict) > 0
print('OK', loader.num_users, loader.num_items, inter.nnz, trust.nnz, len(test_dict))
"
```
Expected: downloads `data/filmtrust/ratings.txt` and `data/filmtrust/trust.txt`, prints dataset statistics, then a final line `OK 1508 2071 <train_nnz> <trust_nnz> <num_test_users>` with no assertion errors (exact `num_users`/`num_items` may differ slightly if filtering changes; the key check is that the script completes with `OK` and no `AssertionError`/exception).

- [ ] **Step 4: Commit**

```bash
git add pipeline/filmtrust_arena/__init__.py pipeline/filmtrust_arena/filmtrust_loader.py
git commit -m "feat(filmtrust): add FilmTrust explicit-trust data loader

Parses the real FilmTrust ratings/trust network from the LibRec
GitHub repo into CSR matrices, with no Jaccard-derived trust
generation."
```

---

### Task 2: `run_filmtrust.py` Social Arena orchestrator

**Files:**
- Create: `pipeline/filmtrust_arena/run_filmtrust.py`

**Interfaces:**
- Consumes: `FilmTrustLoader` from Task 1 (`pipeline.filmtrust_arena.filmtrust_loader`); `LightGCNEngine` (`pipeline.engines.lightgcn_engine`), `TrustSVDEngine` (`pipeline.engines.trust_svd_engine`), `SocialLightGCNEngine` (`pipeline.engines.social_lightgcn_engine`) — all unmodified, each implementing `BaseRecommenderEngine.fit(data: dict)` / `recommend_top_n(user_id: int, top_n: int) -> List[int]`; `recall_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float` and `ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float` from `pipeline.engines.benchmark_arena`.
- Produces: `run_social_arena(...) -> pandas.DataFrame` callable, and a CLI entry point. Writes `models/filmtrust_arena_results.csv`.

- [ ] **Step 1: Write `pipeline/filmtrust_arena/run_filmtrust.py`**

```python
"""
Run FilmTrust Benchmark — Social Arena for Social-Aware Recommendation Models.

Orchestrates end-to-end:
  1. Download the FilmTrust dataset (explicit ratings + explicit trust network)
  2. Parse and index into contiguous sparse matrices
  3. Train 3 engines: LightGCN (no-social baseline), TrustSVD, Social-LightGCN
  4. Evaluate Recall@K, NDCG@K, and inference latency on the held-out test set
  5. Print a comparative results table

This script is STANDALONE and never modifies production code or the Classic Arena.

Usage:
    python -m pipeline.filmtrust_arena.run_filmtrust
    python -m pipeline.filmtrust_arena.run_filmtrust --epochs 30 --dim 64 -k 10
"""
import os
import sys
import time
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.filmtrust_arena.filmtrust_loader import FilmTrustLoader
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.trust_svd_engine import TrustSVDEngine
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine
from pipeline.engines.benchmark_arena import recall_at_k, ndcg_at_k


def run_social_arena(
    data_dir: str = "data/filmtrust",
    n_epochs: int = 30,
    embedding_dim: int = 64,
    num_layers: int = 3,
    k: int = 10,
    max_eval_users: int = 500,
) -> pd.DataFrame:
    """Full Social Arena benchmark: load FilmTrust, train 3 engines, evaluate, compare."""
    print("=" * 70)
    print("  SOCIAL ARENA -- FILMTRUST DATASET (Explicit Trust Network)")
    print("  LightGCN (No-Social Baseline)  vs.  TrustSVD  vs.  Social-LightGCN")
    print("=" * 70, flush=True)

    print("\n[Step 1/4] Loading FilmTrust dataset ...", flush=True)
    loader = FilmTrustLoader(data_dir=data_dir)
    loader.download()
    loader.load_data()

    num_users = loader.num_users
    num_items = loader.num_items

    interaction_csr = loader.get_train_interaction_matrix()
    sym_adj_mat = loader.get_sym_adj_mat()
    trust_csr = loader.get_trust_matrix()
    test_dict = loader.get_test_dict()

    print(f"\n  Data Summary:")
    print(f"    Users        : {num_users:>8,}")
    print(f"    Items        : {num_items:>8,}")
    print(f"    Train ratings: {interaction_csr.nnz:>8,}")
    print(f"    Trust edges  : {trust_csr.nnz:>8,} (symmetrized)")
    print(f"    Test users   : {len(test_dict):>8,}")

    print(f"\n[Step 2/4] Initializing engines (dim={embedding_dim}, layers={num_layers}) ...")
    engines: Dict[str, Any] = {
        "LightGCN (No-Social)": LightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, num_layers=num_layers, n_epochs=n_epochs,
        ),
        "TrustSVD": TrustSVDEngine(n_factors=embedding_dim, n_epochs=n_epochs),
        "Social-LightGCN": SocialLightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, num_layers=num_layers, n_epochs=n_epochs,
        ),
    }

    print("\n[Step 3/4] Training & evaluating ...", flush=True)
    results: List[Dict[str, Any]] = []
    eval_users = list(test_dict.keys())[:max_eval_users]

    for name, engine in engines.items():
        print(f"\n{'=' * 50}")
        print(f"  Training: {name}")
        print(f"{'=' * 50}")

        t0 = time.time()
        if name == "LightGCN (No-Social)":
            engine.fit({"sym_adj_mat": sym_adj_mat, "interaction_matrix": interaction_csr})
        else:
            engine.fit({"interaction_matrix": interaction_csr, "trust_matrix": trust_csr})
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        print(f"  Evaluating {name} ...")
        recalls: List[float] = []
        ndcgs: List[float] = []
        latencies: List[float] = []
        for uid in eval_users:
            ground_truth = test_dict[uid]

            t1 = time.perf_counter()
            top_n = engine.recommend_top_n(uid, top_n=k)
            lat = (time.perf_counter() - t1) * 1000

            recalls.append(recall_at_k(top_n, ground_truth, k))
            ndcgs.append(ndcg_at_k(top_n, ground_truth, k))
            latencies.append(lat)

        result = {
            "Engine": name,
            "Train Time (s)": round(train_time, 1),
            f"Recall@{k}": round(float(np.mean(recalls)), 4),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
            "Latency (ms)": round(float(np.mean(latencies)), 2),
            "P95 Latency (ms)": round(float(np.percentile(latencies, 95)), 2),
        }
        results.append(result)
        print(f"    Recall@{k}: {result[f'Recall@{k}']:.4f}")
        print(f"    NDCG@{k}:   {result[f'NDCG@{k}']:.4f}")
        print(f"    Latency:   {result['Latency (ms)']:.2f}ms (P95: {result['P95 Latency (ms)']:.2f}ms)")

    print("\n" + "=" * 70)
    print("[Step 4/4] BENCHMARK RESULTS -- FILMTRUST SOCIAL ARENA")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    csv_path = "models/filmtrust_arena_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FilmTrust Social Arena: LightGCN vs. TrustSVD vs. Social-LightGCN"
    )
    parser.add_argument("--data_dir", type=str, default="data/filmtrust", help="Directory for FilmTrust dataset files")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("-k", type=int, default=10, help="Top-K for Recall@K / NDCG@K")
    args = parser.parse_args()

    run_social_arena(
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        k=args.k,
    )
```

- [ ] **Step 2: Run the full Social Arena end-to-end**

Run: `python -m pipeline.filmtrust_arena.run_filmtrust`

Expected: prints `[Step 1/4]` through `[Step 4/4]`, trains all 3 engines without exceptions, prints a final ASCII results table with columns `Engine | Train Time (s) | Recall@10 | NDCG@10 | Latency (ms) | P95 Latency (ms)` for `LightGCN (No-Social)`, `TrustSVD`, `Social-LightGCN`, and writes `models/filmtrust_arena_results.csv`. **Record the exact printed numbers** — they are needed verbatim for Task 4.

- [ ] **Step 3: Commit**

```bash
git add pipeline/filmtrust_arena/run_filmtrust.py
git commit -m "feat(filmtrust): add Social Arena CLI orchestrator

Trains LightGCN (no-social baseline), TrustSVD, and Social-LightGCN
on FilmTrust's real trust network and prints a comparative
Recall@10/NDCG@10/latency table. Reuses the existing production
engines and benchmark_arena's recall_at_k/ndcg_at_k unchanged."
```

---

### Task 3: Trim Classic Arena to non-social engines

**Files:**
- Modify: `pipeline/engines/benchmark_arena.py:1-28` (module docstring + imports)
- Modify: `pipeline/engines/benchmark_arena.py:162-178` (engines dict)
- Modify: `pipeline/engines/benchmark_arena.py:188-208` (per-engine fit dispatch)

**Interfaces:**
- Consumes: nothing new — pure removal of `TrustSVDEngine`/`SocialLightGCNEngine` usage from this file. `recall_at_k`/`ndcg_at_k`/`run_arena` signatures are unchanged (Task 2 already depends on `recall_at_k`/`ndcg_at_k` staying in this file with their current signatures).

- [ ] **Step 1: Update the module docstring**

In `pipeline/engines/benchmark_arena.py`, replace:
```python
"""
Benchmark Arena — Multi-engine evaluation orchestrator.

Runs all 4 engines on the same dataset splits, measures:
  - RMSE / MAE (rating prediction accuracy)
  - Recall@K / NDCG@K (ranking quality)
  - ILD@K (intra-list diversity)
  - Latency (inference speed per user)

Usage:
    python -m pipeline.engines.benchmark_arena
"""
```
with:
```python
"""
Benchmark Arena — Multi-engine evaluation orchestrator (Classic Arena).

Compares non-social engines (Funk-SVD, LightGCN, SASRec) on MovieLens-100k, measures:
  - RMSE / MAE (rating prediction accuracy)
  - Recall@K / NDCG@K (ranking quality)
  - ILD@K (intra-list diversity)
  - Latency (inference speed per user)

Social-Aware models (TrustSVD, Social-LightGCN) are intentionally NOT benchmarked here:
MovieLens has no real social graph, and fabricating one via Jaccard similarity on
co-interacted items (see unified_data_loader.py::build_implicit_trust_matrix) is not a
scientifically valid trust network. Those models are evaluated instead in
pipeline/filmtrust_arena/run_filmtrust.py against FilmTrust's real, explicit trust data.

Usage:
    python -m pipeline.engines.benchmark_arena
"""
```

- [ ] **Step 2: Drop the two social-engine imports**

Replace:
```python
from pipeline.engines.unified_data_loader import UnifiedDataLoader
from pipeline.engines.funk_svd_engine import FunkSVDEngine
from pipeline.engines.trust_svd_engine import TrustSVDEngine
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.sasrec_engine import SASRecEngine
from pipeline.engines.social_lightgcn_engine import SocialLightGCNEngine
```
with:
```python
from pipeline.engines.unified_data_loader import UnifiedDataLoader
from pipeline.engines.funk_svd_engine import FunkSVDEngine
from pipeline.engines.lightgcn_engine import LightGCNEngine
from pipeline.engines.sasrec_engine import SASRecEngine
```

- [ ] **Step 3: Drop the two social engines from the `engines` dict**

Replace:
```python
    engines: Dict[str, Any] = {
        "Funk-SVD": FunkSVDEngine(n_factors=50, n_epochs=20),
        "TrustSVD": TrustSVDEngine(n_factors=50, n_epochs=30, lr=0.005, reg=0.02),
        "LightGCN": LightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=64, num_layers=3, n_epochs=30,
        ),
        "Social-LightGCN": SocialLightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=64, num_layers=3, n_epochs=30,
        ),
        "SASRec": SASRecEngine(
            num_items=num_items, max_seq_len=50, hidden_dim=50, n_epochs=30,
        ),
    }
```
with:
```python
    engines: Dict[str, Any] = {
        "Funk-SVD": FunkSVDEngine(n_factors=50, n_epochs=20),
        "LightGCN": LightGCNEngine(
            num_users=num_users, num_items=num_items,
            embedding_dim=64, num_layers=3, n_epochs=30,
        ),
        "SASRec": SASRecEngine(
            num_items=num_items, max_seq_len=50, hidden_dim=50, n_epochs=30,
        ),
    }
```

- [ ] **Step 4: Drop the two social-engine fit branches**

Replace:
```python
        if name == "Funk-SVD":
            engine.fit(train_df)
        elif name == "TrustSVD":
            engine.fit({
                "interaction_matrix": train_interaction,
                "trust_matrix": all_data["trust_matrix"],
            })
        elif name == "LightGCN":
            engine.fit({
                "sym_adj_mat": sym_adj_train,
                "interaction_matrix": train_interaction,
            })
        elif name == "Social-LightGCN":
            engine.fit({
                "interaction_matrix": train_interaction,
                "trust_matrix": all_data["trust_matrix"],
            })
        elif name == "SASRec":
            engine.fit(train_seq_windows)
```
with:
```python
        if name == "Funk-SVD":
            engine.fit(train_df)
        elif name == "LightGCN":
            engine.fit({
                "sym_adj_mat": sym_adj_train,
                "interaction_matrix": train_interaction,
            })
        elif name == "SASRec":
            engine.fit(train_seq_windows)
```

Note: `all_data["trust_matrix"]` is still computed inside `loader.build_all()` (in `unified_data_loader.py`, which is out of scope to modify) but is now unused by this script — this is a deliberate, harmless tradeoff to avoid touching the shared loader that production's `run_pipeline.py` also depends on.

- [ ] **Step 5: Run the trimmed Classic Arena end-to-end**

Run: `py -m pipeline.engines.benchmark_arena`

Expected: trains and evaluates exactly 3 engines — `Funk-SVD`, `LightGCN`, `SASRec` — completes without `NameError`/`ImportError`, prints a 3-row results table, and writes `models/benchmark_arena_results.csv` with exactly those 3 rows.

- [ ] **Step 6: Commit**

```bash
git add pipeline/engines/benchmark_arena.py
git commit -m "refactor(benchmark): trim Classic Arena to non-social engines

Funk-SVD, LightGCN, and SASRec only. TrustSVD and Social-LightGCN
move to the new FilmTrust Social Arena (pipeline/filmtrust_arena),
which evaluates them against a real trust network instead of
MovieLens's Jaccard-fabricated one."
```

---

### Task 4: Update README — Strategic Positioning

**Files:**
- Modify: `README.md` (§2.1 table, new §2.4 section, §7 directory tree, §8 Quick Start)

**Interfaces:**
- Consumes: the exact printed table from Task 2 Step 2 (real `run_filmtrust.py` output) and confirmation from Task 3 Step 5 that the Classic Arena's Funk-SVD/LightGCN/SASRec numbers are unchanged (same code path, same data, same split — removing other dict entries doesn't affect their training).

- [ ] **Step 1: Rewrite §2.1 to drop the two social-engine rows**

In `README.md`, replace:
```markdown
### 2.1. Multi-Engine Benchmarking Arena (Graph, Sequential & Social CF)
Our unified evaluation suite compares classical matrix factorization against modern graph neural networks and sequential transformers:

| Engine | Paradigm | Train Time (s) | Recall@10 | NDCG@10 | Latency (ms) | P95 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Funk-SVD** | Classical Latent Factor | 0.6s | 0.0050 | 0.0019 | 6.96ms | 10.05ms |
| **TrustSVD** | Social-Aware MF | 7.1s | 0.0500 | 0.0189 | 0.29ms | 0.37ms |
| **LightGCN** | Graph Neural Networks | 424.1s | 0.0700 | **0.0367** ⭐ | 0.66ms | 0.90ms |
| **Social-LightGCN** (Proposed) | Early-Fusion Graph | 452.5s | **0.0750** ⭐ | 0.0313 | **0.23ms** | **0.29ms** |
| **SASRec** | Sequential Transformer | 35.7s | 0.0150 | 0.0063 | 2.64ms | 3.51ms |

* **Social-LightGCN (State-of-the-Art / Proposed):** Achieves the highest Recall@10 accuracy (0.0750) and lowest online serving latency (0.23ms) by dynamically fusing collaborative and social signals at the embedding propagation level.
* **LightGCN (Strong Baseline):** Achieves the highest NDCG@10 (0.0367) by propagating embeddings through 3 layers of the bipartite user-item graph structure.
* **TrustSVD (Social Baseline):** Leverages Jaccard-based social trust network regularization, outperforming the baseline Funk-SVD by 10x on Recall@10.
```
with:
```markdown
### 2.1. Classic Multi-Engine Benchmarking Arena (Graph, Sequential & Classical CF)
Our Classic Arena compares **non-social** collaborative filtering paradigms — classical matrix factorization, graph neural networks, and sequential transformers — on MovieLens-100k:

| Engine | Paradigm | Train Time (s) | Recall@10 | NDCG@10 | Latency (ms) | P95 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Funk-SVD** | Classical Latent Factor | 0.6s | 0.0050 | 0.0019 | 6.96ms | 10.05ms |
| **LightGCN** | Graph Neural Networks | 424.1s | **0.0700** ⭐ | **0.0367** ⭐ | 0.66ms | 0.90ms |
| **SASRec** | Sequential Transformer | 35.7s | 0.0150 | 0.0063 | 2.64ms | 3.51ms |

* **LightGCN (Strong Baseline):** Achieves the highest Recall@10 (0.0700) and NDCG@10 (0.0367) by propagating embeddings through 3 layers of the bipartite user-item graph structure.
* **Funk-SVD / SASRec:** Classical and sequential baselines respectively, included to contextualize the graph-based approach's gains.

> **Where are TrustSVD and Social-LightGCN?** Social-Aware models are **not** benchmarked on MovieLens. See §2.4 for why, and where they're evaluated instead.
```

(If Task 3's actual rerun produced different Funk-SVD/LightGCN/SASRec numbers than the ones already in the table — it shouldn't, since their code path is unchanged — use the freshly observed numbers instead of the ones shown here.)

- [ ] **Step 2: Insert new §2.4 after the "Hardware Limitations & Reproducibility Notice" block**

In `README.md`, find this exact block (end of current section 2, immediately before the `---` separator that precedes `## 3. Engineering Contributions...`):
```markdown
### ⚠️ Hardware Limitations & Reproducibility Notice
It is important to note the hardware context of the current benchmark results:
* **Current Execution:** All reported training metrics for Social-LightGCN in this repository were run on a **local CPU** (AMD Ryzen 7 5800H) with capped batch sizes and limited epochs (e.g., 15-30 epochs) to prevent memory overflow on consumer hardware.
* **Impact on Accuracy:** Because the *Adaptive Attention Gate* requires substantial gradient steps to fully calibrate per-user social influence, the restricted CPU training inherently caps the model's potential. The slight drop in `Recall@10` is a direct symptom of early stopping and under-training.
* **Next Steps (GPU Scaling):** For full academic replication, the pipeline is fully CUDA-compatible. Running this architecture on a Cloud GPU (e.g., Google Colab T4 / AWS EC2) for 500+ epochs is expected to eliminate the K=10 precision drop and fully unleash the model's capacity.

---
```
and insert a new subsection immediately before that final `---`, so the block becomes:
```markdown
### ⚠️ Hardware Limitations & Reproducibility Notice
It is important to note the hardware context of the current benchmark results:
* **Current Execution:** All reported training metrics for Social-LightGCN in this repository were run on a **local CPU** (AMD Ryzen 7 5800H) with capped batch sizes and limited epochs (e.g., 15-30 epochs) to prevent memory overflow on consumer hardware.
* **Impact on Accuracy:** Because the *Adaptive Attention Gate* requires substantial gradient steps to fully calibrate per-user social influence, the restricted CPU training inherently caps the model's potential. The slight drop in `Recall@10` is a direct symptom of early stopping and under-training.
* **Next Steps (GPU Scaling):** For full academic replication, the pipeline is fully CUDA-compatible. Running this architecture on a Cloud GPU (e.g., Google Colab T4 / AWS EC2) for 500+ epochs is expected to eliminate the K=10 precision drop and fully unleash the model's capacity.

### 2.4. Social-Aware Arena (FilmTrust — Explicit Trust)
A mentor review flagged a strict academic-validity gap: MovieLens-100k has no real social
graph, so our prior comparison evaluated TrustSVD and Social-LightGCN against a "trust"
matrix fabricated via Jaccard similarity over co-interacted items
(`unified_data_loader.py::build_implicit_trust_matrix`). That is a collaborative-filtering
signal, not a trust network — it cannot isolate what social regularization actually
contributes, so it is not a valid benchmark for Social-Aware models.

To fix this, Social-Aware models are now benchmarked exclusively on **FilmTrust**
(Guo, Zhang & Yorke-Smith, IJCAI 2013), sourced directly from the
[LibRec GitHub repository](https://github.com/guoguibing/librec/tree/master/librec/demo/Datasets/FilmTrust):
35,497 explicit ratings from 1,508 users on 2,071 films, plus 1,853 **directed, user-asserted**
trust edges (symmetrized for GCN propagation). No Jaccard or any other derived-trust step is
used anywhere in this pipeline.

| Engine | Social Signal | Train Time (s) | Recall@10 | NDCG@10 | Latency (ms) | P95 Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGCN** (No-Social Baseline) | None | <TRAIN_TIME_1> | <RECALL_1> | <NDCG_1> | <LAT_1> | <P95_1> |
| **TrustSVD** | Explicit Trust (regularization) | <TRAIN_TIME_2> | <RECALL_2> | <NDCG_2> | <LAT_2> | <P95_2> |
| **Social-LightGCN** (Proposed) | Explicit Trust (graph propagation) | <TRAIN_TIME_3> | <RECALL_3> | <NDCG_3> | <LAT_3> | <P95_3> |

*Run `python -m pipeline.filmtrust_arena.run_filmtrust` to reproduce. See `pipeline/filmtrust_arena/` for the loader and orchestrator.*

---
```

Replace every `<...>` placeholder with the exact numbers printed by Task 2 Step 2's real run output before committing — do not leave placeholders in the committed file.

- [ ] **Step 3: Update the directory tree in §7**

In `README.md`, replace:
```markdown
│   ├── academic_sandbox/            # Yelp Benchmark Sandbox (Isolated -- Section 4.4)
│   │   ├── yelp_data_loader.py      # QRec Yelp downloader and stratified splitter
│   │   ├── model_wrappers.py        # Yelp-optimized model training wrappers
│   │   └── run_yelp_benchmark.py    # Yelp benchmark orchestrator
│   └── hybrid_recommender.py        # Online hybrid blending and diversity engine
```
with:
```markdown
│   ├── academic_sandbox/            # Yelp Benchmark Sandbox (Isolated -- Section 4.4)
│   │   ├── yelp_data_loader.py      # QRec Yelp downloader and stratified splitter
│   │   ├── model_wrappers.py        # Yelp-optimized model training wrappers
│   │   └── run_yelp_benchmark.py    # Yelp benchmark orchestrator
│   ├── filmtrust_arena/             # Social Arena -- Explicit Trust (Section 2.4)
│   │   ├── filmtrust_loader.py      # FilmTrust downloader, explicit-trust CSR builder
│   │   └── run_filmtrust.py         # LightGCN vs. TrustSVD vs. Social-LightGCN orchestrator
│   └── hybrid_recommender.py        # Online hybrid blending and diversity engine
```

- [ ] **Step 4: Add a Quick Start subsection in §8**

In `README.md`, replace:
```markdown
#### B. Yelp Benchmark (SEPT Protocol - Section 4.4)
Runs our `Social-LightGCN` alongside vanilla `LightGCN` on the large **Yelp** dataset (sparse interaction graph + dense trust network).
```bash
python -m pipeline.academic_sandbox.run_yelp_benchmark --epochs 30 --dim 64
```
*Options:* Use `--epochs` to set epochs, `--dim` for embedding size, or `--batch_size` to modify mini-batch sizing.

---

## 9. API Endpoints
```
with:
```markdown
#### B. Yelp Benchmark (SEPT Protocol - Section 4.4)
Runs our `Social-LightGCN` alongside vanilla `LightGCN` on the large **Yelp** dataset (sparse interaction graph + dense trust network).
```bash
python -m pipeline.academic_sandbox.run_yelp_benchmark --epochs 30 --dim 64
```
*Options:* Use `--epochs` to set epochs, `--dim` for embedding size, or `--batch_size` to modify mini-batch sizing.

### 8.6. Run the Social Arena (FilmTrust — Explicit Trust, Section 2.4)
Runs `LightGCN` (no-social baseline), `TrustSVD`, and `Social-LightGCN` against FilmTrust's real, explicit trust network (auto-downloaded on first run, no manual setup needed):
```bash
python -m pipeline.filmtrust_arena.run_filmtrust --epochs 30 --dim 64 -k 10
```
*Options:* Use `--epochs` to set epochs, `--dim` for embedding size, `--layers` for GCN depth, or `-k` for the Recall@K/NDCG@K cutoff.

---

## 9. API Endpoints
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: split Classic Arena (MovieLens) and Social Arena (FilmTrust)

Explains why Social-Aware models moved off MovieLens's Jaccard-
fabricated trust graph onto FilmTrust's real, explicit trust network,
with real measured benchmark numbers from the new Social Arena."
```
