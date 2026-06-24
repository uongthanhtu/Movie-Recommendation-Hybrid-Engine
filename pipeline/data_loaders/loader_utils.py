"""
Loader Utils -- Shared, dataset-agnostic helpers for BaseDatasetLoader implementations.

Extracted from pipeline/data_loaders/explicit_trust_loader.py (sub-project 2) with no
behavior change, so that pipeline/data_loaders/implicit_trust_loader.py (sub-project 3)
can reuse the same download/parse/filter/split/matrix-construction logic instead of
duplicating ~150 lines of it. Both ExplicitTrustLoader and ImplicitTrustLoader call
into this module; neither of them owns it.
"""
from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


class ManualDownloadRequiredError(RuntimeError):
    """
    Raised when a dataset's files cannot be obtained via automated download --
    either no URLs are configured, or every configured URL failed -- and must be
    placed manually. Subclasses RuntimeError for compatibility with any existing
    generic exception handling. Both ExplicitTrustLoader and ImplicitTrustLoader
    raise this same class so callers only need to catch one exception type.
    """


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------
def download_with_fallback(urls: List[str], data_dir: str, dataset_name: str, loader_label: str) -> None:
    """
    Try each URL in order until one succeeds. If the response is a zip, extract it
    into data_dir; otherwise save it as a raw file named from the URL. Does not raise
    on total failure -- callers must check files_exist() afterward and raise their own
    dataset-specific error (different loaders need different files to be present).
    """
    os.makedirs(data_dir, exist_ok=True)
    for url in urls:
        try:
            print(f"  [{loader_label}:{dataset_name}] Downloading from {url} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            print(f"  [{loader_label}:{dataset_name}] Failed to fetch {url}: {e}", flush=True)
            continue

        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(data_dir)
            print(f"  [{loader_label}:{dataset_name}] Extracted zip ({len(data)/1024/1024:.1f} MB) -> {data_dir}", flush=True)
        else:
            dest_name = _guess_filename_for_url(url)
            dest_path = os.path.join(data_dir, dest_name)
            with open(dest_path, "wb") as f:
                f.write(data)
            print(f"  [{loader_label}:{dataset_name}] Saved {len(data):,} bytes -> {dest_path}", flush=True)


def _guess_filename_for_url(url: str) -> str:
    """Pick a destination filename for a raw (non-zip) download from its URL."""
    basename = url.rstrip("/").split("/")[-1].split("?")[0]
    return basename if basename else "downloaded_file.txt"


def files_exist(data_dir: str, filenames: List[str]) -> bool:
    """True if any of the candidate filenames (case-insensitive) exist anywhere under data_dir."""
    lower_targets = {f.lower() for f in filenames}
    for root, _, files in os.walk(data_dir):
        lower_files = {f.lower() for f in files}
        if lower_targets & lower_files:
            return True
    return False


def resolve_path(data_dir: str, filenames: List[str]) -> str:
    """Find the first file under data_dir matching one of the candidate filenames (case-insensitive)."""
    lower_targets = {f.lower() for f in filenames}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() in lower_targets:
                return os.path.join(root, f)
    raise FileNotFoundError(f"No file found under {data_dir} matching {filenames}")


# ------------------------------------------------------------------
# Parsing (shared generic parser for ratings/trust rows)
# ------------------------------------------------------------------
def parse_rows(
    path: str,
    delimiter: str,
    explicit_rating_col_index: int = 2,
    col_names: Tuple[str, str, str] = ("user", "item", "rating"),
) -> pd.DataFrame:
    """
    Generic row parser. Branches on column count after delimiter-splitting:
      >=5 columns -> rating/weight at explicit_rating_col_index (Ciao's categoryId/
                     reviewId layout)
      >=3 columns -> rating/weight at column index 2 (generic "user item rating[...]")
      ==2 columns -> implicit, weight defaults to 1.0
    """
    rows: List[Tuple[str, str, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            parts = split_line(line, delimiter)

            if len(parts) >= 5:
                idx = explicit_rating_col_index
                if idx < len(parts):
                    rows.append((parts[0].strip(), parts[1].strip(), float(parts[idx].strip())))
            elif len(parts) >= 3:
                rows.append((parts[0].strip(), parts[1].strip(), float(parts[2].strip())))
            elif len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip(), 1.0))

    return pd.DataFrame(rows, columns=list(col_names))


def split_line(line: str, delimiter: str) -> List[str]:
    if delimiter == "comma":
        return line.split(",")
    if delimiter == "space":
        return line.split()
    # "auto": try comma first (Ciao's existing heuristic), else whitespace
    if "," in line:
        return line.split(",")
    return line.split()


# ------------------------------------------------------------------
# k-core filtering
# ------------------------------------------------------------------
def k_core_filter(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, int]:
    """Iteratively remove users/items with fewer than k interactions until convergence."""
    n_rounds = 0
    while True:
        n_before = len(df)

        user_counts = df["user"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user"].isin(valid_users)]

        item_counts = df["item"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item"].isin(valid_items)]

        n_rounds += 1
        if len(df) == n_before:
            break

    return df.reset_index(drop=True), n_rounds


# ------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user leave-N-out: hold out test_ratio of each user's items. Requires df to have a u_idx column."""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in df.groupby("u_idx"):
        indices = group.index.tolist()
        n_test = max(1, int(len(indices) * test_ratio))
        if len(indices) < 2:
            train_idx.extend(indices)
        else:
            rng.shuffle(indices)
            test_idx.extend(indices[:n_test])
            train_idx.extend(indices[n_test:])

    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


# ------------------------------------------------------------------
# Sparse Matrix Construction
# ------------------------------------------------------------------
def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, binarize: bool) -> sp.csr_matrix:
    """Requires df to have u_idx/i_idx columns (and a rating column if binarize=False)."""
    rows = df["u_idx"].values.astype(np.int64)
    cols = df["i_idx"].values.astype(np.int64)

    if binarize:
        vals = np.ones(len(rows), dtype=np.float32)
    else:
        vals = df["rating"].values.astype(np.float32)

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    if binarize:
        mat.data[:] = 1.0
    return mat


def build_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Requires df to have u_idx/i_idx columns."""
    result: Dict[int, Set[int]] = {}
    for u, i in zip(df["u_idx"].values, df["i_idx"].values):
        result.setdefault(int(u), set()).add(int(i))
    return result
