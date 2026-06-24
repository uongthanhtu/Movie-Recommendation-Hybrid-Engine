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
