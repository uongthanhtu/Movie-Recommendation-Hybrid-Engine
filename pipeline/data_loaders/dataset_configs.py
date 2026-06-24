"""
Dataset Configs -- Declarative per-dataset parameters for ExplicitTrustLoader.

Adding a new dataset (Douban, Epinions, Flixster -- a future sub-project) means
adding one DatasetConfig entry to DATASET_REGISTRY here, not writing a new loader
class -- this is what makes the factory Open for extension / Closed for modification.

Each config's fields were derived by reading the three existing, real loaders
(pipeline/unified_arena/academic_data_loader.py, pipeline/academic_sandbox/yelp_data_loader.py,
pipeline/filmtrust_arena/filmtrust_loader.py) and capturing their actual behavior,
not guessed from documentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """
    Declarative configuration for ExplicitTrustLoader.

    Fields:
        name: lowercase registry key, also used in log messages.
        data_dir: local directory for downloaded/extracted files.
        ratings_urls, trust_urls: fallback mirrors, tried in order until one succeeds.
            May point to the same URL for both (a shared zip containing both files,
            like Ciao/Yelp) or to two independent raw files (like FilmTrust) --
            the loader deduplicates and downloads each unique URL once.
        ratings_filenames, trust_filenames: candidate filenames (case-insensitive) to
            resolve on disk after download/extraction.
        delimiter: "auto" (try comma, fall back to whitespace -- Ciao's existing
            heuristic), "comma", or "space".
        explicit_rating_col_index: column index used for the rating value ONLY when a
            row has 5+ columns (Ciao's categoryId/reviewId layout). Rows with 3-4
            columns always use column index 2; rows with exactly 2 columns are
            implicit (weight defaults to 1.0).
        k_core: minimum interactions per user/item for iterative core filtering.
            None skips filtering entirely (Yelp, FilmTrust).
        feedback_mode: "threshold_binarize" filters rows to rating >= rating_threshold
            THEN stores binary 1.0 per surviving interaction (Ciao). "explicit" applies
            no filter and stores real rating values (FilmTrust; also correct for Yelp,
            whose raw values are already binary, so passthrough is a no-op).
        rating_threshold: only consulted when feedback_mode == "threshold_binarize".
        test_ratio, seed: stratified per-user train/test split parameters.
    """
    name: str
    data_dir: str
    ratings_urls: List[str]
    trust_urls: List[str]
    ratings_filenames: List[str]
    trust_filenames: List[str]
    delimiter: str = "auto"
    explicit_rating_col_index: int = 2
    k_core: Optional[int] = None
    feedback_mode: str = "explicit"
    rating_threshold: float = 0.0
    test_ratio: float = 0.2
    seed: int = 42


CIAO_CONFIG = DatasetConfig(
    name="ciao",
    data_dir="data/ciao",
    ratings_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
        "https://raw.githubusercontent.com/daicoolb/RecommenderSystem-DataSet/master/CiaoDVD/CiaoDVD.zip",
    ],
    trust_urls=[
        "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
        "https://raw.githubusercontent.com/daicoolb/RecommenderSystem-DataSet/master/CiaoDVD/CiaoDVD.zip",
    ],
    ratings_filenames=["movie-ratings.txt", "ratings.txt"],
    trust_filenames=["trusts.txt"],
    delimiter="auto",
    explicit_rating_col_index=4,
    k_core=5,
    feedback_mode="threshold_binarize",
    rating_threshold=3.0,
    test_ratio=0.2,
    seed=42,
)

YELP_CONFIG = DatasetConfig(
    name="yelp",
    data_dir="data/yelp",
    ratings_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
    trust_urls=[
        "https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=1",
    ],
    ratings_filenames=["ratings.txt", "train.txt"],
    trust_filenames=["trusts.txt", "trust.txt", "trustnetwork.txt", "links.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
)

FILMTRUST_CONFIG = DatasetConfig(
    name="filmtrust",
    data_dir="data/filmtrust",
    ratings_urls=[
        "https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/ratings.txt",
    ],
    trust_urls=[
        "https://raw.githubusercontent.com/guoguibing/librec/master/librec/demo/Datasets/FilmTrust/trust.txt",
    ],
    ratings_filenames=["ratings.txt"],
    trust_filenames=["trust.txt"],
    delimiter="space",
    k_core=None,
    feedback_mode="explicit",
    rating_threshold=0.0,
    test_ratio=0.2,
    seed=42,
)

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "ciao": CIAO_CONFIG,
    "yelp": YELP_CONFIG,
    "filmtrust": FILMTRUST_CONFIG,
}


@dataclass
class ImplicitDatasetConfig:
    """
    Declarative configuration for ImplicitTrustLoader (Mode B / ablation study).

    Unlike DatasetConfig, there is no trust_urls/trust_filenames -- trust is
    synthesized via Jaccard similarity (pipeline/utils/sparse_jaccard.py), not
    downloaded. This is intentionally a separate dataclass from DatasetConfig rather
    than an extension of it, to avoid either type carrying fields that are always
    irrelevant/None for the other's use case.

    Fields:
        name, data_dir: same meaning as DatasetConfig.
        ratings_urls, ratings_filenames: same meaning as DatasetConfig.
        delimiter: "space" handles ML-100K's tab-delimited u.data correctly, since
            Python's str.split() with no argument splits on any whitespace run
            (including tabs) -- the same "space" mode DatasetConfig already uses.
        rating_col_index: column index of the rating value in a parsed row.
            ML-100K's "user item rating timestamp" layout puts it at index 2.
        k_core: minimum interactions per user/item. None for ML-100K -- GroupLens
            already guarantees every user has >=20 ratings.
        jaccard_threshold, jaccard_top_k, jaccard_chunk_size: passed directly to
            compute_sparse_jaccard_trust (pipeline/utils/sparse_jaccard.py).
        test_ratio, seed: same meaning as DatasetConfig.
    """
    name: str
    data_dir: str
    ratings_urls: List[str]
    ratings_filenames: List[str]
    delimiter: str = "space"
    rating_col_index: int = 2
    k_core: Optional[int] = None
    jaccard_threshold: float = 0.3
    jaccard_top_k: Optional[int] = 50
    jaccard_chunk_size: int = 2000
    test_ratio: float = 0.2
    seed: int = 42


ML_100K_CONFIG = ImplicitDatasetConfig(
    name="ml-100k",
    data_dir="data/ml-100k",
    ratings_urls=["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
    ratings_filenames=["u.data"],
)

IMPLICIT_DATASET_REGISTRY: Dict[str, ImplicitDatasetConfig] = {
    "ml-100k": ML_100K_CONFIG,
}
