# Epinions / Douban / Flixster Expansion — Design Spec

Date: 2026-06-24

## Context: this is sub-project 4 of 5

Part of the "Grand Unified Benchmark Arena" initiative:

1. ~~`pipeline/utils/sparse_jaccard.py`~~ — DONE, merged.
2. ~~`dataset_factory.py` + consolidation of Ciao/Yelp/FilmTrust~~ — DONE, merged.
3. ~~`ImplicitTrustLoader` (Mode B)~~ — DONE, merged.
4. **New explicit-trust datasets (Epinions now; Douban manual-only; Flixster deferred)** (THIS SPEC)
5. `grand_arena_runner.py` — config-driven orchestrator tying 1-4 together

## Research findings (verified, not guessed)

The original request named Douban, Epinions, and Flixster as "massive" datasets that
"may not have reliable direct-download URLs." Per this project's standing rule (no
guessed URLs/formats — established with FilmTrust and reaffirmed for every dataset
since), each was checked via WebSearch/WebFetch before any config was written:

- **Epinions**: a real, working, modern mirror exists at `static.preferred.ai`,
  maintained by the `cornac` recommender-systems library (PreferredAI). Both
  `ratings_data.zip` and `trust_data.zip` were fetched directly and confirmed to be
  real zip archives (4MB and 1.6MB respectively), containing `ratings_data.txt`
  (space-delimited `user item rating`, ratings 1-5) and `trust_data.txt`
  (space-delimited `source target trust_value`). This is a moderate-sized dataset —
  not the multi-million-row scale the original request anticipated; that scale
  belongs to a different, less-accessible Epinions product-rating dump, not this
  trust-network variant used by `cornac` and most trust-aware recsys papers
  (TrustSVD, SoRec, SocialMF-style benchmarks).
- **Douban**: the specific dataset used in trust-aware literature (Hao Ma et al.,
  "Recommender systems with social regularization," WSDM 2011 — `uir.index`/
  `social.index` format) has **no working automated download**: the original CUHK
  page (`cse.cuhk.edu.hk/irwin.king/pub/data/douban`, and the `.new` variant) both
  return 404; the ASU Social Computing Data Repository mirror
  (`socialcomputing.asu.edu`) is entirely unreachable (connection refused — the whole
  domain appears decommissioned); the dataset's own description directs manual
  requests to `113333244@qq.com`. A *different* "Douban" dataset exists on Kaggle
  (`DoubanMovieShortComments`) but is movie-review text, not ratings+social data —
  the wrong dataset for this use case, not a substitute.
- **Flixster**: same dead-mirror situation (original SFU page 404s, ASU mirror dead)
  plus a Figshare mirror that blocks automated fetching (403). Unlike Douban, no
  secondary source documenting the exact column format was found either — there is
  currently no way to write a config for Flixster without guessing both the URL and
  the format. **Flixster is deferred entirely**, same precedent as deferring ML-1M,
  ML-10M, and Jester in sub-project 3.
- **Housekeeping finding**: while researching mirrors, the `daicoolb/RecommenderSystem-DataSet`
  fallback URL already present in `CIAO_CONFIG`/`YELP_CONFIG` (sub-project 2) was
  checked directly and returns 404 — it was an unverified, guessed mirror that slipped
  through that sub-project's review. Fixed as part of this sub-project (see below),
  since this work is already touching `dataset_configs.py`.

## Decisions made (binding scope boundaries for this sub-project)

- **Epinions**: fully implemented and verified end-to-end against the real `cornac`
  mirror.
- **Douban**: registered in `DATASET_REGISTRY`, but with `ratings_urls=[]`/
  `trust_urls=[]` — automated download is impossible today, so `DatasetFactory.create("douban").load()`
  raises `ManualDownloadRequiredError` with instructions pointing to the dead sources
  and the email contact found during research. Its `ratings_filenames`/`trust_filenames`
  (`uir.index`/`social.index`) reflect the **best available secondary-source
  documentation, not a primary file inspection** — no real Douban file was ever
  obtained or parsed in this sub-project. This is explicitly flagged as unverified;
  if/when a user manually obtains the real files, the column layout should be
  re-confirmed against them before trusting any benchmark numbers produced from it.
- **Flixster**: no config added in this sub-project. Revisit once a working
  URL/format is found (a future sub-project, not a deadline on this one).
- **`CIAO_CONFIG`/`YELP_CONFIG`'s dead `daicoolb` fallback URL is removed** as part of
  this sub-project's work in `dataset_configs.py` — their primary URLs
  (`guoguibing.github.io`, the Dropbox share) are unaffected and continue to work.
- **No `loader_utils.py` parsing refinement.** The existing `explicit_rating_col_index`
  mechanism (built in sub-project 2) already generalizes to any column count/position.
  None of Epinions's or Douban's confirmed/best-available formats need a 5+-column
  path — adding speculative new column-handling logic for a case none of these
  datasets actually exhibit would be premature.

## Architecture

### `ManualDownloadRequiredError` (new, in `pipeline/data_loaders/loader_utils.py`)

```python
class ManualDownloadRequiredError(RuntimeError):
    """
    Raised when a dataset's files cannot be obtained via automated download --
    either no URLs are configured, or every configured URL failed -- and must be
    placed manually. Subclasses RuntimeError for compatibility with any existing
    generic exception handling.
    """
```

Both `ExplicitTrustLoader._ensure_downloaded` and `ImplicitTrustLoader._ensure_downloaded`
are updated to raise this instead of a generic `RuntimeError` (the only change to
their control flow is the exception type and message construction — no new branching
is needed: `download_with_fallback`'s loop over zero URLs is already a no-op for a
config like Douban's, and the existing post-download `files_exist` check already
detects the still-missing files; this sub-project only changes what gets raised
there). This keeps the exception type consistent across both loaders project-wide,
not just for the three datasets named in this request.

### `DatasetConfig` (modified, in `pipeline/data_loaders/dataset_configs.py`)

Two new fields, both defaulting to a no-op so `CIAO_CONFIG`/`YELP_CONFIG`/
`FILMTRUST_CONFIG` need no changes and have zero behavior change:

```python
manual_download_instructions: str = ""
filter_negative_trust: bool = False
```

- `manual_download_instructions`: folded into `ManualDownloadRequiredError`'s message
  when set; falls back to the existing generic "place files named X into Y" message
  when empty (preserves today's message for Ciao/Yelp/FilmTrust).
- `filter_negative_trust`: if `True`, `ExplicitTrustLoader.load()` drops trust rows
  with `weight <= 0` *before* calling `_build_social_matrix`. This addresses a real,
  pre-existing gap: `_build_social_matrix` already discards the actual trust weight
  value when building `social_csr` (it always stores binary `1.0` per surviving edge,
  regardless of what the parsed weight was) — so without this filter, a distrust
  edge encoded as a negative weight would currently be silently treated as a trust
  edge. Set `True` for `EPINIONS_CONFIG` (distrust is a documented real possibility
  in this dataset family); left `False` (default) for everything else, since Ciao/
  Yelp/FilmTrust/Douban's trust/friendship concepts have no distrust notion.
  **Whether the specific cornac Epinions mirror's `trust_data.txt` actually contains
  negative values has not been confirmed** (the file's exact contents weren't
  inspectable via WebFetch due to size) — this filter is implemented defensively;
  its real necessity is confirmed empirically during this sub-project's verification
  step, by checking the real downloaded file for negative weights.

### New configs (in `pipeline/data_loaders/dataset_configs.py`)

```python
EPINIONS_CONFIG = DatasetConfig(
    name="epinions",
    data_dir="data/epinions",
    ratings_urls=["https://static.preferred.ai/cornac/datasets/epinions/ratings_data.zip"],
    trust_urls=["https://static.preferred.ai/cornac/datasets/epinions/trust_data.zip"],
    ratings_filenames=["ratings_data.txt"],
    trust_filenames=["trust_data.txt"],
    delimiter="space",
    k_core=5,
    feedback_mode="explicit",
    rating_threshold=0.0,
    filter_negative_trust=True,
    test_ratio=0.2,
    seed=42,
)

DOUBAN_CONFIG = DatasetConfig(
    name="douban",
    data_dir="data/douban",
    ratings_urls=[],
    trust_urls=[],
    ratings_filenames=["uir.index", "ratings.txt"],
    trust_filenames=["social.index", "trust.txt"],
    delimiter="space",
    k_core=5,
    feedback_mode="explicit",
    rating_threshold=0.0,
    filter_negative_trust=False,
    test_ratio=0.2,
    seed=42,
    manual_download_instructions=(
        "Douban (Hao Ma et al., 'Recommender systems with social regularization', "
        "WSDM 2011) has no working automated download as of 2026-06-24: the "
        "original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its "
        "'.new' variant both return 404, and the ASU Social Computing Data "
        "Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own "
        "description directs manual requests to 113333244@qq.com. Once obtained, "
        "place 'uir.index' (format: UserId ItemId Rating) and 'social.index' "
        "(format: UserId1 UserId2) into data/douban/. NOTE: this column layout is "
        "from secondary documentation, not a primary file inspection -- verify it "
        "against the real file before trusting any benchmark numbers."
    ),
)

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "ciao": CIAO_CONFIG,
    "yelp": YELP_CONFIG,
    "filmtrust": FILMTRUST_CONFIG,
    "epinions": EPINIONS_CONFIG,
    "douban": DOUBAN_CONFIG,
}
```

`k_core=5` for both new configs is a conservative default mirroring Ciao's pattern
(massive raw social datasets tend to have many low-degree fringe users/items); since
neither dataset's exact raw interaction count was confirmed before download, this
value is treated as provisional and reviewed against the real, empirically-observed
statistics during this sub-project's verification step, not assumed correct in
advance.

### `ExplicitTrustLoader.load()` (modified)

One new step, inserted immediately after parsing both files — specifically **before**
the union-of-users ID-mapping step, not merely before `_build_social_matrix`. This
ordering matters: the union-of-users step reads `df_trust["src"]`/`["dst"]` to decide
the user universe, so a distrust-only user (who appears only in a negative-weight
trust row, never in ratings) must be excluded from that computation too — otherwise
they'd still inflate `num_users` with an all-zero row even after their only edge gets
filtered out of `social_csr` later.

```python
if cfg.filter_negative_trust:
    n_before = len(df_trust)
    df_trust = df_trust[df_trust["weight"] > 0].copy()
    print(f"    Filtered distrust: {n_before:,} -> {len(df_trust):,} trust edges", flush=True)
```

This runs right after `df_trust = lu.parse_rows(...)`, before the `all_users`/
`sorted_users` union-of-users block. It is a no-op for every existing config
(`filter_negative_trust` defaults to `False`), so Ciao/Yelp/FilmTrust's behavior is
unaffected.

## Out of scope

- Flixster — no working URL or confirmed format found; deferred to a future
  sub-project once both can be verified.
- `loader_utils.py` parsing logic changes beyond the `ManualDownloadRequiredError`
  exception type swap — the existing generic parser already handles every format
  found during research.
- `pipeline/utils/sparse_jaccard.py`, `ImplicitTrustLoader`'s own dataset registry,
  `grand_arena_runner.py` — unrelated to this sub-project.
- `pipeline/unified_arena/`, `pipeline/academic_sandbox/`, `pipeline/filmtrust_arena/`
  and their CLI runners — still deferred, unrelated to this sub-project.

## Verification plan

No pytest framework — direct script execution with documented exact expected output,
consistent with every other module in this codebase.

1. **Epinions, real data, end-to-end**: `DatasetFactory.create("epinions").load()`
   against the real, freshly-downloaded `cornac` mirror. Confirm: download succeeds,
   files resolve, raw row/user/item counts are sane (printed and inspected, not
   pre-asserted to an exact value never empirically observed), `feedback_mode="explicit"`
   preserves real 1-5 ratings in `train_csr`, `social_csr` is symmetric and
   zero-diagonal. **Explicitly check the real `trust_data.txt` for any negative
   `weight` values** — this is the empirical confirmation of whether
   `filter_negative_trust` is load-bearing for this specific mirror or a no-op safety
   net; report whichever turns out to be true rather than assuming.
2. **Douban, manual-download-required path**: `DatasetFactory.create("douban").load()`
   (with no `data/douban/` files present) raises `ManualDownloadRequiredError`
   containing the full `manual_download_instructions` text (the 113333244@qq.com
   contact and the dead-source citations must appear in the raised message).
3. **Regression check**: re-run the existing Ciao/Yelp/FilmTrust real-data
   equivalence script (from sub-project 2/3) unchanged, confirm
   `ALL EQUIVALENCE CHECKS PASSED` still holds after removing the dead `daicoolb`
   fallback URL and adding the two new defaulted `DatasetConfig` fields.
4. **`ImplicitTrustLoader` regression**: confirm `DatasetFactory.create("ml-100k").load()`
   still works (the `ManualDownloadRequiredError` exception-type change touches its
   `_ensure_downloaded` too, even though ML-100K's real working URL means this path
   isn't actually exercised by it).
