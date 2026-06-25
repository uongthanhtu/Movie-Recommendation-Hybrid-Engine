# Grand Arena Results

## ciao -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0834 | 0.0542 | 34.0 | 0.18 |
| trustsvd | 0.0024 | 0.0015 | 1.7 | 0.17 |
| social_lightgcn | 0.0021 | 0.0020 | 39.7 | 0.17 |

## epinions -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0304 | 0.0233 | 301.3 | 0.31 |
| trustsvd | 0.0162 | 0.0118 | 12.2 | 0.28 |
| social_lightgcn | 0.0275 | 0.0204 | 579.4 | 0.31 |

## filmtrust -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.6393 | 0.5177 | 50.0 | 0.19 |
| trustsvd | 0.3538 | 0.3023 | 0.8 | 0.17 |
| social_lightgcn | 0.5443 | 0.4482 | 39.3 | 0.19 |

## ml-100k -- Mode B: Implicit Trust (ABLATION STUDY)

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| funksvd | 0.0350 | 0.0887 | 0.5 | 6.06 |
| lightgcn | 0.1722 | 0.3166 | 266.1 | 0.18 |
| trustsvd | 0.0713 | 0.1736 | 1.8 | 0.17 |
| social_lightgcn | 0.1698 | 0.3134 | 188.4 | 0.18 |

## yelp -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0376 | 0.0255 | 292.2 | 0.30 |
| trustsvd | 0.0063 | 0.0045 | 14.3 | 0.36 |
| social_lightgcn | 0.0006 | 0.0012 | 613.1 | 0.41 |

## Skipped Datasets

### douban

```
Could not obtain a usable ratings/trust file for 'douban' from any configured URL.
Manual fallback: place files named one of ['uir.index', 'ratings.txt'] (ratings) and ['social.index', 'trust.txt'] (trust) directly into data/douban.

Douban (Hao Ma et al., 'Recommender systems with social regularization', WSDM 2011) has no working automated download as of 2026-06-24: the original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its '.new' variant both return 404, and the ASU Social Computing Data Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own description directs manual requests to 113333244@qq.com. Once obtained, place 'uir.index' (format: UserId ItemId Rating) and 'social.index' (format: UserId1 UserId2) into data/douban/. NOTE: this column layout is from secondary documentation, not a primary file inspection -- verify it against the real file before trusting any benchmark numbers.
```
