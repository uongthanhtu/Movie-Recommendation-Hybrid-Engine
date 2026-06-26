# Grand Arena Results

## ciao -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0787 | 0.0519 | 30.3 | 0.18 |
| trustsvd | 0.0049 | 0.0030 | 0.5 | 0.16 |
| social_lightgcn | 0.0730 | 0.0462 | 19.9 | 0.17 |

## epinions -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0282 | 0.0222 | 279.9 | 0.32 |
| trustsvd | 0.0162 | 0.0119 | 12.3 | 0.32 |
| social_lightgcn | 0.0339 | 0.0257 | 770.6 | 0.32 |

## filmtrust -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.6392 | 0.5182 | 47.9 | 0.20 |
| trustsvd | 0.3538 | 0.3021 | 0.8 | 0.18 |
| social_lightgcn | 0.6346 | 0.5176 | 32.5 | 0.18 |

## ml-100k -- Mode B: Implicit Trust (ABLATION STUDY)

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| funksvd | 0.0352 | 0.0939 | 0.5 | 5.26 |
| lightgcn | 0.1748 | 0.3202 | 264.0 | 0.20 |
| trustsvd | 0.0712 | 0.1740 | 1.9 | 0.17 |
| social_lightgcn | 0.1544 | 0.2767 | 161.2 | 0.20 |

## yelp -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0376 | 0.0255 | 283.3 | 0.34 |
| trustsvd | 0.0061 | 0.0041 | 7.0 | 0.31 |
| social_lightgcn | 0.0362 | 0.0242 | 586.6 | 0.32 |

## Skipped Datasets

### douban

```
Could not obtain a usable ratings/trust file for 'douban' from any configured URL.
Manual fallback: place files named one of ['uir.index', 'ratings.txt'] (ratings) and ['social.index', 'trust.txt'] (trust) directly into data/douban.

Douban (Hao Ma et al., 'Recommender systems with social regularization', WSDM 2011) has no working automated download as of 2026-06-24: the original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its '.new' variant both return 404, and the ASU Social Computing Data Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own description directs manual requests to 113333244@qq.com. Once obtained, place 'uir.index' (format: UserId ItemId Rating) and 'social.index' (format: UserId1 UserId2) into data/douban/. NOTE: this column layout is from secondary documentation, not a primary file inspection -- verify it against the real file before trusting any benchmark numbers.
```
