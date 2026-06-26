# Grand Arena Results

## ciao -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0812 | 0.0528 | 30.6 | 0.17 |
| trustsvd | 0.0073 | 0.0038 | 0.5 | 0.15 |
| social_lightgcn | 0.0636 | 0.0416 | 18.8 | 0.16 |

## epinions -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0281 | 0.0221 | 281.8 | 0.35 |
| trustsvd | 0.0162 | 0.0117 | 12.5 | 0.30 |
| social_lightgcn | 0.0278 | 0.0220 | 530.3 | 0.32 |

## filmtrust -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.6393 | 0.5180 | 47.7 | 0.19 |
| trustsvd | 0.3538 | 0.3019 | 0.7 | 0.17 |
| social_lightgcn | 0.6390 | 0.5196 | 30.8 | 0.17 |

## ml-100k -- Mode B: Implicit Trust (ABLATION STUDY)

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| funksvd | 0.0374 | 0.0957 | 0.6 | 5.84 |
| lightgcn | 0.1679 | 0.3104 | 263.8 | 0.18 |
| trustsvd | 0.0713 | 0.1730 | 1.9 | 0.17 |
| social_lightgcn | 0.1560 | 0.2727 | 157.9 | 0.18 |

## yelp -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0376 | 0.0255 | 278.5 | 0.42 |
| trustsvd | 0.0063 | 0.0046 | 7.1 | 0.32 |
| social_lightgcn | 0.0352 | 0.0243 | 354.3 | 0.35 |

## Skipped Datasets

### douban

```
Could not obtain a usable ratings/trust file for 'douban' from any configured URL.
Manual fallback: place files named one of ['uir.index', 'ratings.txt'] (ratings) and ['social.index', 'trust.txt'] (trust) directly into data/douban.

Douban (Hao Ma et al., 'Recommender systems with social regularization', WSDM 2011) has no working automated download as of 2026-06-24: the original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its '.new' variant both return 404, and the ASU Social Computing Data Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own description directs manual requests to 113333244@qq.com. Once obtained, place 'uir.index' (format: UserId ItemId Rating) and 'social.index' (format: UserId1 UserId2) into data/douban/. NOTE: this column layout is from secondary documentation, not a primary file inspection -- verify it against the real file before trusting any benchmark numbers.
```
