# Grand Arena Results

## ciao -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0787 | 0.0515 | 35.3 | 0.19 |
| trustsvd | 0.0036 | 0.0018 | 0.6 | 0.16 |
| social_lightgcn | 0.0014 | 0.0012 | 38.2 | 0.16 |

## epinions -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0295 | 0.0228 | 300.8 | 0.37 |
| trustsvd | 0.0162 | 0.0118 | 13.5 | 0.31 |
| social_lightgcn | 0.0269 | 0.0200 | 592.9 | 0.40 |

## filmtrust -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.6386 | 0.5175 | 50.4 | 0.18 |
| trustsvd | 0.3538 | 0.3023 | 0.8 | 0.16 |
| social_lightgcn | 0.5546 | 0.4591 | 40.8 | 0.18 |

## ml-100k -- Mode B: Implicit Trust (ABLATION STUDY)

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| funksvd | 0.0371 | 0.0905 | 0.5 | 5.20 |
| lightgcn | 0.1718 | 0.3155 | 276.0 | 0.20 |
| trustsvd | 0.0713 | 0.1728 | 2.0 | 0.19 |
| social_lightgcn | 0.1624 | 0.3019 | 195.5 | 0.20 |

## yelp -- Mode A: Explicit Trust

| Model | Recall@10 | NDCG@10 | Train Time (s) | Latency (ms) |
|---|---|---|---|---|
| lightgcn | 0.0378 | 0.0256 | 291.7 | 0.33 |
| trustsvd | 0.0065 | 0.0043 | 7.2 | 0.34 |
| social_lightgcn | 0.0004 | 0.0012 | 421.1 | 0.42 |

## Skipped Datasets

### douban

```
Could not obtain a usable ratings/trust file for 'douban' from any configured URL.
Manual fallback: place files named one of ['uir.index', 'ratings.txt'] (ratings) and ['social.index', 'trust.txt'] (trust) directly into data/douban.

Douban (Hao Ma et al., 'Recommender systems with social regularization', WSDM 2011) has no working automated download as of 2026-06-24: the original CUHK source (cse.cuhk.edu.hk/irwin.king/pub/data/douban) and its '.new' variant both return 404, and the ASU Social Computing Data Repository mirror (socialcomputing.asu.edu) is offline. The dataset's own description directs manual requests to 113333244@qq.com. Once obtained, place 'uir.index' (format: UserId ItemId Rating) and 'social.index' (format: UserId1 UserId2) into data/douban/. NOTE: this column layout is from secondary documentation, not a primary file inspection -- verify it against the real file before trusting any benchmark numbers.
```
