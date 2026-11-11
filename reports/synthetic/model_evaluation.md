# Model evaluation (K = 10)

Dataset: synthetic · split: per_user_time (test fraction 0.2) · a held-out rating >= 4.0 is relevant · items rated in train are excluded from each user's list.

| model                    |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |   coverage |   diversity |     rmse |      mae |
|:-------------------------|------------------:|---------------:|-------------:|-----------:|------------:|---------:|---------:|
| popularity               |            0.0509 |         0.1373 |       0.1027 |     0.1200 |      0.9002 |   0.9375 |   0.7122 |
| implicit_als             |            0.0366 |         0.0940 |       0.0677 |     0.4075 |      0.8832 | n/a      | n/a      |
| als_matrix_factorization |            0.0283 |         0.0628 |       0.0468 |     0.5663 |      0.8219 |   0.9282 |   0.7388 |
| content_based_genres     |            0.0171 |         0.0329 |       0.0244 |     0.9525 |      0.3950 |   0.8351 |   0.6691 |

Users evaluated: 434 (test users with at least one relevant held-out item).

**Best ranking model by NDCG@10: `popularity`.**

Precision/Recall/NDCG@K are averaged over users. Coverage = share of the catalogue recommended to at least one user. Diversity = 1 - mean pairwise cosine similarity of genre vectors within a user's list, averaged over users. RMSE/MAE are over all held-out ratings for the models that predict explicit ratings.
