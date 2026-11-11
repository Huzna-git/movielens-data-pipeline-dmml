# Model evaluation (K = 10)

Dataset: movielens-100k · split: per_user_time (test fraction 0.2) · a held-out rating >= 4.0 is relevant · items rated in train are excluded from each user's list.

| model                    |   precision_at_10 |   recall_at_10 |   ndcg_at_10 |   coverage |   diversity |     rmse |      mae |
|:-------------------------|------------------:|---------------:|-------------:|-----------:|------------:|---------:|---------:|
| popularity               |            0.0773 |         0.0659 |       0.0957 |     0.0422 |      0.6458 |   1.0734 |   0.8587 |
| implicit_als             |            0.1102 |         0.1341 |       0.1476 |     0.3532 |      0.7010 | n/a      | n/a      |
| als_matrix_factorization |            0.0760 |         0.0634 |       0.0903 |     0.2027 |      0.6878 |   0.9744 |   0.7695 |
| content_based_genres     |            0.0178 |         0.0167 |       0.0232 |     0.4810 |      0.1305 |   1.1246 |   0.8933 |
| hybrid                   |            0.1161 |         0.1411 |       0.1544 |     0.2438 |      0.5087 | n/a      | n/a      |

Users evaluated: 908 (test users with at least one relevant held-out item).

**Best ranking model by NDCG@10: `hybrid`.**

Precision/Recall/NDCG@K are averaged over users. Coverage = share of the catalogue recommended to at least one user. Diversity = 1 - mean pairwise cosine similarity of genre vectors within a user's list, averaged over users. RMSE/MAE are over all held-out ratings for the models that predict explicit ratings.
