# Data validation report

Generated: 2026-11-11T04:33:16+00:00

## Tables

| Table | Rows in | Error rows (removed) | Warning rows (kept) | Rows out |
|---|---:|---:|---:|---:|
| users | 500 | 0 | 0 | 500 |
| items | 800 | 0 | 0 | 800 |
| interactions | 39,161 | 1,400 | 0 | 37,761 |

## Checks

| Table | Check | Column(s) | Severity | Failures | Status |
|---|---|---|---|---:|---|
| users | invalid_type | user_id,age,gender,occupation,zip_code | error | 0 | PASS |
| users | missing_value | non-nullable columns | error | 0 | PASS |
| users | duplicate_row | all | error | 0 | PASS |
| users | conflicting_duplicate_key | user_id | error | 0 | PASS |
| items | invalid_type | item_id,title,release_date,imdb_url,genres | error | 0 | PASS |
| items | missing_value | non-nullable columns | error | 0 | PASS |
| items | duplicate_row | all | error | 0 | PASS |
| items | conflicting_duplicate_key | item_id | error | 0 | PASS |
| interactions | invalid_type | user_id,item_id,rating,timestamp | error | 199 | FAIL |
| interactions | missing_value | non-nullable columns | error | 395 | FAIL |
| interactions | duplicate_row | all | error | 233 | FAIL |
| interactions | conflicting_duplicate_key | user_id,item_id | error | 103 | FAIL |
| users | duplicate_key | user_id | error | 0 | PASS |
| users | out_of_range | age in [5, 110] | error | 0 | PASS |
| users | invalid_category | gender in {M,F} | error | 0 | PASS |
| items | duplicate_key | item_id | error | 0 | PASS |
| items | unparseable_date | release_date (dd-Mon-yyyy) | error | 0 | PASS |
| items | missing_release_date | release_date | warning | 0 | PASS |
| items | duplicate_title | title | warning | 0 | PASS |
| interactions | out_of_range | rating in [1.0, 5.0]; timestamp in [2020-01-01, 2030-01-01] | error | 394 | FAIL |
| interactions | unknown_user | user_id -> users.user_id | error | 194 | FAIL |
| interactions | unknown_item | item_id -> items.item_id | error | 189 | FAIL |

## Sample of flagged rows (first 25)

| table        |   row | column   | check        | severity   | value   |
|:-------------|------:|:---------|:-------------|:-----------|:--------|
| interactions |    59 | rating   | invalid_type | error      | five    |
| interactions |    94 | rating   | invalid_type | error      | five    |
| interactions |   381 | rating   | invalid_type | error      | five    |
| interactions |  1737 | rating   | invalid_type | error      | five    |
| interactions |  1749 | rating   | invalid_type | error      | five    |
| interactions |  1795 | rating   | invalid_type | error      | five    |
| interactions |  1800 | rating   | invalid_type | error      | five    |
| interactions |  2107 | rating   | invalid_type | error      | five    |
| interactions |  2567 | rating   | invalid_type | error      | five    |
| interactions |  3067 | rating   | invalid_type | error      | five    |
| interactions |  3647 | rating   | invalid_type | error      | five    |
| interactions |  3911 | rating   | invalid_type | error      | five    |
| interactions |  3926 | rating   | invalid_type | error      | five    |
| interactions |  3962 | rating   | invalid_type | error      | five    |
| interactions |  3995 | rating   | invalid_type | error      | five    |
| interactions |  4237 | rating   | invalid_type | error      | five    |
| interactions |  4622 | rating   | invalid_type | error      | five    |
| interactions |  4641 | rating   | invalid_type | error      | five    |
| interactions |  4865 | rating   | invalid_type | error      | five    |
| interactions |  5136 | rating   | invalid_type | error      | five    |
| interactions |  5162 | rating   | invalid_type | error      | five    |
| interactions |  5265 | rating   | invalid_type | error      | five    |
| interactions |  5546 | rating   | invalid_type | error      | five    |
| interactions |  5632 | rating   | invalid_type | error      | five    |
| interactions |  5777 | rating   | invalid_type | error      | five    |
