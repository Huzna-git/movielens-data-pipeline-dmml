# Data validation report

Generated: 2026-11-12T05:18:51+00:00

## Tables

| Table | Rows in | Error rows (removed) | Warning rows (kept) | Rows out |
|---|---:|---:|---:|---:|
| users | 958 | 15 | 0 | 943 |
| items | 1,697 | 15 | 41 | 1,682 |
| interactions | 100,360 | 360 | 0 | 100,000 |

## Checks

| Table | Check | Column(s) | Severity | Failures | Status |
|---|---|---|---|---:|---|
| users | invalid_type | user_id,age,gender,occupation,zip_code | error | 2 | FAIL |
| users | missing_value | non-nullable columns | error | 2 | FAIL |
| users | duplicate_row | all | error | 5 | FAIL |
| users | conflicting_duplicate_key | user_id | error | 0 | PASS |
| items | invalid_type | item_id,title,release_date,imdb_url,genres | error | 0 | PASS |
| items | missing_value | non-nullable columns | error | 5 | FAIL |
| items | duplicate_row | all | error | 5 | FAIL |
| items | conflicting_duplicate_key | item_id | error | 0 | PASS |
| interactions | invalid_type | user_id,item_id,rating,timestamp | error | 66 | FAIL |
| interactions | missing_value | non-nullable columns | error | 160 | FAIL |
| interactions | duplicate_row | all | error | 40 | FAIL |
| interactions | conflicting_duplicate_key | user_id,item_id | error | 202 | FAIL |
| users | duplicate_key | user_id | error | 5 | FAIL |
| users | out_of_range | age in [5, 110] | error | 3 | FAIL |
| users | invalid_category | gender in {M,F} | error | 5 | FAIL |
| items | duplicate_key | item_id | error | 5 | FAIL |
| items | unparseable_date | release_date (dd-Mon-yyyy) | error | 5 | FAIL |
| items | missing_release_date | release_date | warning | 1 | WARN |
| items | duplicate_title | title | warning | 45 | WARN |
| interactions | out_of_range | rating in [1.0, 5.0]; timestamp in [1990-01-01, 2030-01-01] | error | 80 | FAIL |
| interactions | unknown_user | user_id -> users.user_id | error | 40 | FAIL |
| interactions | unknown_item | item_id -> items.item_id | error | 40 | FAIL |

## Sample of flagged rows (first 25)

| table   |   row | column   | check            | severity   | value                        |
|:--------|------:|:---------|:-----------------|:-----------|:-----------------------------|
| users   |   950 | age      | invalid_type     | error      | abc                          |
| users   |   952 | age      | invalid_type     | error      | abc                          |
| users   |   943 | user_id  | duplicate_key    | error      | 190                          |
| users   |   944 | user_id  | duplicate_key    | error      | 657                          |
| users   |   945 | user_id  | duplicate_key    | error      | 107                          |
| users   |   946 | user_id  | duplicate_key    | error      | 476                          |
| users   |   947 | user_id  | duplicate_key    | error      | 118                          |
| users   |   950 | age      | missing_value    | error      | <NA>                         |
| users   |   952 | age      | missing_value    | error      | <NA>                         |
| users   |   948 | age      | out_of_range     | error      | -3                           |
| users   |   949 | age      | out_of_range     | error      | -3                           |
| users   |   951 | age      | out_of_range     | error      | 200                          |
| users   |   953 | gender   | invalid_category | error      | X                            |
| users   |   954 | gender   | invalid_category | error      | Q                            |
| users   |   955 | gender   | invalid_category | error      | X                            |
| users   |   956 | gender   | invalid_category | error      | Q                            |
| users   |   957 | gender   | invalid_category | error      | Q                            |
| users   |   943 | user_id  | duplicate_row    | error      | exact copy of an earlier row |
| users   |   944 | user_id  | duplicate_row    | error      | exact copy of an earlier row |
| users   |   945 | user_id  | duplicate_row    | error      | exact copy of an earlier row |
| users   |   946 | user_id  | duplicate_row    | error      | exact copy of an earlier row |
| users   |   947 | user_id  | duplicate_row    | error      | exact copy of an earlier row |
| items   |  1682 | item_id  | duplicate_key    | error      | 623                          |
| items   |  1683 | item_id  | duplicate_key    | error      | 309                          |
| items   |  1684 | item_id  | duplicate_key    | error      | 154                          |
