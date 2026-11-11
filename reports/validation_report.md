# Data validation report

Generated: 2026-11-04T06:12:05+00:00

## Tables

| Table | Rows in | Error rows (removed) | Warning rows (kept) | Rows out |
|---|---:|---:|---:|---:|
| users | 943 | 0 | 0 | 943 |
| items | 1,682 | 0 | 37 | 1,682 |
| interactions | 100,000 | 0 | 0 | 100,000 |

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
| interactions | invalid_type | user_id,item_id,rating,timestamp | error | 0 | PASS |
| interactions | missing_value | non-nullable columns | error | 0 | PASS |
| interactions | duplicate_row | all | error | 0 | PASS |
| interactions | conflicting_duplicate_key | user_id,item_id | error | 0 | PASS |
| users | duplicate_key | user_id | error | 0 | PASS |
| users | out_of_range | age in [5, 110] | error | 0 | PASS |
| users | invalid_category | gender in {M,F} | error | 0 | PASS |
| items | duplicate_key | item_id | error | 0 | PASS |
| items | unparseable_date | release_date (dd-Mon-yyyy) | error | 0 | PASS |
| items | missing_release_date | release_date | warning | 1 | WARN |
| items | duplicate_title | title | warning | 36 | WARN |
| interactions | out_of_range | rating in [1.0, 5.0]; timestamp in [1990-01-01, 2030-01-01] | error | 0 | PASS |
| interactions | unknown_user | user_id -> users.user_id | error | 0 | PASS |
| interactions | unknown_item | item_id -> items.item_id | error | 0 | PASS |

## Sample of flagged rows (first 25)

| table   |   row | column       | check                | severity   | value                          |
|:--------|------:|:-------------|:---------------------|:-----------|:-------------------------------|
| items   |   266 | release_date | missing_release_date | warning    | <NA>                           |
| items   |   245 | title        | duplicate_title      | warning    | Chasing Amy (1997)             |
| items   |   265 | title        | duplicate_title      | warning    | Kull the Conqueror (1997)      |
| items   |   267 | title        | duplicate_title      | warning    | Chasing Amy (1997)             |
| items   |   296 | title        | duplicate_title      | warning    | Ulee's Gold (1997)             |
| items   |   302 | title        | duplicate_title      | warning    | Ulee's Gold (1997)             |
| items   |   303 | title        | duplicate_title      | warning    | Fly Away Home (1996)           |
| items   |   304 | title        | duplicate_title      | warning    | Ice Storm, The (1997)          |
| items   |   308 | title        | duplicate_title      | warning    | Deceiver (1997)                |
| items   |   328 | title        | duplicate_title      | warning    | Desperate Measures (1998)      |
| items   |   347 | title        | duplicate_title      | warning    | Desperate Measures (1998)      |
| items   |   499 | title        | duplicate_title      | warning    | Fly Away Home (1996)           |
| items   |   572 | title        | duplicate_title      | warning    | Body Snatchers (1993)          |
| items   |   669 | title        | duplicate_title      | warning    | Body Snatchers (1993)          |
| items   |   679 | title        | duplicate_title      | warning    | Kull the Conqueror (1997)      |
| items   |   710 | title        | duplicate_title      | warning    | Substance of Fire, The (1996)  |
| items   |   864 | title        | duplicate_title      | warning    | Ice Storm, The (1997)          |
| items   |   875 | title        | duplicate_title      | warning    | Money Talks (1997)             |
| items   |   877 | title        | duplicate_title      | warning    | That Darn Cat! (1997)          |
| items   |   880 | title        | duplicate_title      | warning    | Money Talks (1997)             |
| items   |  1002 | title        | duplicate_title      | warning    | That Darn Cat! (1997)          |
| items   |  1174 | title        | duplicate_title      | warning    | Hugo Pool (1997)               |
| items   |  1233 | title        | duplicate_title      | warning    | Chairman of the Board (1998)   |
| items   |  1255 | title        | duplicate_title      | warning    | Designated Mourner, The (1997) |
| items   |  1256 | title        | duplicate_title      | warning    | Designated Mourner, The (1997) |
