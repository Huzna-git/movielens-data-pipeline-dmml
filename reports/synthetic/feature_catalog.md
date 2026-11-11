# Feature catalog

Every feature is computed by `src/features/build.py` from the **train split only** (the reference time for recency features is the latest train timestamp), stored by `src/features/store.py` under `data/features/`, and read back by the same store at training and at prediction time.

## User features (key: `user_id`, 30 features)

| Feature | Type | Description | Source | Computed over |
|---|---|---|---|---|
| `age` | int | Age reported in the user profile | users | profile |
| `gender` | category | Gender code (M/F) from the user profile | users | profile |
| `occupation` | category | Occupation from the user profile | users | profile |
| `user_n_interactions` | int | Number of ratings the user gave (activity count) | interactions | train split |
| `user_n_liked` | int | Number of ratings >= like threshold (4.0) | interactions | train split |
| `user_like_rate` | float | Share of the user's ratings that are 'likes' | interactions | train split |
| `user_mean_rating` | float | User's average rating (rating bias) | interactions | train split |
| `user_rating_std` | float | Std-dev of the user's ratings (0 when a single rating) | interactions | train split |
| `user_days_since_last_interaction` | float | Days from the user's last rating to the reference time (recency) | interactions | train split |
| `user_active_span_days` | float | Days between the user's first and last rating | interactions | train split |
| `user_preferred_genre` | category | Genre with the most liked ratings (ties: higher mean rating) | interactions + items | train split |
| `user_pref_unknown` | float | Share of the user's liked ratings in genre 'unknown' (genre profile vector) | interactions + items | train split |
| `user_pref_action` | float | Share of the user's liked ratings in genre 'action' (genre profile vector) | interactions + items | train split |
| `user_pref_adventure` | float | Share of the user's liked ratings in genre 'adventure' (genre profile vector) | interactions + items | train split |
| `user_pref_animation` | float | Share of the user's liked ratings in genre 'animation' (genre profile vector) | interactions + items | train split |
| `user_pref_children_s` | float | Share of the user's liked ratings in genre 'children_s' (genre profile vector) | interactions + items | train split |
| `user_pref_comedy` | float | Share of the user's liked ratings in genre 'comedy' (genre profile vector) | interactions + items | train split |
| `user_pref_crime` | float | Share of the user's liked ratings in genre 'crime' (genre profile vector) | interactions + items | train split |
| `user_pref_documentary` | float | Share of the user's liked ratings in genre 'documentary' (genre profile vector) | interactions + items | train split |
| `user_pref_drama` | float | Share of the user's liked ratings in genre 'drama' (genre profile vector) | interactions + items | train split |
| `user_pref_fantasy` | float | Share of the user's liked ratings in genre 'fantasy' (genre profile vector) | interactions + items | train split |
| `user_pref_film_noir` | float | Share of the user's liked ratings in genre 'film_noir' (genre profile vector) | interactions + items | train split |
| `user_pref_horror` | float | Share of the user's liked ratings in genre 'horror' (genre profile vector) | interactions + items | train split |
| `user_pref_musical` | float | Share of the user's liked ratings in genre 'musical' (genre profile vector) | interactions + items | train split |
| `user_pref_mystery` | float | Share of the user's liked ratings in genre 'mystery' (genre profile vector) | interactions + items | train split |
| `user_pref_romance` | float | Share of the user's liked ratings in genre 'romance' (genre profile vector) | interactions + items | train split |
| `user_pref_sci_fi` | float | Share of the user's liked ratings in genre 'sci_fi' (genre profile vector) | interactions + items | train split |
| `user_pref_thriller` | float | Share of the user's liked ratings in genre 'thriller' (genre profile vector) | interactions + items | train split |
| `user_pref_war` | float | Share of the user's liked ratings in genre 'war' (genre profile vector) | interactions + items | train split |
| `user_pref_western` | float | Share of the user's liked ratings in genre 'western' (genre profile vector) | interactions + items | train split |

## Item features (key: `item_id`, 29 features)

| Feature | Type | Description | Source | Computed over |
|---|---|---|---|---|
| `item_n_ratings` | int | Number of ratings the item received (popularity count) | interactions | train split |
| `item_mean_rating` | float | Item's average rating (NaN when never rated in train) | interactions | train split |
| `item_rating_std` | float | Std-dev of the item's ratings | interactions | train split |
| `item_like_rate` | float | Share of the item's ratings that are 'likes' | interactions | train split |
| `item_popularity_score` | float | log1p(n_ratings) * mean_rating - popular AND liked | interactions | train split |
| `item_popularity_rank` | int | Rank of the popularity score (1 = most popular) | interactions | train split |
| `item_days_since_first_rating` | float | Days from the item's first rating to the reference time (catalog age) | interactions | train split |
| `item_release_year` | int | Release year; missing years are filled with the catalogue median | items | catalogue |
| `item_release_year_missing` | bool | True when the release year was absent in the source | items | catalogue |
| `item_n_genres` | int | How many genres the item carries | items | catalogue |
| `genre_unknown` | bool | Item is tagged with genre 'unknown' | items | catalogue |
| `genre_action` | bool | Item is tagged with genre 'action' | items | catalogue |
| `genre_adventure` | bool | Item is tagged with genre 'adventure' | items | catalogue |
| `genre_animation` | bool | Item is tagged with genre 'animation' | items | catalogue |
| `genre_children_s` | bool | Item is tagged with genre 'children_s' | items | catalogue |
| `genre_comedy` | bool | Item is tagged with genre 'comedy' | items | catalogue |
| `genre_crime` | bool | Item is tagged with genre 'crime' | items | catalogue |
| `genre_documentary` | bool | Item is tagged with genre 'documentary' | items | catalogue |
| `genre_drama` | bool | Item is tagged with genre 'drama' | items | catalogue |
| `genre_fantasy` | bool | Item is tagged with genre 'fantasy' | items | catalogue |
| `genre_film_noir` | bool | Item is tagged with genre 'film_noir' | items | catalogue |
| `genre_horror` | bool | Item is tagged with genre 'horror' | items | catalogue |
| `genre_musical` | bool | Item is tagged with genre 'musical' | items | catalogue |
| `genre_mystery` | bool | Item is tagged with genre 'mystery' | items | catalogue |
| `genre_romance` | bool | Item is tagged with genre 'romance' | items | catalogue |
| `genre_sci_fi` | bool | Item is tagged with genre 'sci_fi' | items | catalogue |
| `genre_thriller` | bool | Item is tagged with genre 'thriller' | items | catalogue |
| `genre_war` | bool | Item is tagged with genre 'war' | items | catalogue |
| `genre_western` | bool | Item is tagged with genre 'western' | items | catalogue |

## Interaction features (key: `(user_id, item_id)`, 4 features)

| Feature | Type | Description | Source | Computed over |
|---|---|---|---|---|
| `ui_genre_affinity` | float | Mean of the user's genre-profile shares over the item's genres (0 = never liked any of them) | user + item features | train split |
| `ui_days_since_prev_interaction` | float | Days since the same user's previous rating (0 for the first) | interactions | train split |
| `ui_user_bias_minus_item_bias` | float | user_mean_rating - item_mean_rating (who is 'harder', the user or the item) | user + item features | train split |
| `ui_item_popularity_pct` | float | Percentile of the item's popularity rank (1.0 = most popular) | item features | train split |
