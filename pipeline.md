# Technical Pipeline: Tourism Popularity Prediction

## 1. Objective

This project builds a monthly machine-learning dataset that combines:

- Google Trends search interest as a demand/popularity proxy
- Reddit post activity, engagement, and sentiment as social perception signals

The prediction target is next-month city popularity:

```python
target_trend_next_month = trend_score(t + 1)
```

## 2. Data Layers

```text
data/raw        -> original collected CSV files
data/interim    -> cleaned and aggregated intermediate tables
data/processed  -> final modeling dataset
outputs/reports -> metrics and predictions
outputs/models  -> saved models
```

Default period:

```text
2018-01-01 to 2025-12-31
```

Default city count:

```text
20 European destination cities
```

## 3. Scripts

### 00_audit_city_candidates.py

Runs a small Reddit pilot sample for each candidate city before full collection.

Output:

```text
data/raw/reddit_city_audit.csv
```

The audit helps decide whether each city has enough Reddit volume, enough month coverage, and acceptable topical quality.

### 01_collect_google_trends.py

Collects Google Trends data with the anchor-city method:

- Batches cities into Google Trends query groups
- Uses Paris as the anchor city
- Scales batches into one comparable index
- Aggregates weekly data to monthly data

Output:

```text
data/interim/google_trends_weekly_wide.csv
data/interim/google_trends_monthly.csv
```

### 02_collect_reddit_posts.py

Collects Reddit submissions for each city keyword.
Primary mode is local dump parsing (`.zst`) from selected subreddits, with optional Pushshift mode as fallback.

Output:

```text
data/raw/reddit_posts.csv
data/raw/reddit_collection_status.json
```

The script checkpoints processed files/cities through `data/raw/reddit_collection_status.json`.

### 03_prepare_reddit_sentiment.py

Cleans Reddit text, filters low-quality rows, scores sentiment, and aggregates by city-month.

Output:

```text
data/interim/reddit_posts_clean.csv
data/interim/reddit_monthly_features.csv
```

### 04_build_monthly_dataset.py

Merges Google Trends and Reddit monthly tables, then creates:

- trend lags
- Reddit lags
- 3-month and 6-month rolling means
- change features
- trend x sentiment interaction
- COVID-period indicator
- next-month target

Output:

```text
data/processed/model_dataset.csv
```

### 05_modeling.py

Runs baseline and machine-learning models:

- naive baseline: next month equals current trend
- linear regression
- random forest regression

Output:

```text
outputs/reports/model_metrics.csv
outputs/reports/model_predictions.csv
outputs/models/random_forest.joblib
```

## 4. Modeling Logic

Each row is a city-month observation. Features at month `t` predict Google Trends popularity at month `t+1`.

Important columns:

```text
city
month
trend_score
trend_lag_1
trend_lag_2
trend_rolling_3
reddit_post_count
reddit_total_score
reddit_total_comments
reddit_avg_sentiment
reddit_sentiment_lag_1
trend_sentiment_interaction
target_trend_next_month
```

## 5. Thesis Interpretation

Google Trends captures explicit search intent. Reddit captures user-generated discussion, engagement, and perception. If Reddit-based variables improve next-month trend prediction over the naive baseline, the study supports the idea that social media signals complement search behavior in destination popularity forecasting.
