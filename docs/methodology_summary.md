# Methodology and Implementation Summary

This document summarizes the full empirical pipeline implemented for the thesis project. It is intended as source material for writing the methodology, implementation, and findings sections.

## 1. Research Aim

The project investigates whether Reddit-based social media signals can help predict future destination popularity measured through Google Trends.

The main prediction task is:

```python
target = Google Trends score at month t+1
```

Each observation represents one `city-month`. Features from month `t` are used to predict Google Trends popularity in month `t+1`.

The central research logic is:

- Google Trends captures search-based user intent.
- Reddit captures user-generated discussion, engagement, and sentiment.
- Combining both may improve destination popularity forecasting.

## 2. Study Scope

### Time Period

The final study period is:

```text
January 2018 to December 2025
```

### Cities

The analysis uses 20 European destination cities:

```text
Paris, Rome, Barcelona, Amsterdam, Prague,
Vienna, Budapest, Lisbon, Berlin, Athens,
London, Madrid, Istanbul, Copenhagen, Stockholm,
Dublin, Brussels, Warsaw, Florence, Venice
```

## 3. Data Sources

### 3.1 Google Trends

Google Trends data was collected using the `pytrends` Python package in Google Colab because the local machine had network restrictions when connecting to `trends.google.com`.

Google Trends was collected for the 20 city names between 2018 and 2025.

Because Google Trends limits comparisons to a small number of keywords per request, an anchor-city normalization approach was used:

- `Paris` was used as the anchor city.
- Cities were split into batches.
- Each batch included `Paris`.
- The mean value of Paris in the first batch was used as the reference.
- Other batches were rescaled using the ratio between the reference Paris mean and the Paris mean in that batch.

The weekly Google Trends series was aggregated to monthly frequency using monthly mean values.

Final Google Trends columns:

```text
date
city
trend_score
```

Final path:

```text
data/raw/google_trends_monthly.csv
```

### 3.2 Reddit

The original plan was to collect Reddit posts through Pushshift API. However, Pushshift API access was restricted, and the local machine also produced network errors such as:

```text
WinError 10013
```

As a result, the project switched to local Reddit dump parsing using AcademicTorrents / Pushshift dump files.

Selected subreddit submission dump files were downloaded as `.zst` files through qBittorrent. The selected dump files included travel-oriented subreddits such as:

```text
Europetravel
GreeceTravel
ItalyTravel
LondonTravel
solotravel
travel
uktravel
```

These dump files were stored locally under:

```text
data/raw/reddit_dump_submissions/
```

The dump files themselves were intentionally excluded from GitHub because they are large data files.

## 4. Reddit Data Extraction

The script `scripts/02_collect_reddit_posts.py` was rewritten to support two modes:

```python
REDDIT_SOURCE_MODE = "dump"
REDDIT_SOURCE_MODE = "pushshift"
```

The final analysis used `dump` mode.

The dump parser:

- reads `.zst` compressed Reddit submission dumps,
- parses each line as JSON,
- filters by selected subreddit,
- filters by date range,
- searches post title and selftext for city names,
- assigns matching city labels,
- deduplicates by `post_id + city`,
- checkpoints processed files in `reddit_collection_status.json`,
- appends matched posts into `reddit_posts.csv`.

The extraction produced approximately:

```text
207,013 Reddit post-city rows
```

The resulting raw Reddit file:

```text
data/raw/reddit_posts.csv
```

The date coverage was:

```text
2018-01-01 to 2025-12-30
```

## 5. Reddit Preprocessing and Sentiment Analysis

The script `scripts/03_prepare_reddit_sentiment.py` performs Reddit preprocessing.

### Cleaning Steps

The script:

- removes duplicates,
- converts text to lowercase,
- removes URLs,
- removes punctuation and non-letter characters,
- removes very short texts,
- filters low-score posts,
- removes likely bot authors,
- combines title and selftext into one cleaned text field.

### Sentiment Method

The pipeline uses `vaderSentiment` when available. VADER was available in the final environment, so the sentiment scores were produced with VADER compound sentiment.

A small manually defined positive/negative lexicon exists only as a fallback if VADER is unavailable.

The VADER compound score is then mapped to labels:

```text
positive if score >= 0.05
negative if score <= -0.05
neutral otherwise
```

### Monthly Reddit Aggregation

Reddit posts were aggregated to city-month level.

Monthly Reddit features include:

```text
reddit_post_count
reddit_total_score
reddit_avg_score
reddit_total_comments
reddit_avg_comments
reddit_avg_sentiment
reddit_sentiment_volatility
reddit_positive_share
reddit_negative_share
```

Main output:

```text
data/interim/reddit_monthly_features.csv
```

## 6. Dataset Construction

The script `scripts/04_build_monthly_dataset.py` merges Google Trends and Reddit monthly features.

Merge key:

```text
city + month
```

The final target variable is:

```text
target_trend_next_month = trend_score shifted by -1 month within each city
```

The final modeling dataset contains approximately:

```text
1,840 city-month observations
```

Final dataset:

```text
data/processed/model_dataset.csv
```

## 7. Feature Engineering

### Google Trends Features

The following trend-based features were created:

```text
trend_score
trend_lag_1
trend_lag_2
trend_lag_3
trend_rolling_3
trend_rolling_6
trend_volatility_3
trend_volatility_6
trend_change
trend_pct_change
trend_momentum_2
```

### Reddit Volume and Engagement Features

The following Reddit activity features were used:

```text
reddit_post_count
reddit_total_score
reddit_avg_score
reddit_total_comments
reddit_avg_comments
reddit_post_count_lag_1
reddit_post_count_lag_2
reddit_post_count_lag_3
reddit_post_count_rolling_3
reddit_post_count_rolling_6
```

### Reddit Sentiment Features

The following sentiment features were used:

```text
reddit_avg_sentiment
reddit_avg_sentiment_lag_1
reddit_avg_sentiment_lag_2
reddit_avg_sentiment_lag_3
reddit_sentiment_rolling_3
reddit_sentiment_rolling_6
reddit_sentiment_change
reddit_sentiment_volatility
reddit_positive_share
reddit_negative_share
trend_sentiment_interaction
```

### Seasonality and Structural Break Features

Tourism is seasonal, so month-based cyclic features were added:

```text
month_sin
month_cos
```

A COVID period indicator was also added:

```text
covid_period = 1 for March 2020 to December 2021
```

## 8. Modeling

The script `scripts/05_modeling.py` implements the baseline and primary models.

### Baseline

The naive baseline assumes:

```text
trend(t+1) = trend(t)
```

This is an important benchmark because Google Trends series are often highly persistent.

### Machine Learning Models

Two models were trained:

```text
Linear Regression
Random Forest Regression
```

Preprocessing:

- `city` was one-hot encoded.
- numeric features were standardized for linear models.

Evaluation metrics:

```text
MAE
RMSE
R-squared
```

The main holdout split was time-based. The final portion of the time series was used as the test period.

## 9. Main Model Results

The primary model results were:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Naive current trend | 2.374 | 7.235 | 0.901 |
| Linear regression | 2.786 | 7.113 | 0.905 |
| Random forest | 2.572 | 7.131 | 0.904 |

Interpretation:

- The ML models achieved high R2 values around 0.90.
- However, the naive baseline had the lowest MAE.
- This indicates that Google Trends destination popularity is highly persistent from one month to the next.
- Reddit features do not consistently outperform the strong persistence baseline at the global level.

## 10. Additional Diagnostic Analyses

The script `scripts/06_research_diagnostics.py` was created to strengthen the thesis methodology.

It performs:

```text
ablation study
stable vs volatile city segmentation
city-level error analysis
lag sensitivity analysis
seasonality-enhanced modeling
outlier control through winsorization
rolling time-series cross-validation
Diebold-Mariano style error comparison
```

### 10.1 Ablation Study

The ablation study compares:

```text
naive baseline
trend-only features
trend + Reddit volume features
trend + Reddit volume + sentiment features
```

Main ablation results:

| Feature Group | Model | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Naive | naive | 2.343 | 6.980 | 0.908 |
| Trend only | Linear | 2.721 | 7.787 | 0.886 |
| Trend only | RF | 2.603 | 7.095 | 0.905 |
| Trend + Reddit volume | Linear | 2.834 | 7.794 | 0.886 |
| Trend + Reddit volume | RF | 2.592 | 7.042 | 0.907 |
| Trend + Reddit full | Linear | 2.894 | 7.901 | 0.882 |
| Trend + Reddit full | RF | 2.566 | 7.072 | 0.906 |

Interpretation:

- Reddit volume and sentiment did not produce a large global improvement over the naive baseline.
- Random forest benefited slightly from Reddit features compared with trend-only RF in some metrics.
- Linear models became worse when Reddit variables were added, suggesting noise or multicollinearity.

### 10.2 Stable vs Volatile City Segmentation

Cities were segmented by trend volatility.

Results show that:

- stable cities are very well predicted by the naive baseline,
- volatile cities are harder to predict,
- Reddit features provide limited incremental improvement at the segment level.

For stable cities:

```text
Naive MAE = 0.325
Full RF MAE = 0.466
```

For volatile cities:

```text
Naive MAE = 4.360
Full RF MAE = 4.667
```

Interpretation:

- In stable destinations, persistence dominates.
- In volatile destinations, prediction is harder, but Reddit features still did not consistently beat the baseline.

### 10.3 Lag Sensitivity Analysis

Sentiment lag configurations were compared:

```text
lag1
lag1 + lag2
lag1 + lag2 + lag3
```

Results were very similar across configurations:

| Lag configuration | MAE | RMSE | R2 |
|---|---:|---:|---:|
| lag1 | 2.894 | 7.901 | 0.882 |
| lag1 + lag2 | 2.894 | 7.900 | 0.882 |
| lag1 + lag2 + lag3 | 2.894 | 7.901 | 0.882 |

Interpretation:

- Adding longer sentiment lags did not materially improve predictions.
- The monthly aggregated sentiment signal may be weak or noisy relative to trend persistence.

### 10.4 Rolling Time-Series Cross-Validation

Rolling-origin validation was used to ensure results were not due to one arbitrary holdout split.

Average cross-validation performance:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Naive current trend | 1.812 | 5.101 | 0.919 |
| Linear regression | 2.420 | 5.439 | 0.906 |
| Random forest | 2.071 | 5.263 | 0.913 |

Interpretation:

- Across rolling time splits, the naive baseline remains strongest on average.
- Random forest performs closer to the baseline than linear regression.
- The result supports the conclusion that destination search interest is highly autoregressive.

### 10.5 Diebold-Mariano Style Error Comparison

A Diebold-Mariano style test was used to compare squared forecast errors between baseline and full Reddit models.

Key results:

```text
Global full linear vs naive: p ≈ 0.069
Global full RF vs naive: p ≈ 0.851
Stable full linear vs naive: p ≈ 0.0017
Volatile full RF vs naive: p ≈ 0.875
```

Interpretation:

- There is no strong evidence that the full Reddit model significantly improves over the naive baseline.
- In stable cities, the full linear model is significantly worse than the naive baseline.

## 11. Technical Issues and Adjustments

Several practical issues were encountered and solved:

### Pushshift API Restriction

Live Pushshift API access was restricted and unreliable. The project switched to local `.zst` dump parsing.

### Network Restrictions

The local environment had network restrictions causing `WinError 10013` for Google Trends, Pushshift, GitHub, and package downloads. Google Trends was therefore collected in Colab.

### Large Files and GitHub

Initial GitHub push failed because large dump files were included in the first commit. The solution was:

- add `data/raw/reddit_dump_submissions/` to `.gitignore`,
- remove dump files from Git tracking,
- keep them locally,
- push only code, small CSV outputs, and reports.

### Dependency Issue

The `zstandard` package was required to parse `.zst` files. Because `pip install` was blocked by network restrictions, the wheel file was downloaded manually and installed locally.

## 12. Final Interpretation

The implemented pipeline shows that Reddit data can be integrated into a destination popularity forecasting framework. The project successfully combines:

```text
Google Trends search interest
Reddit discussion volume
Reddit engagement
Reddit sentiment
time-series features
machine-learning models
diagnostic tests
```

The empirical results suggest that Reddit-based signals are usable but do not consistently outperform a strong naive trend-persistence baseline. This is not a failure of the thesis. Instead, it is a meaningful empirical finding:

```text
Destination popularity measured through Google Trends is highly persistent at monthly frequency.
Reddit-derived social media signals provide limited and heterogeneous incremental predictive value.
Their usefulness may depend on city volatility, event-driven demand, data quality, and the level of aggregation.
```

This supports a nuanced conclusion:

```text
Reddit sentiment and discussion features can complement Google Trends in a tourism popularity forecasting pipeline, but their predictive contribution is conditional rather than universally superior.
```

## 13. Files and Outputs

Important scripts:

```text
scripts/01_collect_google_trends.py
scripts/02_collect_reddit_posts.py
scripts/03_prepare_reddit_sentiment.py
scripts/04_build_monthly_dataset.py
scripts/05_modeling.py
scripts/06_research_diagnostics.py
```

Important outputs:

```text
data/raw/google_trends_monthly.csv
data/raw/reddit_posts.csv
data/interim/reddit_monthly_features.csv
data/processed/model_dataset.csv
outputs/reports/model_metrics.csv
outputs/reports/ablation_metrics.csv
outputs/reports/segment_metrics.csv
outputs/reports/lag_sensitivity_metrics.csv
outputs/reports/time_cv_metrics.csv
outputs/reports/dm_test_metrics.csv
outputs/reports/city_error_summary.csv
```

## 14. Suggested Thesis Framing

A suitable methodology framing:

```text
This study constructs a city-month panel dataset combining Google Trends and Reddit-derived social media indicators. Google Trends is used as a proxy for destination search popularity, while Reddit provides alternative user-generated signals through discussion volume, engagement, and sentiment. The target variable is next-month Google Trends popularity. The study evaluates whether Reddit-derived variables improve forecasting performance over a naive persistence baseline and trend-only machine-learning models.
```

A suitable findings framing:

```text
The results show that destination popularity is strongly persistent at monthly frequency. Although Reddit-derived features can be incorporated into machine-learning models, they do not consistently outperform a naive current-trend baseline. Additional analyses, including ablation tests, city volatility segmentation, lag sensitivity, and rolling time-series cross-validation, indicate that the incremental predictive value of Reddit data is limited and heterogeneous across destinations.
```
