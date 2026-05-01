# Tourism Popularity Prediction Pipeline

This project measures whether Reddit discussion signals can help predict future Google Trends popularity for European city destinations.

## Pipeline Order

```powershell
python scripts/00_audit_city_candidates.py
python scripts/01_collect_google_trends.py
python scripts/02_collect_reddit_posts.py
python scripts/03_prepare_reddit_sentiment.py
python scripts/04_build_monthly_dataset.py
python scripts/05_modeling.py
python scripts/06_research_diagnostics.py
```

Collection scripts require internet access for Google Trends. Reddit collection can run from local dump files.

The default study window is January 2018 to December 2025 and the default city list contains 20 European destinations.

If your network requires a proxy, set one before running collection scripts:

```powershell
$env:HTTPS_PROXY="http://proxy.company.com:8080"
$env:HTTP_PROXY="http://proxy.company.com:8080"
python scripts/00_audit_city_candidates.py
```

You can also set `PROXY_URL` in `config.py`.

## Reddit Source Modes

`scripts/02_collect_reddit_posts.py` supports two source modes controlled in `config.py`:

- `REDDIT_SOURCE_MODE = "dump"`: parse local `.zst` files from `data/raw/reddit_dump_submissions`
- `REDDIT_SOURCE_MODE = "pushshift"`: use API calls (requires access and network)

For dump mode:

1. Download selected subreddit `*_submissions.zst` files into `data/raw/reddit_dump_submissions` (subfolders are allowed).
2. Set `REDDIT_TARGET_SUBREDDITS` in `config.py`.
3. Run:

```powershell
python scripts/02_collect_reddit_posts.py
```

Checkpointing stores processed dump files in `data/raw/reddit_collection_status.json`.

## Main Outputs

- `data/raw/google_trends_monthly.csv`: raw or previously collected Google Trends monthly data
- `data/raw/reddit_city_audit.csv`: pilot audit for city suitability
- `data/raw/reddit_city_audit_errors.csv`: per-city request errors from audit runs
- `data/raw/reddit_posts.csv`: raw Reddit submissions
- `data/raw/reddit_collection_status.json`: checkpoint file for completed Reddit cities
- `data/interim/reddit_monthly_features.csv`: monthly Reddit volume, engagement, and sentiment features
- `data/processed/model_dataset.csv`: final city-month modeling table
- `outputs/reports/model_metrics.csv`: model and baseline performance
- `outputs/reports/model_predictions.csv`: test-period predictions
- `outputs/reports/ablation_metrics.csv`: feature-group contribution comparison
- `outputs/reports/segment_metrics.csv`: stable vs volatile city performance
- `outputs/reports/lag_sensitivity_metrics.csv`: sentiment lag configuration comparison
- `outputs/reports/time_cv_metrics.csv`: rolling-origin time-series CV metrics
- `outputs/reports/dm_test_metrics.csv`: Diebold-Mariano style significance checks vs baseline

## Research Framing

Target variable:

```python
target_trend_next_month = trend_score(t + 1)
```

Core question:

Can current search interest and Reddit discussion signals predict next-month destination popularity?

## Reddit Checkpointing

`scripts/02_collect_reddit_posts.py` appends each fetched page directly into `data/raw/reddit_posts.csv`. If the process stops, running the script again resumes unfinished cities and skips cities already recorded in `data/raw/reddit_collection_status.json`.

## City Suitability Audit

Run `scripts/00_audit_city_candidates.py` before the full Reddit collection. It samples each city and reports volume, monthly coverage, travel-related text share, noisy-topic share, top subreddits, and a keep/review/drop recommendation.
