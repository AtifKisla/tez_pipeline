import sys
import math
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    GOOGLE_TRENDS_START,
    GOOGLE_TRENDS_MONTHLY_PATH,
    GOOGLE_TRENDS_RAW_PATH,
    MODEL_DATASET_PATH,
    PROCESSED_DIR,
    REDDIT_MONTHLY_PATH,
)


REDDIT_FILL_ZERO_COLUMNS = [
    "reddit_post_count",
    "reddit_total_score",
    "reddit_avg_score",
    "reddit_total_comments",
    "reddit_avg_comments",
    "reddit_avg_sentiment",
    "reddit_sentiment_volatility",
    "reddit_positive_share",
    "reddit_negative_share",
]


def load_trends():
    path = GOOGLE_TRENDS_MONTHLY_PATH if GOOGLE_TRENDS_MONTHLY_PATH.exists() else GOOGLE_TRENDS_RAW_PATH
    if not path.exists():
        raise FileNotFoundError("Missing Google Trends monthly file.")

    trends = pd.read_csv(path)
    if "trend" in trends.columns and "trend_score" not in trends.columns:
        trends = trends.rename(columns={"trend": "trend_score"})

    required = {"city", "date", "trend_score"}
    missing = required - set(trends.columns)
    if missing:
        raise ValueError(f"Google Trends file is missing columns: {sorted(missing)}")

    trends["month"] = pd.to_datetime(trends["date"]).dt.to_period("M").dt.to_timestamp("M")
    trends = trends[["city", "month", "trend_score"]]
    expected_start = pd.to_datetime(GOOGLE_TRENDS_START).to_period("M").to_timestamp("M")
    min_month = trends["month"].min()
    if pd.notna(min_month) and min_month > expected_start:
        raise ValueError(
            f"Google Trends data starts at {min_month.date()}, expected start is {expected_start.date()}. "
            "Please provide a wider-range monthly file."
        )
    return trends.sort_values(["city", "month"])


def load_reddit():
    if not REDDIT_MONTHLY_PATH.exists():
        raise FileNotFoundError(f"Missing Reddit monthly file: {REDDIT_MONTHLY_PATH}")
    reddit = pd.read_csv(REDDIT_MONTHLY_PATH)
    reddit["month"] = pd.to_datetime(reddit["month"]).dt.to_period("M").dt.to_timestamp("M")
    return reddit


def add_features(data):
    data = data.sort_values(["city", "month"]).copy()
    grouped = data.groupby("city", group_keys=False)

    data["month_number"] = data["month"].dt.month
    data["month_sin"] = (2 * math.pi * data["month_number"] / 12).apply(math.sin)
    data["month_cos"] = (2 * math.pi * data["month_number"] / 12).apply(math.cos)

    data["trend_lag_1"] = grouped["trend_score"].shift(1)
    data["trend_lag_2"] = grouped["trend_score"].shift(2)
    data["trend_lag_3"] = grouped["trend_score"].shift(3)
    data["trend_rolling_3"] = grouped["trend_score"].transform(lambda values: values.rolling(3, min_periods=1).mean())
    data["trend_rolling_6"] = grouped["trend_score"].transform(lambda values: values.rolling(6, min_periods=1).mean())
    data["trend_volatility_3"] = grouped["trend_score"].transform(lambda values: values.rolling(3, min_periods=2).std())
    data["trend_volatility_6"] = grouped["trend_score"].transform(lambda values: values.rolling(6, min_periods=2).std())
    data["trend_change"] = grouped["trend_score"].diff()
    data["trend_pct_change"] = grouped["trend_score"].pct_change()
    data["trend_momentum_2"] = data["trend_score"] - data["trend_lag_2"]

    data["reddit_post_count_lag_1"] = grouped["reddit_post_count"].shift(1)
    data["reddit_post_count_lag_2"] = grouped["reddit_post_count"].shift(2)
    data["reddit_post_count_lag_3"] = grouped["reddit_post_count"].shift(3)
    data["reddit_avg_sentiment_lag_1"] = grouped["reddit_avg_sentiment"].shift(1)
    data["reddit_avg_sentiment_lag_2"] = grouped["reddit_avg_sentiment"].shift(2)
    data["reddit_avg_sentiment_lag_3"] = grouped["reddit_avg_sentiment"].shift(3)
    data["reddit_sentiment_rolling_3"] = grouped["reddit_avg_sentiment"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    data["reddit_sentiment_rolling_6"] = grouped["reddit_avg_sentiment"].transform(
        lambda values: values.rolling(6, min_periods=1).mean()
    )
    data["reddit_post_count_rolling_3"] = grouped["reddit_post_count"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    data["reddit_post_count_rolling_6"] = grouped["reddit_post_count"].transform(
        lambda values: values.rolling(6, min_periods=1).mean()
    )
    data["reddit_sentiment_change"] = grouped["reddit_avg_sentiment"].diff()
    data["trend_sentiment_interaction"] = data["trend_score"] * data["reddit_avg_sentiment"]
    data["covid_period"] = data["month"].between(pd.Timestamp("2020-03-31"), pd.Timestamp("2021-12-31")).astype(int)
    data["target_trend_next_month"] = grouped["trend_score"].shift(-1)

    return data


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    trends = load_trends()
    reddit = load_reddit()
    data = trends.merge(reddit, on=["city", "month"], how="left")

    for column in REDDIT_FILL_ZERO_COLUMNS:
        if column not in data.columns:
            data[column] = 0
        data[column] = data[column].fillna(0)

    data = add_features(data)
    data = data.dropna(subset=["target_trend_next_month", "trend_lag_1", "trend_lag_2", "trend_lag_3"]).copy()
    data = data.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    data.to_csv(MODEL_DATASET_PATH, index=False)

    print(f"Saved modeling dataset: {MODEL_DATASET_PATH}")
    print(f"Rows: {len(data)}")


if __name__ == "__main__":
    main()
