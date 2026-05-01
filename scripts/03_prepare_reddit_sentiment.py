import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    INTERIM_DIR,
    MIN_REDDIT_SCORE,
    MIN_TEXT_LENGTH,
    REDDIT_CLEAN_PATH,
    REDDIT_MONTHLY_PATH,
    REDDIT_RAW_PATH,
)


POSITIVE_WORDS = {
    "amazing",
    "beautiful",
    "best",
    "enjoy",
    "excellent",
    "favorite",
    "friendly",
    "good",
    "great",
    "happy",
    "love",
    "nice",
    "perfect",
    "safe",
    "wonderful",
}

NEGATIVE_WORDS = {
    "awful",
    "bad",
    "boring",
    "crowded",
    "dangerous",
    "dirty",
    "expensive",
    "hate",
    "horrible",
    "overrated",
    "problem",
    "rude",
    "scam",
    "terrible",
    "unsafe",
    "worst",
}


def build_sentiment_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return lambda text: analyzer.polarity_scores(text)["compound"]
    except ImportError:
        return lexicon_sentiment


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lexicon_sentiment(text):
    words = text.split()
    if not words:
        return 0.0
    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)
    return float(np.clip((positive - negative) / max(len(words), 1), -1, 1))


def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def main():
    if not REDDIT_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw Reddit file: {REDDIT_RAW_PATH}")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    analyzer = build_sentiment_analyzer()

    posts = pd.read_csv(REDDIT_RAW_PATH)
    if "city" in posts.columns:
        posts = posts.drop_duplicates(subset=["post_id", "city"]).copy()
    else:
        posts = posts.drop_duplicates(subset=["post_id"]).copy()
    posts["score"] = pd.to_numeric(posts["score"], errors="coerce").fillna(0)
    posts["num_comments"] = pd.to_numeric(posts["num_comments"], errors="coerce").fillna(0)

    posts["title_clean"] = posts["title"].apply(clean_text)
    posts["selftext_clean"] = posts["selftext"].apply(clean_text)
    posts["text_clean"] = (posts["title_clean"] + " " + posts["selftext_clean"]).str.strip()
    posts["text_length"] = posts["text_clean"].str.len()

    posts = posts[
        (posts["score"] >= MIN_REDDIT_SCORE)
        & (posts["text_length"] >= MIN_TEXT_LENGTH)
        & (~posts["author"].fillna("").str.lower().str.contains("bot"))
    ].copy()

    posts["created_at"] = pd.to_datetime(posts["created_utc"], unit="s", errors="coerce", utc=True)
    posts = posts.dropna(subset=["created_at", "city"])
    posts["month"] = posts["created_at"].dt.to_period("M").dt.to_timestamp("M")
    posts["sentiment_score"] = posts["text_clean"].apply(analyzer)
    posts["sentiment_label"] = posts["sentiment_score"].apply(sentiment_label)

    monthly = (
        posts.groupby(["city", "month"], as_index=False)
        .agg(
            reddit_post_count=("post_id", "count"),
            reddit_total_score=("score", "sum"),
            reddit_avg_score=("score", "mean"),
            reddit_total_comments=("num_comments", "sum"),
            reddit_avg_comments=("num_comments", "mean"),
            reddit_avg_sentiment=("sentiment_score", "mean"),
            reddit_sentiment_volatility=("sentiment_score", "std"),
            reddit_positive_share=("sentiment_label", lambda values: (values == "positive").mean()),
            reddit_negative_share=("sentiment_label", lambda values: (values == "negative").mean()),
        )
        .fillna({"reddit_sentiment_volatility": 0})
    )

    posts.to_csv(REDDIT_CLEAN_PATH, index=False)
    monthly.to_csv(REDDIT_MONTHLY_PATH, index=False)

    print(f"Saved cleaned Reddit posts: {REDDIT_CLEAN_PATH}")
    print(f"Saved monthly Reddit features: {REDDIT_MONTHLY_PATH}")


if __name__ == "__main__":
    main()
