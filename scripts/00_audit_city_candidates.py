import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    AUDIT_POSTS_PER_CITY,
    CITIES,
    PROXY_URL,
    RAW_DIR,
    REDDIT_CITY_AUDIT_PATH,
    REDDIT_CITY_AUDIT_ERRORS_PATH,
    REDDIT_END,
    REDDIT_START,
    REQUEST_SLEEP_SECONDS,
)


PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"

TRAVEL_HINTS = {
    "airport",
    "backpack",
    "booking",
    "city",
    "destination",
    "flight",
    "holiday",
    "hostel",
    "hotel",
    "itinerary",
    "museum",
    "restaurant",
    "sightseeing",
    "solo",
    "tour",
    "tourism",
    "tourist",
    "travel",
    "trip",
    "vacation",
    "visit",
    "visiting",
}

NOISY_HINTS = {
    "ancient",
    "empire",
    "historymemes",
    "minecraft",
    "nba",
    "nfl",
    "politics",
    "religion",
    "stock",
    "war",
}


def to_utc_timestamp(date_text):
    return int(datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc).timestamp())


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower()


def build_session():
    session = requests.Session()
    proxy_from_env = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    proxy = PROXY_URL or proxy_from_env
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    return session, proxy


def fetch_sample(session, city, start_ts, end_ts):
    posts = []
    before = end_ts

    while len(posts) < AUDIT_POSTS_PER_CITY:
        params = {
            "q": city,
            "after": start_ts,
            "before": before,
            "size": min(100, AUDIT_POSTS_PER_CITY - len(posts)),
            "sort": "desc",
            "sort_type": "created_utc",
        }
        response = session.get(PUSHSHIFT_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            break

        posts.extend(data)
        timestamps = [post["created_utc"] for post in data if post.get("created_utc")]
        if not timestamps:
            break
        next_before = min(timestamps)
        if next_before >= before:
            break
        before = next_before
        time.sleep(REQUEST_SLEEP_SECONDS)

    return posts


def summarize_city(city, posts):
    if not posts:
        return {
            "city": city,
            "sample_posts": 0,
            "months_covered": 0,
            "avg_score": 0,
            "avg_comments": 0,
            "travel_hint_share": 0,
            "noisy_hint_share": 0,
            "top_subreddits": "",
            "suitability_score": 0,
            "recommendation": "drop_or_review",
        }

    rows = []
    for post in posts:
        text = f"{normalize_text(post.get('title'))} {normalize_text(post.get('selftext'))}"
        subreddit = normalize_text(post.get("subreddit"))
        rows.append(
            {
                "created_at": pd.to_datetime(post.get("created_utc"), unit="s", errors="coerce", utc=True),
                "score": post.get("score", 0) or 0,
                "num_comments": post.get("num_comments", 0) or 0,
                "subreddit": subreddit,
                "has_travel_hint": any(word in text for word in TRAVEL_HINTS),
                "has_noisy_hint": any(word in text or word in subreddit for word in NOISY_HINTS),
            }
        )

    frame = pd.DataFrame(rows).dropna(subset=["created_at"])
    months_covered = frame["created_at"].dt.to_period("M").nunique()
    travel_hint_share = frame["has_travel_hint"].mean()
    noisy_hint_share = frame["has_noisy_hint"].mean()
    top_subreddits = ", ".join(frame["subreddit"].value_counts().head(5).index.tolist())

    coverage_score = min(months_covered / 24, 1)
    volume_score = min(len(frame) / AUDIT_POSTS_PER_CITY, 1)
    topical_score = min(travel_hint_share / 0.20, 1)
    noise_penalty = min(noisy_hint_share / 0.30, 1)
    suitability_score = round((0.35 * coverage_score + 0.30 * volume_score + 0.25 * topical_score - 0.20 * noise_penalty) * 100, 2)

    if suitability_score >= 65:
        recommendation = "keep"
    elif suitability_score >= 45:
        recommendation = "review"
    else:
        recommendation = "drop_or_review"

    return {
        "city": city,
        "sample_posts": len(frame),
        "months_covered": int(months_covered),
        "avg_score": round(frame["score"].mean(), 2),
        "avg_comments": round(frame["num_comments"].mean(), 2),
        "travel_hint_share": round(float(travel_hint_share), 3),
        "noisy_hint_share": round(float(noisy_hint_share), 3),
        "top_subreddits": top_subreddits,
        "suitability_score": suitability_score,
        "recommendation": recommendation,
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = to_utc_timestamp(REDDIT_START)
    end_ts = to_utc_timestamp(REDDIT_END)
    session, proxy = build_session()
    if proxy:
        print(f"Proxy enabled: {proxy}")
    else:
        print("Proxy not set. Using direct connection.")

    summaries = []
    errors = []
    for city in CITIES:
        print(f"Auditing {city}")
        try:
            posts = fetch_sample(session, city, start_ts, end_ts)
        except requests.RequestException as error:
            print(f"{city}: request failed -> {error}")
            errors.append({"city": city, "error": str(error)})
            continue
        summary = summarize_city(city, posts)
        summaries.append(summary)
        pd.DataFrame(summaries).sort_values("suitability_score", ascending=False).to_csv(
            REDDIT_CITY_AUDIT_PATH,
            index=False,
        )
        print(f"{city}: {summary['recommendation']} ({summary['suitability_score']})")

    if not summaries:
        pd.DataFrame(
            columns=[
                "city",
                "sample_posts",
                "months_covered",
                "avg_score",
                "avg_comments",
                "travel_hint_share",
                "noisy_hint_share",
                "top_subreddits",
                "suitability_score",
                "recommendation",
            ]
        ).to_csv(REDDIT_CITY_AUDIT_PATH, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(REDDIT_CITY_AUDIT_ERRORS_PATH, index=False)
        print(f"Saved city audit errors: {REDDIT_CITY_AUDIT_ERRORS_PATH}")

    print(f"Saved city audit: {REDDIT_CITY_AUDIT_PATH}")


if __name__ == "__main__":
    main()
