import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CITIES,
    MAX_POSTS_PER_CITY,
    PROXY_URL,
    RAW_DIR,
    REDDIT_DUMP_DIR,
    REDDIT_DUMP_FILE_GLOB,
    REDDIT_END,
    REDDIT_RAW_PATH,
    REDDIT_SOURCE_MODE,
    REDDIT_START,
    REDDIT_STATUS_PATH,
    REDDIT_TARGET_SUBREDDITS,
    REQUEST_SLEEP_SECONDS,
)


PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"


def to_utc_timestamp(date_text):
    return int(datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc).timestamp())


def load_status():
    if not REDDIT_STATUS_PATH.exists():
        return {"completed_cities": [], "processed_files": [], "source_mode": REDDIT_SOURCE_MODE}
    with REDDIT_STATUS_PATH.open("r", encoding="utf-8") as file:
        status = json.load(file)
    status.setdefault("completed_cities", [])
    status.setdefault("processed_files", [])
    status.setdefault("source_mode", REDDIT_SOURCE_MODE)
    return status


def save_status(status):
    with REDDIT_STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)


def load_existing_posts():
    if not REDDIT_RAW_PATH.exists():
        return pd.DataFrame(), set(), {city: 0 for city in CITIES}

    posts = pd.read_csv(REDDIT_RAW_PATH)
    if posts.empty:
        return posts, set(), {city: 0 for city in CITIES}

    if "city" not in posts.columns or "post_id" not in posts.columns:
        return posts, set(), {city: 0 for city in CITIES}

    seen_keys = set((posts["post_id"].astype(str) + "|" + posts["city"].astype(str)).tolist())
    city_counts = posts["city"].value_counts().to_dict()
    for city in CITIES:
        city_counts.setdefault(city, 0)
    return posts, seen_keys, city_counts


def append_posts(posts):
    if not posts:
        return
    frame = pd.DataFrame(posts)
    frame.to_csv(REDDIT_RAW_PATH, mode="a", header=not REDDIT_RAW_PATH.exists(), index=False)


def compile_city_patterns(cities):
    patterns = {}
    for city in cities:
        pattern = r"\b" + re.escape(city.lower()) + r"\b"
        patterns[city] = re.compile(pattern)
    return patterns


def match_cities(text, city_patterns):
    matches = []
    for city, pattern in city_patterns.items():
        if pattern.search(text):
            matches.append(city)
    return matches


def build_session():
    session = requests.Session()
    proxy_from_env = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    proxy = PROXY_URL or proxy_from_env
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
    return session, proxy


def collect_from_dump(start_ts, end_ts):
    if not REDDIT_DUMP_DIR.exists():
        raise FileNotFoundError(f"Dump folder not found: {REDDIT_DUMP_DIR}")

    status = load_status()
    processed_files = set(status.get("processed_files", []))
    _, seen_keys, city_counts = load_existing_posts()
    city_patterns = compile_city_patterns(CITIES)
    target_subreddits = {name.lower() for name in REDDIT_TARGET_SUBREDDITS}

    dump_files = sorted(REDDIT_DUMP_DIR.glob(REDDIT_DUMP_FILE_GLOB))
    if not dump_files:
        raise FileNotFoundError(f"No files matched pattern {REDDIT_DUMP_FILE_GLOB} under {REDDIT_DUMP_DIR}")

    total_appended = 0
    for file_path in dump_files:
        relative_name = str(file_path.relative_to(REDDIT_DUMP_DIR))
        if relative_name in processed_files:
            print(f"Skipping processed file: {relative_name}")
            continue

        print(f"Processing dump file: {relative_name}")
        batch = []
        with file_path.open("rb") as source:
            if file_path.suffix.lower() == ".zst":
                try:
                    import zstandard as zstd
                except ImportError as error:
                    raise ImportError(
                        "Missing dependency 'zstandard'. Install it with 'pip install zstandard' "
                        "or switch REDDIT_DUMP_FILE_GLOB to plain .ndjson files."
                    ) from error
                reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(source)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            else:
                text_stream = io.TextIOWrapper(source, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    continue

                subreddit = str(post.get("subreddit", "")).lower()
                if target_subreddits and subreddit not in target_subreddits:
                    continue

                created_utc = post.get("created_utc")
                try:
                    created_utc = int(created_utc)
                except (TypeError, ValueError):
                    continue
                if created_utc < start_ts or created_utc > end_ts:
                    continue

                title = str(post.get("title") or "")
                selftext = str(post.get("selftext") or "")
                full_text = f"{title} {selftext}".lower()
                cities_in_post = match_cities(full_text, city_patterns)
                if not cities_in_post:
                    continue

                post_id = str(post.get("id") or "")
                if not post_id:
                    continue

                for city in cities_in_post:
                    if MAX_POSTS_PER_CITY is not None and city_counts.get(city, 0) >= MAX_POSTS_PER_CITY:
                        continue

                    dedup_key = f"{post_id}|{city}"
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)
                    city_counts[city] = city_counts.get(city, 0) + 1
                    batch.append(
                        {
                            "post_id": post_id,
                            "city": city,
                            "title": title,
                            "selftext": selftext,
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "created_utc": created_utc,
                            "subreddit": post.get("subreddit"),
                            "author": post.get("author"),
                            "url": post.get("full_link") or post.get("url"),
                            "source_mode": "dump",
                            "source_file": relative_name,
                        }
                    )
                    total_appended += 1

                if len(batch) >= 1000:
                    append_posts(batch)
                    batch = []
                    print(f"{relative_name}: checkpoint appended, total rows {total_appended}")

        append_posts(batch)
        processed_files.add(relative_name)
        status["processed_files"] = sorted(processed_files)
        status["source_mode"] = "dump"
        save_status(status)
        print(f"Completed file: {relative_name}")

    print(f"Saved Reddit raw data: {REDDIT_RAW_PATH}")
    print(f"Total appended rows in this run: {total_appended}")


def starting_before_for_city(existing_posts, city, end_ts):
    if existing_posts.empty or "city" not in existing_posts.columns or "created_utc" not in existing_posts.columns:
        return end_ts
    city_posts = existing_posts[existing_posts["city"] == city].copy()
    if city_posts.empty:
        return end_ts
    city_posts["created_utc"] = pd.to_numeric(city_posts["created_utc"], errors="coerce")
    oldest_timestamp = city_posts["created_utc"].dropna().min()
    if pd.isna(oldest_timestamp):
        return end_ts
    return int(oldest_timestamp)


def fetch_city_posts_pushshift(session, city, start_ts, end_ts, existing_posts, seen_keys, city_counts):
    before = starting_before_for_city(existing_posts, city, end_ts)
    print(f"{city}: starting from {datetime.fromtimestamp(before, tz=timezone.utc).date()}")

    while True:
        if MAX_POSTS_PER_CITY is not None and city_counts.get(city, 0) >= MAX_POSTS_PER_CITY:
            print(f"{city}: reached MAX_POSTS_PER_CITY={MAX_POSTS_PER_CITY}")
            break

        params = {
            "q": city,
            "after": start_ts,
            "before": before,
            "size": 100,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        response = session.get(PUSHSHIFT_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            break

        batch_posts = []
        for post in data:
            post_id = str(post.get("id") or "")
            if not post_id:
                continue
            dedup_key = f"{post_id}|{city}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            city_counts[city] = city_counts.get(city, 0) + 1
            batch_posts.append(
                {
                    "post_id": post_id,
                    "city": city,
                    "title": post.get("title"),
                    "selftext": post.get("selftext"),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc"),
                    "subreddit": post.get("subreddit"),
                    "author": post.get("author"),
                    "url": post.get("full_link") or post.get("url"),
                    "source_mode": "pushshift",
                    "source_file": "",
                }
            )
            if MAX_POSTS_PER_CITY is not None and city_counts.get(city, 0) >= MAX_POSTS_PER_CITY:
                break

        append_posts(batch_posts)
        timestamps = [post["created_utc"] for post in data if post.get("created_utc")]
        if not timestamps:
            break
        next_before = min(timestamps)
        if next_before >= before:
            break
        before = next_before
        print(f"{city}: collected/checkpointed {city_counts.get(city, 0)} posts")
        time.sleep(REQUEST_SLEEP_SECONDS)


def collect_from_pushshift(start_ts, end_ts):
    session, proxy = build_session()
    if proxy:
        print(f"Proxy enabled: {proxy}")
    else:
        print("Proxy not set. Using direct connection.")

    status = load_status()
    completed_cities = set(status.get("completed_cities", []))
    existing_posts, seen_keys, city_counts = load_existing_posts()

    for city in CITIES:
        if city in completed_cities:
            print(f"Skipping {city}: already completed")
            continue
        print(f"Fetching Reddit posts for {city}")
        try:
            fetch_city_posts_pushshift(session, city, start_ts, end_ts, existing_posts, seen_keys, city_counts)
        except requests.RequestException as error:
            print(f"{city}: request failed -> {error}")
            continue
        completed_cities.add(city)
        status["completed_cities"] = sorted(completed_cities)
        status["source_mode"] = "pushshift"
        save_status(status)
        existing_posts, seen_keys, city_counts = load_existing_posts()
        print(f"{city}: completed with {city_counts.get(city, 0)} checkpointed posts")

    print(f"Saved Reddit raw data: {REDDIT_RAW_PATH}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_ts = to_utc_timestamp(REDDIT_START)
    end_ts = to_utc_timestamp(REDDIT_END)

    if REDDIT_SOURCE_MODE == "dump":
        collect_from_dump(start_ts, end_ts)
        return
    if REDDIT_SOURCE_MODE == "pushshift":
        collect_from_pushshift(start_ts, end_ts)
        return
    raise ValueError(f"Unsupported REDDIT_SOURCE_MODE: {REDDIT_SOURCE_MODE}")


if __name__ == "__main__":
    main()
