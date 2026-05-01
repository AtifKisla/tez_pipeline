import sys
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ANCHOR_CITY,
    CITIES,
    GOOGLE_TRENDS_END,
    GOOGLE_TRENDS_MONTHLY_PATH,
    GOOGLE_TRENDS_START,
    GOOGLE_TRENDS_WEEKLY_PATH,
    INTERIM_DIR,
    REQUEST_SLEEP_SECONDS,
)


def make_batches(cities, anchor_city, max_group_size=5):
    other_cities = [city for city in cities if city != anchor_city]
    batch_size = max_group_size - 1
    return [[anchor_city, *other_cities[i : i + batch_size]] for i in range(0, len(other_cities), batch_size)]


def fetch_group(pytrends, keywords, timeframe):
    pytrends.build_payload(keywords, timeframe=timeframe, geo="")
    time.sleep(REQUEST_SLEEP_SECONDS)
    data = pytrends.interest_over_time()
    if data.empty:
        raise ValueError(f"No Google Trends data returned for group: {keywords}")
    return data.drop(columns=["isPartial"], errors="ignore")


def normalize_with_anchor(group_data, reference_anchor_mean):
    anchor_mean = group_data[ANCHOR_CITY].mean()
    if anchor_mean == 0:
        raise ValueError(f"Anchor city mean is zero for {ANCHOR_CITY}; cannot normalize group.")
    return group_data * (reference_anchor_mean / anchor_mean)


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    timeframe = f"{GOOGLE_TRENDS_START} {GOOGLE_TRENDS_END}"
    pytrends = TrendReq()
    batches = make_batches(CITIES, ANCHOR_CITY)

    scaled_groups = []
    reference_anchor_mean = None

    for index, batch in enumerate(batches, start=1):
        print(f"Fetching Google Trends batch {index}: {batch}")
        group_data = fetch_group(pytrends, batch, timeframe)

        if reference_anchor_mean is None:
            reference_anchor_mean = group_data[ANCHOR_CITY].mean()
            scaled_groups.append(group_data)
        else:
            scaled_groups.append(normalize_with_anchor(group_data, reference_anchor_mean))

    weekly_wide = pd.concat(
        [scaled_groups[0], *[group.drop(columns=[ANCHOR_CITY]) for group in scaled_groups[1:]]],
        axis=1,
    )
    weekly_wide.index.name = "date"
    weekly_wide.to_csv(GOOGLE_TRENDS_WEEKLY_PATH)

    monthly_wide = weekly_wide.resample("ME").mean()
    monthly_long = (
        monthly_wide.reset_index()
        .melt(id_vars="date", var_name="city", value_name="trend_score")
        .sort_values(["city", "date"])
    )
    monthly_long.to_csv(GOOGLE_TRENDS_MONTHLY_PATH, index=False)

    print(f"Saved weekly data: {GOOGLE_TRENDS_WEEKLY_PATH}")
    print(f"Saved monthly data: {GOOGLE_TRENDS_MONTHLY_PATH}")


if __name__ == "__main__":
    main()
