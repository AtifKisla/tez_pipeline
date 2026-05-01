import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_DATASET_PATH, REPORT_DIR  # noqa: E402


TARGET = "target_trend_next_month"

TREND_FEATURES = [
    "city",
    "month_sin",
    "month_cos",
    "trend_score",
    "trend_lag_1",
    "trend_lag_2",
    "trend_lag_3",
    "trend_rolling_3",
    "trend_rolling_6",
    "trend_volatility_3",
    "trend_volatility_6",
    "trend_change",
    "trend_pct_change",
    "trend_momentum_2",
    "covid_period",
]

REDDIT_VOLUME_FEATURES = [
    "reddit_post_count",
    "reddit_total_score",
    "reddit_avg_score",
    "reddit_total_comments",
    "reddit_avg_comments",
    "reddit_post_count_lag_1",
    "reddit_post_count_lag_2",
    "reddit_post_count_lag_3",
    "reddit_post_count_rolling_3",
    "reddit_post_count_rolling_6",
]

REDDIT_SENTIMENT_FEATURES = [
    "reddit_avg_sentiment",
    "reddit_avg_sentiment_lag_1",
    "reddit_avg_sentiment_lag_2",
    "reddit_avg_sentiment_lag_3",
    "reddit_sentiment_rolling_3",
    "reddit_sentiment_rolling_6",
    "reddit_sentiment_change",
    "reddit_sentiment_volatility",
    "reddit_positive_share",
    "reddit_negative_share",
    "trend_sentiment_interaction",
]


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(features):
    numeric_features = [column for column in features if column != "city"]
    return ColumnTransformer(
        transformers=[
            ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
            ("num", StandardScaler(), numeric_features),
        ]
    )


def build_model(model_type, features):
    if model_type == "linear":
        estimator = LinearRegression()
    elif model_type == "rf":
        estimator = RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return Pipeline(steps=[("preprocessor", build_preprocessor(features)), ("model", estimator)])


def split_by_time(data, test_share=0.2):
    months = sorted(data["month"].unique())
    split_index = max(1, int(len(months) * (1 - test_share)))
    split_month = months[split_index]
    train = data[data["month"] < split_month].copy()
    test = data[data["month"] >= split_month].copy()
    return train, test, split_month


def winsorize_columns(frame, columns, lower=0.01, upper=0.99):
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        low = result[column].quantile(lower)
        high = result[column].quantile(upper)
        result[column] = result[column].clip(lower=low, upper=high)
    return result


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def dm_test(loss_baseline, loss_model):
    d = np.asarray(loss_baseline) - np.asarray(loss_model)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return {"dm_stat": np.nan, "p_value": np.nan}
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan}
    dm_stat = mean_d / math.sqrt(var_d / len(d))
    p_value = 2 * (1 - normal_cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def holdout_ablation(train, test):
    groups = {
        "trend_only": TREND_FEATURES,
        "trend_plus_reddit_volume": TREND_FEATURES + REDDIT_VOLUME_FEATURES,
        "trend_plus_reddit_full": TREND_FEATURES + REDDIT_VOLUME_FEATURES + REDDIT_SENTIMENT_FEATURES,
    }

    rows = []
    predictions = test[["city", "month", TARGET, "trend_score"]].copy()
    predictions["naive_current_trend"] = predictions["trend_score"]
    naive_metrics = evaluate(test[TARGET], predictions["naive_current_trend"])
    rows.append({"feature_group": "naive", "model": "naive_current_trend", **naive_metrics})

    for group_name, features in groups.items():
        for model_type in ("linear", "rf"):
            model = build_model(model_type, features)
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            column_name = f"{group_name}_{model_type}"
            predictions[column_name] = pred
            rows.append({"feature_group": group_name, "model": model_type, **evaluate(test[TARGET], pred)})

    return pd.DataFrame(rows), predictions


def city_volatility_segments(train):
    volatility = train.groupby("city")["trend_change"].std().fillna(0)
    threshold = volatility.median()
    segment = pd.Series(np.where(volatility >= threshold, "volatile", "stable"), index=volatility.index)
    return segment.to_dict(), float(threshold)


def segment_performance(predictions, segments):
    records = []
    model_columns = [column for column in predictions.columns if column not in {"city", "month", TARGET, "trend_score"}]
    for segment_name in ("stable", "volatile"):
        segment_data = predictions[predictions["city"].map(segments) == segment_name]
        if segment_data.empty:
            continue
        for model_name in model_columns:
            metrics = evaluate(segment_data[TARGET], segment_data[model_name])
            records.append(
                {
                    "segment": segment_name,
                    "model": model_name,
                    "rows": int(len(segment_data)),
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def lag_sensitivity(train, test):
    base_features = TREND_FEATURES + REDDIT_VOLUME_FEATURES
    lag_sets = {
        "lag1": [
            "reddit_avg_sentiment",
            "reddit_avg_sentiment_lag_1",
            "reddit_sentiment_rolling_3",
            "reddit_sentiment_change",
            "reddit_sentiment_volatility",
            "reddit_positive_share",
            "reddit_negative_share",
            "trend_sentiment_interaction",
        ],
        "lag1_lag2": [
            "reddit_avg_sentiment",
            "reddit_avg_sentiment_lag_1",
            "reddit_avg_sentiment_lag_2",
            "reddit_sentiment_rolling_3",
            "reddit_sentiment_rolling_6",
            "reddit_sentiment_change",
            "reddit_sentiment_volatility",
            "reddit_positive_share",
            "reddit_negative_share",
            "trend_sentiment_interaction",
        ],
        "lag1_lag2_lag3": REDDIT_SENTIMENT_FEATURES,
    }

    rows = []
    for lag_name, lag_features in lag_sets.items():
        features = base_features + lag_features
        model = build_model("linear", features)
        model.fit(train[features], train[TARGET])
        pred = model.predict(test[features])
        rows.append({"lag_config": lag_name, **evaluate(test[TARGET], pred)})
    return pd.DataFrame(rows)


def rolling_time_cv(data, features, min_train_months=24, test_horizon=6, step=6):
    months = sorted(data["month"].unique())
    rows = []
    fold = 0
    start = min_train_months
    while start + test_horizon <= len(months):
        fold += 1
        train_months = months[:start]
        test_months = months[start : start + test_horizon]
        train = data[data["month"].isin(train_months)]
        test = data[data["month"].isin(test_months)]
        if train.empty or test.empty:
            start += step
            continue

        naive_pred = test["trend_score"].values
        naive_metrics = evaluate(test[TARGET], naive_pred)
        rows.append(
            {
                "fold": fold,
                "model": "naive_current_trend",
                "train_end": pd.to_datetime(train_months[-1]).date(),
                "test_start": pd.to_datetime(test_months[0]).date(),
                "test_end": pd.to_datetime(test_months[-1]).date(),
                **naive_metrics,
            }
        )

        for model_name in ("linear", "rf"):
            model = build_model(model_name, features)
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "train_end": pd.to_datetime(train_months[-1]).date(),
                    "test_start": pd.to_datetime(test_months[0]).date(),
                    "test_end": pd.to_datetime(test_months[-1]).date(),
                    **evaluate(test[TARGET], pred),
                }
            )
        start += step

    return pd.DataFrame(rows)


def run_dm_tests(predictions, segments):
    rows = []
    target = predictions[TARGET].values
    naive_loss = (predictions["naive_current_trend"].values - target) ** 2
    for model_name in [column for column in predictions.columns if column.startswith("trend_plus_reddit_full_")]:
        model_loss = (predictions[model_name].values - target) ** 2
        dm = dm_test(naive_loss, model_loss)
        rows.append({"scope": "global", "model": model_name, **dm})

    for segment_name in ("stable", "volatile"):
        idx = predictions["city"].map(segments) == segment_name
        if idx.sum() < 5:
            continue
        segment_target = predictions.loc[idx, TARGET].values
        segment_naive_loss = (predictions.loc[idx, "naive_current_trend"].values - segment_target) ** 2
        for model_name in [column for column in predictions.columns if column.startswith("trend_plus_reddit_full_")]:
            segment_model_loss = (predictions.loc[idx, model_name].values - segment_target) ** 2
            dm = dm_test(segment_naive_loss, segment_model_loss)
            rows.append({"scope": segment_name, "model": model_name, **dm})

    return pd.DataFrame(rows)


def main():
    if not MODEL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {MODEL_DATASET_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(MODEL_DATASET_PATH)
    data["month"] = pd.to_datetime(data["month"])

    full_features = TREND_FEATURES + REDDIT_VOLUME_FEATURES + REDDIT_SENTIMENT_FEATURES
    numeric_features = [feature for feature in full_features if feature != "city"]
    data_robust = winsorize_columns(data, numeric_features)

    train, test, split_month = split_by_time(data_robust)
    print(f"Holdout split month: {pd.to_datetime(split_month).date()}")
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

    ablation_metrics, predictions = holdout_ablation(train, test)
    ablation_metrics.to_csv(REPORT_DIR / "ablation_metrics.csv", index=False)
    predictions.to_csv(REPORT_DIR / "ablation_predictions.csv", index=False)

    segments, threshold = city_volatility_segments(train)
    pd.DataFrame([{"volatility_threshold": threshold}]).to_csv(REPORT_DIR / "volatility_threshold.csv", index=False)
    pd.DataFrame([{"city": city, "segment": segment} for city, segment in segments.items()]).to_csv(
        REPORT_DIR / "city_segments.csv",
        index=False,
    )
    segment_metrics = segment_performance(predictions, segments)
    segment_metrics.to_csv(REPORT_DIR / "segment_metrics.csv", index=False)

    lag_metrics = lag_sensitivity(train, test)
    lag_metrics.to_csv(REPORT_DIR / "lag_sensitivity_metrics.csv", index=False)

    cv_metrics = rolling_time_cv(data_robust, full_features)
    cv_metrics.to_csv(REPORT_DIR / "time_cv_metrics.csv", index=False)

    dm_metrics = run_dm_tests(predictions, segments)
    dm_metrics.to_csv(REPORT_DIR / "dm_test_metrics.csv", index=False)

    print("Saved: ablation_metrics.csv, ablation_predictions.csv")
    print("Saved: city_segments.csv, segment_metrics.csv, lag_sensitivity_metrics.csv")
    print("Saved: time_cv_metrics.csv, dm_test_metrics.csv")


if __name__ == "__main__":
    main()
