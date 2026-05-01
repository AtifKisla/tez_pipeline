import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import METRICS_PATH, MODEL_DATASET_PATH, MODEL_DIR, PREDICTIONS_PATH, REPORT_DIR  # noqa: E402


FEATURE_COLUMNS = [
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
    "reddit_post_count",
    "reddit_total_score",
    "reddit_avg_score",
    "reddit_total_comments",
    "reddit_avg_comments",
    "reddit_avg_sentiment",
    "reddit_sentiment_volatility",
    "reddit_positive_share",
    "reddit_negative_share",
    "reddit_post_count_lag_1",
    "reddit_post_count_lag_2",
    "reddit_post_count_lag_3",
    "reddit_avg_sentiment_lag_1",
    "reddit_avg_sentiment_lag_2",
    "reddit_avg_sentiment_lag_3",
    "reddit_sentiment_rolling_3",
    "reddit_sentiment_rolling_6",
    "reddit_post_count_rolling_3",
    "reddit_post_count_rolling_6",
    "reddit_sentiment_change",
    "trend_sentiment_interaction",
    "covid_period",
]

TARGET_COLUMN = "target_trend_next_month"


def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }


def build_preprocessor():
    numeric_features = [column for column in FEATURE_COLUMNS if column != "city"]
    return ColumnTransformer(
        transformers=[
            ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
            ("num", StandardScaler(), numeric_features),
        ]
    )


def split_by_time(data, test_share=0.2):
    months = sorted(data["month"].unique())
    split_index = max(1, int(len(months) * (1 - test_share)))
    split_month = months[split_index]
    train = data[data["month"] < split_month].copy()
    test = data[data["month"] >= split_month].copy()
    return train, test


def main():
    if not MODEL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing modeling dataset: {MODEL_DATASET_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(MODEL_DATASET_PATH)
    data["month"] = pd.to_datetime(data["month"])
    train, test = split_by_time(data)

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    models = {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)),
            ]
        ),
    }

    metrics = [evaluate(y_test, test["trend_score"], "naive_current_trend")]
    predictions = test[["city", "month", "trend_score", TARGET_COLUMN]].copy()
    predictions["naive_current_trend"] = test["trend_score"]

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics.append(evaluate(y_test, y_pred, name))
        predictions[name] = y_pred
        if name == "random_forest":
            joblib.dump(model, MODEL_DIR / "random_forest.joblib")

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(pd.DataFrame(metrics).sort_values("mae"))


if __name__ == "__main__":
    main()
