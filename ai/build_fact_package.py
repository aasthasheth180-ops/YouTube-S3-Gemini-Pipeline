import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from build_history import (
    list_historical_snapshots,
    combine_historical_snapshots,
)

from analytics.historical_intelligence import (
    prepare_historical_intelligence,
)

from ml.build_ml_dataset import (
    create_historical_features,
    create_daily_prediction_rows,
)

from ml.train_model import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "trending_persistence_model.joblib",
)


def safe_value(value):
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def records_from_df(df, columns=None, limit=None):
    """
    Convert selected DataFrame rows into JSON-safe records.
    """

    result = df.copy()

    if columns is not None:
        available_columns = [
            column
            for column in columns
            if column in result.columns
        ]

        result = result[available_columns]

    if limit is not None:
        result = result.head(limit)

    records = result.to_dict(orient="records")

    return [
        {
            key: safe_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


def load_raw_history():
    """
    Load historical YouTube snapshots from S3.
    """

    keys = list_historical_snapshots()
    records = combine_historical_snapshots(keys)

    return pd.DataFrame(records)


def add_prediction_results(raw_df, latest_video_state):
    """
    Attach the Phase 8 Random Forest prediction to the latest
    end-of-day video-region states where model features are available.

    Gemini receives the prediction result as a calculated fact.
    Gemini does not calculate the prediction.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Reuse the exact Phase 8 feature engineering.
    feature_df = create_historical_features(raw_df)

    daily_rows = create_daily_prediction_rows(
        feature_df
    )

    latest_date = daily_rows["fetch_date"].max()

    prediction_rows = (
        daily_rows[
            daily_rows["fetch_date"] == latest_date
        ]
        .copy()
    )

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = prediction_rows[feature_columns]

    prediction_rows[
        "still_trending_tomorrow_prediction"
    ] = model.predict(X)

    if hasattr(model, "predict_proba"):
        prediction_rows[
            "still_trending_tomorrow_probability"
        ] = model.predict_proba(X)[:, 1]
    else:
        prediction_rows[
            "still_trending_tomorrow_probability"
        ] = np.nan

    prediction_columns = [
        "id",
        "region_code",
        "fetch_date",
        "still_trending_tomorrow_prediction",
        "still_trending_tomorrow_probability",
    ]

    predictions = prediction_rows[
        prediction_columns
    ].copy()

    result = latest_video_state.merge(
        predictions,
        on=["id", "region_code"],
        how="left",
        suffixes=("", "_prediction"),
    )

    return result, latest_date


def build_fact_package():
    """
    Build a compact, structured collection of facts for Gemini.

    All numeric calculations happen before Gemini receives the data.
    """

    raw_df = load_raw_history()

    intelligence = prepare_historical_intelligence(
        raw_df
    )

    latest_state, prediction_date = (
        add_prediction_results(
            raw_df,
            intelligence["latest_video_state"],
        )
    )

    # ---------------------------------------------------------
    # RISING VIDEOS
    # ---------------------------------------------------------

    # ---------------------------------------------------------
# CURRENT RISING VIDEOS
# ---------------------------------------------------------

# The Phase 8 prediction date represents the latest
# collection day available to the prediction system.
    current_date = pd.to_datetime(
        prediction_date
    ).normalize()

    latest_state["fetch_date_normalized"] = (
        pd.to_datetime(
            latest_state["fetch_timestamp"],
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Only videos actually observed on the current collection day
    # can be described as currently rising.
    rising = latest_state[
        (latest_state["rank_movement"] > 0)
        & (
            latest_state["fetch_date_normalized"]
            == current_date
        )
    ].copy()

    rising = rising.sort_values(
        [
            "rank_movement",
            "view_velocity",
        ],
        ascending=[False, False],
    )

    rising = rising.sort_values(
        [
            "rank_movement",
            "view_velocity",
        ],
        ascending=[False, False],
    )

    rising_facts = records_from_df(
        rising,
        columns=[
            "id",
            "video_title",
            "channel_name",
            "category_name",
            "region_code",
            "fetch_timestamp",
            "previous_rank",
            "trending_rank",
            "rank_movement",
            "view_count",
            "view_growth",
            "view_velocity",
            "like_velocity",
            "comment_velocity",
            "engagement_rate",
            "engagement_rate_movement",
            "trending_observations",
            "hours_trending",
            "days_trending",
            "still_trending_tomorrow_prediction",
            "still_trending_tomorrow_probability",
        ],
        limit=10,
    )

    # ---------------------------------------------------------
    # CHANNEL FACTS
    # ---------------------------------------------------------

    channel_facts_df = (
        intelligence["channel_trends"]
        .sort_values(
            [
                "unique_trending_videos",
                "trending_observations",
            ],
            ascending=False,
        )
    )

    channel_facts = records_from_df(
        channel_facts_df,
        limit=12,
    )

    # ---------------------------------------------------------
    # CATEGORY FACTS
    # ---------------------------------------------------------

    category_facts_df = (
        intelligence["category_trends"]
        .sort_values(
            "trending_observations",
            ascending=False,
        )
    )

    category_facts = records_from_df(
        category_facts_df,
        limit=12,
    )

    # ---------------------------------------------------------
    # REGIONAL FACTS
    # ---------------------------------------------------------

    regional_facts = records_from_df(
        intelligence["regional_trends"]
        .sort_values("region_code")
    )

    # ---------------------------------------------------------
    # GLOBAL VS REGIONAL FACTS
    # ---------------------------------------------------------

    global_regional_facts = records_from_df(
        intelligence["global_vs_regional"]
    )

    package = {
        "metadata": {
            "fact_source": (
                "Python-calculated historical "
                "YouTube metrics"
            ),
            "historical_observations": int(
                len(raw_df)
            ),
            "regions": sorted(
                raw_df["region_code"]
                .dropna()
                .unique()
                .tolist()
            ),
            "prediction_model": (
                "Random Forest trend persistence model"
            ),
            "prediction_question": (
                "Will this video still be trending "
                "tomorrow in the same region?"
            ),
            "prediction_date": safe_value(
                prediction_date
            ),
            "important_constraint": (
                "All calculations are performed in "
                "Python. AI may interpret these facts "
                "but must not invent or recalculate "
                "metrics."
            ),
        },

        "rising_videos": rising_facts,

        "channel_trends": channel_facts,

        "category_trends": category_facts,

        "regional_trends": regional_facts,

        "global_vs_regional": (
            global_regional_facts
        ),
    }

    return package


if __name__ == "__main__":

    fact_package = build_fact_package()

    print(
        json.dumps(
            fact_package,
            indent=2,
            ensure_ascii=False,
        )
    )