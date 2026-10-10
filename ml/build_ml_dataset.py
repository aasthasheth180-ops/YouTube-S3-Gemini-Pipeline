import os
import sys
import pandas as pd


# ---------------------------------------------------------
# ALLOW IMPORTS FROM PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from build_history import (
    list_historical_snapshots,
    combine_historical_snapshots
)


# =========================================================
# LOAD HISTORICAL YOUTUBE DATA
# =========================================================

def load_historical_data():

    snapshot_keys = list_historical_snapshots()

    print(
        f"Found {len(snapshot_keys)} historical snapshots"
    )

    records = combine_historical_snapshots(
        snapshot_keys
    )

    print(
        f"Found {len(records)} historical video observations"
    )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "Historical dataset is empty."
        )

    df["fetch_timestamp"] = pd.to_datetime(
        df["fetch_timestamp"],
        utc=True,
        errors="coerce"
    )

    df["fetch_date"] = pd.to_datetime(
        df["fetch_date"],
        errors="coerce"
    )

    return df


def create_historical_features(df):
    """
    Create leakage-safe historical features.

    Every feature is calculated using information available
    at or before the current observation.
    """

    feature_df = df.copy()

    # =====================================================
    # BASIC CLEANING
    # =====================================================

    feature_df["fetch_timestamp"] = pd.to_datetime(
        feature_df["fetch_timestamp"],
        utc=True,
        errors="coerce"
    )

    feature_df["fetch_date"] = pd.to_datetime(
        feature_df["fetch_date"],
        errors="coerce"
    )

    numeric_columns = [
        "trending_rank",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count"
    ]

    for column in numeric_columns:

        if column in feature_df.columns:

            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce"
            )

    # Statistics are stored inside the YouTube
    # statistics dictionary.

    feature_df["view_count"] = pd.to_numeric(
        feature_df["statistics"].apply(
            lambda x: x.get("viewCount")
            if isinstance(x, dict)
            else None
        ),
        errors="coerce"
    )

    feature_df["like_count"] = pd.to_numeric(
        feature_df["statistics"].apply(
            lambda x: x.get("likeCount")
            if isinstance(x, dict)
            else None
        ),
        errors="coerce"
    )

    feature_df["comment_count"] = pd.to_numeric(
        feature_df["statistics"].apply(
            lambda x: x.get("commentCount")
            if isinstance(x, dict)
            else None
        ),
        errors="coerce"
    )

    # Chronological order is critical before shift/diff/cumcount.

    feature_df = feature_df.sort_values(
        [
            "id",
            "region_code",
            "fetch_timestamp"
        ]
    ).reset_index(drop=True)

    video_region_group = feature_df.groupby(
        [
            "id",
            "region_code"
        ],
        sort=False
    )

    # =====================================================
    # GROUP 1 — RANK FEATURES
    # =====================================================

    feature_df["previous_rank"] = (
        video_region_group[
            "trending_rank"
        ].shift(1)
    )

    # Positive value means the video moved toward rank #1.
    #
    # Example:
    # previous rank = 20
    # current rank  = 12
    # rank movement = +8

    feature_df["rank_movement"] = (
        feature_df["previous_rank"]
        - feature_df["trending_rank"]
    )

    # trending_rank itself is our current-rank feature.

    # =====================================================
    # GROUP 1 — LIFECYCLE FEATURES
    # =====================================================

    # Current observation included:
    # first observation = 1, second = 2, etc.

    feature_df["observations_so_far"] = (
        video_region_group.cumcount() + 1
    )

    feature_df["first_seen_so_far"] = (
    video_region_group[
        "fetch_timestamp"
    ].transform("min")
    )

    feature_df["hours_trending_so_far"] = (
        (
            feature_df["fetch_timestamp"]
            - feature_df["first_seen_so_far"]
        )
        .dt.total_seconds()
        / 3600
    )

    feature_df["days_trending_so_far"] = (
        feature_df["hours_trending_so_far"]
        / 24
    )

    # =====================================================
    # GROUP 2 — PREVIOUS OBSERVATION TIME
    # =====================================================

    feature_df["previous_fetch_timestamp"] = (
        video_region_group[
            "fetch_timestamp"
        ].shift(1)
    )

    feature_df["elapsed_hours"] = (
        (
            feature_df["fetch_timestamp"]
            - feature_df["previous_fetch_timestamp"]
        )
        .dt.total_seconds()
        / 3600
    )

    # =====================================================
    # GROUP 2 — VIEW GROWTH + VELOCITY
    # =====================================================

    feature_df["previous_view_count"] = (
        video_region_group[
            "view_count"
        ].shift(1)
    )

    feature_df["view_growth"] = (
        feature_df["view_count"]
        - feature_df["previous_view_count"]
    )

    feature_df["view_velocity"] = (
        feature_df["view_growth"]
        / feature_df["elapsed_hours"]
    )

    feature_df.loc[
        feature_df["elapsed_hours"] <= 0,
        "view_velocity"
    ] = pd.NA

    # =====================================================
    # GROUP 2 — LIKE VELOCITY
    # =====================================================

    feature_df["previous_like_count"] = (
        video_region_group[
            "like_count"
        ].shift(1)
    )

    feature_df["like_growth"] = (
        feature_df["like_count"]
        - feature_df["previous_like_count"]
    )

    feature_df["like_velocity"] = (
        feature_df["like_growth"]
        / feature_df["elapsed_hours"]
    )

    feature_df.loc[
        feature_df["elapsed_hours"] <= 0,
        "like_velocity"
    ] = pd.NA

    # =====================================================
    # GROUP 2 — COMMENT VELOCITY
    # =====================================================

    feature_df["previous_comment_count"] = (
        video_region_group[
            "comment_count"
        ].shift(1)
    )

    feature_df["comment_growth"] = (
        feature_df["comment_count"]
        - feature_df["previous_comment_count"]
    )

    feature_df["comment_velocity"] = (
        feature_df["comment_growth"]
        / feature_df["elapsed_hours"]
    )

    feature_df.loc[
        feature_df["elapsed_hours"] <= 0,
        "comment_velocity"
    ] = pd.NA

    # =====================================================
    # GROUP 2 — ENGAGEMENT
    # =====================================================

    feature_df["engagement_rate"] = (
        (
            feature_df["like_count"]
            + feature_df["comment_count"]
        )
        / feature_df["view_count"]
    )

    feature_df.loc[
        feature_df["view_count"] <= 0,
        "engagement_rate"
    ] = pd.NA

    feature_df["previous_engagement_rate"] = (
        video_region_group[
            "engagement_rate"
        ].shift(1)
    )

    feature_df["engagement_movement"] = (
        feature_df["engagement_rate"]
        - feature_df["previous_engagement_rate"]
    )

    # =====================================================
    # GROUP 3 — CONTENT / CATEGORY FEATURES
    # =====================================================

    # category_name is already enriched in the historical
    # pipeline. Keep it as a categorical ML feature.

    if "category_name" not in feature_df.columns:
        feature_df["category_name"] = "Unknown"

    feature_df["category_name"] = (
        feature_df["category_name"]
        .fillna("Unknown")
        .astype(str)
    )

    # =====================================================
    # GROUP 3 — CHANNEL FEATURES
    # =====================================================

    channel_columns = [
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count"
    ]

    for column in channel_columns:

        if column not in feature_df.columns:
            feature_df[column] = pd.NA

        feature_df[column] = pd.to_numeric(
            feature_df[column],
            errors="coerce"
        )

    # =====================================================
    # GROUP 4 — REGIONAL / CROSS-REGION FEATURES
    # =====================================================

    # Count how many regions contain the same video
    # during the same collection day.
    #
    # IMPORTANT:
    # This uses only the current day's presence,
    # not future region information.

    daily_region_reach = (
        feature_df[
            [
                "id",
                "fetch_date",
                "region_code"
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "id",
                "fetch_date"
            ]
        )["region_code"]
        .nunique()
        .rename(
            "regions_trending_today"
        )
        .reset_index()
    )

    feature_df = feature_df.merge(
        daily_region_reach,
        on=[
            "id",
            "fetch_date"
        ],
        how="left"
    )

    feature_df["cross_region_today"] = (
        feature_df[
            "regions_trending_today"
        ] >= 2
    ).astype(int)

    return feature_df



def create_daily_prediction_rows(feature_df):
    """
    Keep the latest available observation for each
    video-region-day.

    This becomes the prediction point for:
    'Will this video still be trending tomorrow?'
    """

    daily_prediction_rows = (
        feature_df
        .sort_values(
            "fetch_timestamp"
        )
        .groupby(
            [
                "id",
                "region_code",
                "fetch_date"
            ],
            as_index=False
        )
        .tail(1)
        .copy()
    )

    daily_prediction_rows = (
        daily_prediction_rows
        .sort_values(
            [
                "fetch_date",
                "fetch_timestamp",
                "region_code",
                "id"
            ]
        )
        .reset_index(drop=True)
    )

    return daily_prediction_rows


# =========================================================
# CREATE NEXT-DAY TARGET
# =========================================================

def create_next_day_target(df):

    # -----------------------------------------------------
    # ONE VIDEO / REGION / DAY
    # -----------------------------------------------------

    daily_video_presence = (
        df[
            [
                "id",
                "region_code",
                "fetch_date"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    # -----------------------------------------------------
    # DAYS EACH REGION WAS ACTUALLY COLLECTED
    # -----------------------------------------------------

    region_collection_days = (
        daily_video_presence[
            [
                "region_code",
                "fetch_date"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    # -----------------------------------------------------
    # NEXT-DAY VIDEO LOOKUP
    #
    # Shift tomorrow backward one day so that:
    #
    # Oct 9 presence
    # becomes lookup information for Oct 8.
    # -----------------------------------------------------

    next_day_video_presence = (
        daily_video_presence.copy()
    )

    next_day_video_presence["fetch_date"] = (
        next_day_video_presence["fetch_date"]
        - pd.Timedelta(days=1)
    )

    next_day_video_presence[
        "still_trending_tomorrow"
    ] = 1

    # -----------------------------------------------------
    # NEXT-DAY REGION COLLECTION LOOKUP
    # -----------------------------------------------------

    next_day_region_collection = (
        region_collection_days.copy()
    )

    next_day_region_collection["fetch_date"] = (
        next_day_region_collection["fetch_date"]
        - pd.Timedelta(days=1)
    )

    next_day_region_collection[
        "has_next_day_collection"
    ] = True

    # -----------------------------------------------------
    # MERGE TARGET INFORMATION
    # -----------------------------------------------------

    target_df = daily_video_presence.merge(
        next_day_region_collection,
        on=[
            "region_code",
            "fetch_date"
        ],
        how="left"
    )

    target_df = target_df.merge(
        next_day_video_presence[
            [
                "id",
                "region_code",
                "fetch_date",
                "still_trending_tomorrow"
            ]
        ],
        on=[
            "id",
            "region_code",
            "fetch_date"
        ],
        how="left"
    )

    # -----------------------------------------------------
    # CREATE NEGATIVE LABELS
    # -----------------------------------------------------

    # Tomorrow exists for this region, but this video
    # does not appear tomorrow.
    target_df.loc[
        (
            target_df["has_next_day_collection"] == True
        )
        &
        (
            target_df[
                "still_trending_tomorrow"
            ].isna()
        ),
        "still_trending_tomorrow"
    ] = 0

    # If tomorrow itself was not collected, we do not know
    # whether the video remained trending.
    target_df.loc[
        target_df[
            "has_next_day_collection"
        ].isna(),
        "still_trending_tomorrow"
    ] = pd.NA

    target_df[
        "still_trending_tomorrow"
    ] = target_df[
        "still_trending_tomorrow"
    ].astype("Int64")

    return target_df


def build_ml_dataset(daily_features, target_df):
    """
    Merge leakage-safe daily features with the
    next-day trending target.
    """

    ml_df = daily_features.merge(
        target_df[
            [
                "id",
                "region_code",
                "fetch_date",
                "still_trending_tomorrow"
            ]
        ],
        on=[
            "id",
            "region_code",
            "fetch_date"
        ],
        how="left",
        validate="one_to_one"
    )

    # Rows without a next-day collection cannot be used
    # for supervised training.
    ml_df = ml_df.dropna(
        subset=[
            "still_trending_tomorrow"
        ]
    ).copy()

    ml_df[
        "still_trending_tomorrow"
    ] = ml_df[
        "still_trending_tomorrow"
    ].astype(int)

    return ml_df



def prepare_ml_dataset():
    """
    Build and return the complete leakage-safe
    supervised learning dataset.
    """

    df = load_historical_data()

    feature_df = create_historical_features(
        df
    )

    daily_features = create_daily_prediction_rows(
        feature_df
    )

    target_df = create_next_day_target(
        df
    )

    ml_df = build_ml_dataset(
        daily_features,
        target_df
    )

    return ml_df



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":


    df = load_historical_data()

    feature_df = create_historical_features(
        df
    )

    daily_features = create_daily_prediction_rows(
        feature_df
    )

    target_df = create_next_day_target(
        df
    )

    ml_df = build_ml_dataset(
    daily_features,
    target_df
    )

    print(
        "\nPhase 8 — Prediction target:"
    )

    print(
        target_df[
            "still_trending_tomorrow"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nTotal daily video-region observations:",
        len(target_df)
    )

    usable_targets = target_df[
        "still_trending_tomorrow"
    ].notna().sum()

    unknown_targets = target_df[
        "still_trending_tomorrow"
    ].isna().sum()

    print(
        "Usable ML targets:",
        usable_targets
    )

    print(
        "Unknown targets due to missing next-day collection:",
        unknown_targets
    )

    usable_target_df = target_df.dropna(
        subset=[
            "still_trending_tomorrow"
        ]
    )

    print(
        "\nTarget percentage:"
    )

    print(
        (
            usable_target_df[
                "still_trending_tomorrow"
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )


    print(
    "\nPhase 8 — Historical feature verification:"
    )

    feature_columns = [
    "trending_rank",
    "previous_rank",
    "rank_movement",
    "observations_so_far",
    "hours_trending_so_far",
    "days_trending_so_far",
    "view_count",
    "view_growth",
    "view_velocity",
    "like_velocity",
    "comment_velocity",
    "engagement_rate",
    "engagement_movement",
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count",
    "regions_trending_today",
    "cross_region_today"
]

    print(
        daily_features[
            feature_columns
        ]
        .describe()
        .T
    )

    print(
        "\nDaily prediction rows:",
        len(daily_features)
    )

    print(
        "\nMissing values:"
    )

    print(
        daily_features[
            feature_columns
        ]
        .isna()
        .sum()
    )

    print(
    "\nCategory verification:"
)

    print(
        daily_features[
            "category_name"
        ]
        .value_counts(
            dropna=False
        )
        .head(15)
    )

    print(
        "\nRegional feature verification:"
    )

    print(
        daily_features[
            [
                "regions_trending_today",
                "cross_region_today"
            ]
        ]
        .value_counts()
        .sort_index()
    )

    print(
    "\nFinal ML dataset verification:"
    )

    print(
        "Rows:",
        len(ml_df)
    )

    print(
        "Columns:",
        len(ml_df.columns)
    )

    print(
        "\nTarget counts:"
    )

    print(
        ml_df[
            "still_trending_tomorrow"
        ].value_counts()
    )

    print(
        "\nDate range:"
    )

    print(
        "Start:",
        ml_df["fetch_date"].min()
    )

    print(
        "End:",
        ml_df["fetch_date"].max()
    )

    print(
        "\nDuplicate video-region-day rows:",
        ml_df.duplicated(
            subset=[
                "id",
                "region_code",
                "fetch_date"
            ]
        ).sum()
    )