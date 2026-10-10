import os
import pandas as pd
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

PREFIX = "raw/trending/"


def list_historical_snapshots():
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    paginator = s3.get_paginator("list_objects_v2")

    snapshot_keys = []

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=PREFIX
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith(".json"):
                snapshot_keys.append(key)

    return snapshot_keys

def read_snapshot(s3, key):
    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )

    body = response["Body"].read()

    snapshot = json.loads(body)

    return snapshot

def combine_historical_snapshots(snapshot_keys):
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    all_records = []

    for key in snapshot_keys:
        snapshot = read_snapshot(
            s3,
            key
        )

        schema_version = snapshot.get("metadata", {}).get("schema_version","legacy")

        items = snapshot.get("items", [])

        for item in items:
             item["schema_version"] = schema_version


        all_records.extend(items)

    return all_records

def load_audit_history(s3):
    """
    Load historical pipeline audit records from S3.
    """

    audit_records = []

    paginator = s3.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix="audit/runs/"
    )

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.endswith(".json"):
                continue

            response = s3.get_object(
                Bucket=S3_BUCKET,
                Key=key
            )

            audit_record = json.loads(
                response["Body"].read().decode("utf-8")
            )

            audit_records.append(audit_record)

    audit_df = pd.DataFrame(audit_records)

    if audit_df.empty:
        return audit_df

    audit_df["started_at"] = pd.to_datetime(
        audit_df["started_at"],
        utc=True
    )

    audit_df["ended_at"] = pd.to_datetime(
        audit_df["ended_at"],
        utc=True
    )

    audit_df = audit_df.sort_values(
        "ended_at"
    ).reset_index(drop=True)

    return audit_df


if __name__ == "__main__":


    # ---------------------------------------------------------
    # S3 CLIENT
    # ---------------------------------------------------------

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    # ---------------------------------------------------------
    # LOAD PIPELINE AUDIT HISTORY
    # ---------------------------------------------------------

    audit_df = load_audit_history(s3)

    print(
        "\nHistorical audit records loaded:",
        len(audit_df)
    )

    # ---------------------------------------------------------
    # LOAD HISTORICAL YOUTUBE SNAPSHOTS
    # ---------------------------------------------------------

    keys = list_historical_snapshots()

    print(f"Found {len(keys)} historical snapshots")

    records = combine_historical_snapshots(keys)

    print(f"Found {len(records)} historical video observations")


    df = pd.DataFrame(records)

    print(df.columns.tolist())

    duplicate_mask = df.duplicated(
        subset=["id", "region_code", "fetch_timestamp"],
        keep=False
    )

    duplicate_rows = df[duplicate_mask]

    print(
        f"Duplicate historical observations: {len(duplicate_rows)}"
    )

    video_counts = df["id"].value_counts()

    repeated_videos = video_counts[
        video_counts > 1
    ]

    print(
        f"Videos appearing more than once historically: "
        f"{len(repeated_videos)}"
    )

    print(repeated_videos.head(10))

    df["fetch_timestamp"] = pd.to_datetime(
        df["fetch_timestamp"],
        errors="coerce"
    )

    error = df["fetch_timestamp"].isna().sum()

    if error == 0:
        print("All fetch timestamps converted successfully")
    else:
        print(f"Bad timestamps found: {error}")

        bad_timestamp_rows = df[
            df["fetch_timestamp"].isna()
        ]

        print(bad_timestamp_rows)

    numeric_columns = [
    "trending_rank",
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        error = df[column].isna().sum()

        if error == 0:
            print(f"{column}: converted successfully")
        else:
            print(f"{column}: {error} missing/non-numeric values found")
    
    for column in numeric_columns:
        missing_count = df[column].isna().sum()

        if missing_count == 0:
            print(f"{column}: no missing values")
        else:
            print(f"{column}: {missing_count} missing values")

        if missing_count > 0:
            missing_rows = df[df[column].isna()]

            print(f"\nMissing {column} by fetch_date:")

            print(
                missing_rows["fetch_date"]
                .value_counts() 
                .sort_index()
            )


    channel_enrichment_missing = df[
        df["channel_name"].isna()
        & df["channel_subscriber_count"].isna()
        & df["channel_view_count"].isna()
        & df["channel_video_count"].isna()
    ]

    print(
        f"Rows with missing channel enrichment: "
        f"{len(channel_enrichment_missing)}"
    )

    print("\nSchema version counts:")
    print(df["schema_version"].value_counts(dropna=False))


    expected_columns = [
    "kind",
    "etag",
    "id",
    "snippet",
    "schema_version",
    "contentDetails",
    "statistics",
    "fetch_timestamp",
    "fetch_date",
    "region_code",
    "trending_rank",
    "category_name",
    "channel_name",
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count"
]

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    print(f"Missing expected columns: {missing_columns}")

    df = df[expected_columns]
    print(df.columns.tolist())


    df["view_count"] = df["statistics"].apply(
    lambda x: x.get("viewCount") if isinstance(x, dict) else None
)

    df["view_count"] = pd.to_numeric(
    df["view_count"],
    errors="coerce"
)

    df["like_count"] = df["statistics"].apply(
        lambda x: x.get("likeCount") if isinstance(x, dict) else None
    )

    df["comment_count"] = df["statistics"].apply(
        lambda x: x.get("commentCount") if isinstance(x, dict) else None
    )

    df["like_count"] = pd.to_numeric(
        df["like_count"],
        errors="coerce"
    )

    df["comment_count"] = pd.to_numeric(
        df["comment_count"],
        errors="coerce"
    )

    print("\nView count check:")
    print(
        df[
            ["id", "region_code", "fetch_timestamp", "view_count"]
        ].head(10)
    )

    print("\nMissing view counts:", df["view_count"].isna().sum())


    video_lifecycle = (
    df.groupby(["id", "region_code"])
    .agg(
        first_seen=("fetch_timestamp", "min"),
        last_seen=("fetch_timestamp", "max"),
        trending_observations=("fetch_timestamp", "count"),
        trending_days=("fetch_date", "nunique")
    )
    .reset_index()
)

   
    peak_indices = (
    df.groupby(["id", "region_code"])["trending_rank"]
    .idxmin())

    peak_rows = df.loc[
    peak_indices,
    ["id", "region_code", "fetch_timestamp", "trending_rank"]].copy()

    peak_rows = peak_rows.rename(
    columns={
        "fetch_timestamp": "peak_rank_timestamp",
        "trending_rank": "peak_rank"
    }
)
    
    video_lifecycle = video_lifecycle.merge(
    peak_rows,
    on=["id", "region_code"],
    how="left")

    video_lifecycle["observed_trending_span_hours"] = (
                     video_lifecycle["last_seen"] - video_lifecycle["first_seen"]
                    ).dt.total_seconds() / 3600
    
    video_lifecycle["time_to_peak_hours"] = (
                    video_lifecycle["peak_rank_timestamp"] - video_lifecycle["first_seen"]
                    ).dt.total_seconds() / 3600
    

    df = df.merge(
    video_lifecycle[
        ["id", "region_code", "first_seen"]
    ],
    on=["id", "region_code"],
    how="left"
)

    df["is_new_entry"] = (
        df["fetch_timestamp"] == df["first_seen"]
    )


    snapshot_times = (
        df[
            ["region_code", "fetch_timestamp"]
        ]
        .drop_duplicates()
        .sort_values(
            ["region_code", "fetch_timestamp"]
        )
    )

    regions = df["region_code"].dropna().unique()

    dropout_events = []

    for region in regions:
        region_times = (
            df[
                df["region_code"] == region
            ]["fetch_timestamp"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        first_time = region_times[0]

        seen_before = set(
            df[
                (df["region_code"] == region)
                & (df["fetch_timestamp"] == first_time)
            ]["id"]
        )


        for i in range(len(region_times) - 1):
            previous_time = region_times[i]
            next_time = region_times[i + 1]
            gap_hours = (next_time - previous_time).total_seconds() / 3600


            expected_gap = gap_hours <= 12

            previous_videos = set(
                df[
                    (df["region_code"] == region)
                    & (df["fetch_timestamp"] == previous_time)
                ]["id"]
            )

            next_videos = set(
                df[
                    (df["region_code"] == region)
                    & (df["fetch_timestamp"] == next_time)
                ]["id"]
            )

            dropped_videos = previous_videos - next_videos
            entered_videos = next_videos - previous_videos
            stayed_videos = previous_videos & next_videos
            reentered_videos = entered_videos & seen_before
            first_entries = entered_videos - seen_before
            seen_before.update(next_videos)
         

            dropout_events.append(
                {
                    "region_code": region,
                    "previous_time": previous_time,
                    "next_time": next_time,
                    "gap_hours": gap_hours,
                    "expected_gap" : expected_gap,
                    "previous_count": len(previous_videos),
                    "next_count": len(next_videos),
                    "dropped_count": len(dropped_videos),
                    "entered_count": len(entered_videos),
                    "stayed_count": len(stayed_videos),
                    "dropped_video_ids": list(dropped_videos),
                    "reentry_count": len(reentered_videos),
                    "first_entry_count": len(first_entries),
                    "reentered_video_ids": list(reentered_videos),
                }
            )

    dropout_df = pd.DataFrame(dropout_events)



    df = df.sort_values(
        by=["id", "region_code", "fetch_timestamp"]
    )

    df["previous_rank"] = (
        df.groupby(["id", "region_code"])["trending_rank"]
        .shift(1)
    )

    df["rank_movement"] = (
        df["previous_rank"] - df["trending_rank"]
    )

    df["absolute_rank_movement"] = df["rank_movement"].abs()

# ==============================
# GROWTH & ENGAGEMENT INTELLIGENCE
# ==============================

    df["previous_view_count"] = (
        df.groupby(["id", "region_code"])["view_count"]
        .shift(1)
    )

    df["view_growth"] = (
        df["view_count"] - df["previous_view_count"]
    )

    df["previous_fetch_timestamp"] = (
    df.groupby(["id", "region_code"])["fetch_timestamp"]
    .shift(1)
    )

    df["elapsed_hours"] = (
        df["fetch_timestamp"] - df["previous_fetch_timestamp"]
    ).dt.total_seconds() / 3600

    df["view_velocity"] = (
        df["view_growth"] / df["elapsed_hours"]
    )

    df.loc[df["elapsed_hours"] <= 0, "view_velocity"] = pd.NA

    df["previous_view_velocity"] = (
    df.groupby(["id", "region_code"])["view_velocity"]
    .shift(1)
    )

    df["view_velocity_change"] = (
    df["view_velocity"] - df["previous_view_velocity"]
    )

    df["view_growth_trend"] = "stable"

    df.loc[
        df["view_velocity_change"] > 0,
        "view_growth_trend"
    ] = "accelerating"

    df.loc[
        df["view_velocity_change"] < 0,
        "view_growth_trend"
    ] = "decelerating"

    df.loc[
        df["view_velocity_change"].isna(),
        "view_growth_trend"
    ] = pd.NA

    df["previous_like_count"] = (
    df.groupby(["id", "region_code"])["like_count"]
    .shift(1)
    )

    df["like_growth"] = (
        df["like_count"] - df["previous_like_count"]
    )

    df["like_velocity"] = (
        df["like_growth"] / df["elapsed_hours"]
    )

    df.loc[df["elapsed_hours"] <= 0, "like_velocity"] = pd.NA


    df["previous_comment_count"] = (
        df.groupby(["id", "region_code"])["comment_count"]
        .shift(1)
    )

    df["comment_growth"] = (
        df["comment_count"] - df["previous_comment_count"]
    )

    df["comment_velocity"] = (
        df["comment_growth"] / df["elapsed_hours"]
    )

    df.loc[df["elapsed_hours"] <= 0, "comment_velocity"] = pd.NA

    df["engagement_rate"] = (
    (df["like_count"] + df["comment_count"])
    / df["view_count"]
    )

    df.loc[df["view_count"] <= 0, "engagement_rate"] = pd.NA

    df["previous_engagement_rate"] = (
        df.groupby(["id", "region_code"])["engagement_rate"]
        .shift(1)
    )

    df["engagement_rate_movement"] = (
        df["engagement_rate"]
        - df["previous_engagement_rate"]
    )

    

    rank_volatility = (
    df.groupby(["id", "region_code"])
    .agg(
        average_rank_movement=(
            "absolute_rank_movement",
            "mean"
        ),
        rank_changes=(
            "rank_movement",
            "count"
        )
    )
    .reset_index()
)
    
    rank_volatility = rank_volatility[
    rank_volatility["rank_changes"] >= 3
]
    rank_volatility = rank_volatility.sort_values(
    "average_rank_movement",
    ascending=False
)

    fastest_rising = (
    df[df["rank_movement"] > 0]
    .sort_values("rank_movement", ascending=False)
)
    

    fastest_falling = (
        df[df["rank_movement"] < 0]
        .sort_values("rank_movement", ascending=True)
    )
    print("\nFastest rising videos:")
    print(
        fastest_rising[
            [
                "id",
                "region_code",
                "fetch_timestamp",
                "previous_rank",
                "trending_rank",
                "rank_movement"
            ]
        ].head(10)
    )

    print("\nFastest falling videos:")
    print(
        fastest_falling[
            [
                "id",
                "region_code",
                "fetch_timestamp",
                "previous_rank",
                "trending_rank",
                "rank_movement"
            ]
        ].head(10)
    )


    video_id = "kw0YhiqOPFg"

    video_history = (
        df[
            df["id"] == video_id
        ]
        .sort_values(
            ["region_code", "fetch_timestamp"]
        )
    )

    print(
    video_history[
        [
            "id",
            "region_code",
            "fetch_timestamp",
            "trending_rank",
            "previous_rank",
            "rank_movement"
        ]
    ]
)
    video_dropouts = dropout_df[
    dropout_df["dropped_video_ids"].apply(
        lambda ids: video_id in ids
    )
]

    print("\nDropout events for selected video:")

    print(
        video_dropouts[
            [
                "region_code",
                "previous_time",
                "next_time",
                "gap_hours",
                "expected_gap"
            ]
        ]
    )

    video_reentries = dropout_df[
    dropout_df["reentered_video_ids"].apply(
        lambda ids: video_id in ids
    )
]

    print("\nRe-entry events for selected video:")

    print(
        video_reentries[
            [
                "region_code",
                "previous_time",
                "next_time",
                "gap_hours",
                "expected_gap"
            ]
        ]
    )

    print("\nMost volatile videos:")

    print(
        rank_volatility[
            [
                "id",
                "region_code",
                "average_rank_movement",
                "rank_changes"
            ]
        ].head(10)
    )

    print("\nView growth check:")

    print(
        df[
            [
                "id",
                "region_code",
                "fetch_timestamp",
                "previous_view_count",
                "view_count",
                "view_growth"
            ]
        ].dropna(subset=["previous_view_count"]).head(10)
    )

    print("\nView velocity check:")

    print(
        df[
            [
                "id",
                "region_code",
                "fetch_timestamp",
                "previous_fetch_timestamp",
                "elapsed_hours",
                "view_growth",
                "view_velocity"
            ]
        ]
        .dropna(subset=["previous_fetch_timestamp"])
        .head(10)
    )

    print("\nEngagement growth/velocity check:")

    print(
        df[
            [
                "id",
                "region_code",
                "elapsed_hours",
                "like_count",
                "like_growth",
                "like_velocity",
                "comment_count",
                "comment_growth",
                "comment_velocity"
            ]
        ]
        .dropna(subset=["elapsed_hours"])
        .head(10)
    )

    print("\nEngagement rate movement check:")

    print(
        df[
            [
                "id",
                "region_code",
                "view_count",
                "like_count",
                "comment_count",
                "previous_engagement_rate",
                "engagement_rate",
                "engagement_rate_movement"
            ]
        ]
        .dropna(subset=["previous_engagement_rate"])
        .head(10)
    )


    print("\nView growth acceleration/deceleration check:")

    print(
        df[
            [
                "id",
                "region_code",
                "elapsed_hours",
                "previous_view_velocity",
                "view_velocity",
                "view_velocity_change",
                "view_growth_trend"
            ]
        ]
        .dropna(subset=["previous_view_velocity"])
        .head(10)
    )

# ---------------------------------------------------------
# CHANNEL INTELLIGENCE
# ---------------------------------------------------------

    channel_frequency = (
        df.groupby(["channel_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique")
        )
        .reset_index()
    )

    channel_frequency = channel_frequency.sort_values(
            by=[
                "region_code",
                "unique_trending_videos",
                "trending_observations"
            ],
            ascending=[True, False, False]
        )


# ---------------------------------------------------------
# CHANNEL MOMENTUM
# ---------------------------------------------------------

    channel_daily_presence = (
            df.groupby(["channel_name", "region_code", "fetch_date"])
            .agg(
                trending_observations=("id", "count"),
                unique_trending_videos=("id", "nunique")
            )
            .reset_index()
        )

    channel_daily_presence["fetch_date"] = pd.to_datetime(
            channel_daily_presence["fetch_date"]
        )


# Create a complete daily timeline for each channel + region
    complete_channel_days = []

    for (channel, region), group in channel_daily_presence.groupby(
            ["channel_name", "region_code"]
        ):
                group = group.sort_values("fetch_date").set_index("fetch_date")

                full_date_range = pd.date_range(
                    start=group.index.min(),
                    end=group.index.max(),
                    freq="D"
                )

        # THESE LINES MUST BE INSIDE THE LOOP
                group = group.reindex(full_date_range)

                group["channel_name"] = channel
                group["region_code"] = region

                group["trending_observations"] = (
                    group["trending_observations"].fillna(0)
                )

                group["unique_trending_videos"] = (
                    group["unique_trending_videos"].fillna(0)
                )

                group.index.name = "fetch_date"

                complete_channel_days.append(group.reset_index())


    # This happens AFTER all channel/region groups are processed
    channel_daily_presence = pd.concat(
                complete_channel_days,
                ignore_index=True
            )

    channel_daily_presence = channel_daily_presence.sort_values(
                ["channel_name", "region_code", "fetch_date"]
            )


            # Previous day's trending presence
    channel_daily_presence["previous_trending_observations"] = (
                channel_daily_presence
                .groupby(["channel_name", "region_code"])["trending_observations"]
                .shift(1)
            )


            # Current presence - previous presence
    channel_daily_presence["channel_momentum"] = (
                channel_daily_presence["trending_observations"]
                - channel_daily_presence["previous_trending_observations"]
            )


        # Momentum label
    channel_daily_presence["momentum_trend"] = "stable"

    channel_daily_presence.loc[
            channel_daily_presence["channel_momentum"] > 0,
            "momentum_trend"
        ] = "positive"

    channel_daily_presence.loc[
            channel_daily_presence["channel_momentum"] < 0,
            "momentum_trend"
        ] = "negative"

    channel_daily_presence.loc[
            channel_daily_presence["channel_momentum"].isna(),
            "momentum_trend"
        ] = pd.NA


# ---------------------------------------------------------
# CHANNEL DOMINANCE
# ---------------------------------------------------------

# Total trending observations for each region and day
    daily_region_totals = (
        df.groupby(["region_code", "fetch_date"])
        .size()
        .reset_index(name="total_region_observations")
    )


# Trending observations belonging to each channel
    channel_daily_dominance = (
        df.groupby(["channel_name", "region_code", "fetch_date"])
        .size()
        .reset_index(name="channel_observations")
    )


# Attach the region/day total
    channel_daily_dominance = channel_daily_dominance.merge(
        daily_region_totals,
        on=["region_code", "fetch_date"],
        how="left"
    )


# Percentage of the region's trending observations
# belonging to this channel
    channel_daily_dominance["dominance_pct"] = (
        channel_daily_dominance["channel_observations"]
        / channel_daily_dominance["total_region_observations"]
    ) * 100


# ---------------------------------------------------------
# LARGE VS SMALLER CHANNELS
# ---------------------------------------------------------

    channel_size_data = (
        df[
            [
                "channel_name",
                "channel_subscriber_count"
            ]
        ]
        .dropna()
        .drop_duplicates(subset=["channel_name"])
    )

    q1 = channel_size_data["channel_subscriber_count"].quantile(0.25)
    q2 = channel_size_data["channel_subscriber_count"].quantile(0.50)
    q3 = channel_size_data["channel_subscriber_count"].quantile(0.75)


    def classify_channel_size(subscribers):
        if subscribers <= q1:
            return "small"
        elif subscribers <= q2:
            return "medium"
        elif subscribers <= q3:
            return "large"
        else:
            return "very_large"


    channel_size_data["channel_size"] = (
        channel_size_data["channel_subscriber_count"]
        .apply(classify_channel_size)
    )

    # Attach channel size to historical observations
    df = df.merge(
        channel_size_data[
            ["channel_name", "channel_size"]
        ],
        on="channel_name",
        how="left"
    )


# Compare trending performance by channel size
    channel_size_performance = (
        df.dropna(subset=["channel_size"])
        .groupby("channel_size", observed=True)
        .agg(
            unique_channels=("channel_name", "nunique"),
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            average_trending_rank=("trending_rank", "mean"),
            median_trending_rank=("trending_rank", "median")
        )
        .reset_index()
    )

# ---------------------------------------------------------
# SUBSCRIBER COUNT VS TRENDING PERFORMANCE
# ---------------------------------------------------------

    channel_performance = (
        df.dropna(
            subset=[
                "channel_name",
                "channel_subscriber_count",
                "trending_rank"
            ]
        )
        .groupby("channel_name")
        .agg(
            subscriber_count=("channel_subscriber_count", "last"),
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            average_trending_rank=("trending_rank", "mean"),
            best_trending_rank=("trending_rank", "min")
        )
        .reset_index()
    )


    
    # Correlation between subscriber count and trending performance
    subscriber_correlations = (
        channel_performance[
            [
                "subscriber_count",
                "unique_trending_videos",
                "trending_observations",
                "average_trending_rank",
                "best_trending_rank"
            ]
        ]
        .corr(method="spearman")
    )


    # ---------------------------------------------------------
# HISTORICAL COMPETITOR / CHANNEL COMPARISON
# ---------------------------------------------------------

    def compare_channels(df, channel_a, channel_b):

        comparison_df = df[
            df["channel_name"].isin([channel_a, channel_b])
        ].copy()

        channel_comparison = (
            comparison_df
            .groupby("channel_name")
            .agg(
                subscriber_count=("channel_subscriber_count", "last"),
                unique_trending_videos=("id", "nunique"),
                trending_observations=("id", "count"),
                active_trending_days=("fetch_date", "nunique"),
                average_trending_rank=("trending_rank", "mean"),
                best_trending_rank=("trending_rank", "min"),
                average_view_velocity=("view_velocity", "mean"),
                average_engagement_rate=("engagement_rate", "mean")
            )
            .reset_index()
        )

        return channel_comparison


    # ---------------------------------------------------------
# CATEGORY TRENDS
# ---------------------------------------------------------

    category_daily_trends = (
        df.dropna(subset=["category_name"])
        .groupby(
            ["category_name", "region_code", "fetch_date"]
        )
        .agg(
            trending_observations=("id", "count"),
            unique_trending_videos=("id", "nunique")
        )
        .reset_index()
    )

    category_daily_trends["fetch_date"] = pd.to_datetime(
        category_daily_trends["fetch_date"]
    )

    complete_category_days = []

    for (category, region), group in category_daily_trends.groupby(
        ["category_name", "region_code"]
    ):
        group = group.sort_values("fetch_date").set_index("fetch_date")

        full_date_range = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq="D"
        )

        group = group.reindex(full_date_range)

        group["category_name"] = category
        group["region_code"] = region

        group["trending_observations"] = (
            group["trending_observations"].fillna(0)
        )

        group["unique_trending_videos"] = (
            group["unique_trending_videos"].fillna(0)
        )

        group.index.name = "fetch_date"

        complete_category_days.append(
            group.reset_index()
        )

    category_daily_trends = pd.concat(
        complete_category_days,
        ignore_index=True
    )

    category_daily_trends = category_daily_trends.sort_values(
        ["category_name", "region_code", "fetch_date"]
    )

    category_daily_trends["previous_trending_observations"] = (
        category_daily_trends
        .groupby(
            ["category_name", "region_code"]
        )["trending_observations"]
        .shift(1)
    )

    category_daily_trends["observation_change"] = (
        category_daily_trends["trending_observations"]
        - category_daily_trends["previous_trending_observations"]
    )

# ---------------------------------------------------------
# CATEGORY TRENDING FREQUENCY
# ---------------------------------------------------------

    category_frequency = (
        df.dropna(subset=["category_name"])
        .groupby(["category_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique")
        )
        .reset_index()
    )

    category_frequency = category_frequency.sort_values(
        by=[
            "region_code",
            "trending_observations"
        ],
        ascending=[True, False]
    )

# ---------------------------------------------------------
# CATEGORY DOMINANCE OVER TIME
# ---------------------------------------------------------

    daily_region_category_totals = (
        df.dropna(subset=["category_name"])
        .groupby(
            ["category_name", "region_code", "fetch_date"]
        )
        .size()
        .reset_index(name="category_observations")
    )

    daily_region_totals = (
        df.groupby(
            ["region_code", "fetch_date"]
        )
        .size()
        .reset_index(name="total_region_observations")
    )

    category_dominance = daily_region_category_totals.merge(
        daily_region_totals,
        on=["region_code", "fetch_date"],
        how="left"
    )

    category_dominance["dominance_pct"] = (
        category_dominance["category_observations"]
        / category_dominance["total_region_observations"]
        * 100
    )

    category_dominance = category_dominance.sort_values(
        ["region_code", "fetch_date", "dominance_pct"],
        ascending=[True, True, False]
    )

# ---------------------------------------------------------
# CATEGORY COMPARISON BETWEEN REGIONS
# ---------------------------------------------------------

    category_region_comparison = (
        df.dropna(subset=["category_name"])
        .groupby(["category_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique"),
            average_trending_rank=("trending_rank", "mean"),
            best_trending_rank=("trending_rank", "min")
        )
        .reset_index()
    )

    # Total observations collected for each region
    region_observation_totals = (
        df.groupby("region_code")
        .size()
        .reset_index(name="total_region_observations")
    )

    category_region_comparison = category_region_comparison.merge(
        region_observation_totals,
        on="region_code",
        how="left"
    )

    # Historical share of the region represented by each category
    category_region_comparison["historical_dominance_pct"] = (
        category_region_comparison["trending_observations"]
        / category_region_comparison["total_region_observations"]
        * 100
    )

    # ---------------------------------------------------------
# CATEGORY RANK / GROWTH PATTERNS
# ---------------------------------------------------------

    category_performance = (
        df.dropna(subset=["category_name"])
        .groupby(["category_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            average_trending_rank=("trending_rank", "mean"),
            median_trending_rank=("trending_rank", "median"),
            best_trending_rank=("trending_rank", "min"),
            average_view_growth=("view_growth", "mean"),
            average_view_velocity=("view_velocity", "mean"),
            average_like_velocity=("like_velocity", "mean"),
            average_comment_velocity=("comment_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean")
        )
        .reset_index()
    )

    # Minimum number of unique videos required
    # for a category to be treated as sufficiently represented
    MIN_CATEGORY_VIDEOS = 20

    category_performance["sufficient_sample"] = (
        category_performance["unique_trending_videos"]
        >= MIN_CATEGORY_VIDEOS
    )

    
    category_performance_reliable = (
        category_performance[
            category_performance["sufficient_sample"]
        ]
        .copy()
    )

# ---------------------------------------------------------
# REGIONAL COMPARISON
# ---------------------------------------------------------

    regional_comparison = (
        df.groupby("region_code")
        .agg(
            historical_observations=("id", "count"),
            unique_trending_videos=("id", "nunique"),
            unique_channels=("channel_name", "nunique"),
            active_collection_days=("fetch_date", "nunique"),
            average_trending_rank=("trending_rank", "mean"),
            median_trending_rank=("trending_rank", "median"),
            average_view_velocity=("view_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean")
        )
        .reset_index()
    )

# ---------------------------------------------------------
# RANK COMPARISON BETWEEN COUNTRIES
# ---------------------------------------------------------

    df["video_title"] = df["snippet"].apply(
        lambda x: x.get("title") if isinstance(x, dict) else None
    )

    video_region_rank = (
        df.groupby(["id", "region_code"])
        .agg(
            video_title=("video_title", "last"),
            average_rank=("trending_rank", "mean"),
            best_rank=("trending_rank", "min"),
            trending_observations=("id", "count")
        )
        .reset_index()
    )

    # Keep videos observed in at least two regions
    multi_region_video_ids = (
        video_region_rank
        .groupby("id")["region_code"]
        .nunique()
    )

    multi_region_video_ids = multi_region_video_ids[
        multi_region_video_ids >= 2
    ].index

    cross_country_rank = video_region_rank[
        video_region_rank["id"].isin(multi_region_video_ids)
    ].copy()


# ---------------------------------------------------------
# MULTI-REGION TRENDING VIDEOS
# ---------------------------------------------------------

    multi_region_videos = (
        df.groupby("id")
        .agg(
            video_title=("video_title", "last"),
            regions_trending=("region_code", "nunique"),
            region_list=(
                "region_code",
                lambda x: sorted(x.dropna().unique().tolist())
            ),
            total_trending_observations=("id", "count"),
            first_seen=("fetch_timestamp", "min"),
            last_seen=("fetch_timestamp", "max")
        )
        .reset_index()
    )

    multi_region_videos = (
        multi_region_videos[
            multi_region_videos["regions_trending"] >= 2
        ]
        .sort_values(
            ["regions_trending", "total_trending_observations"],
            ascending=[False, False]
        )
    )
# ---------------------------------------------------------
# REGION-SPECIFIC TRENDING VIDEOS
# ---------------------------------------------------------

    video_region_summary = (
        df.groupby("id")
        .agg(
            video_title=("video_title", "last"),
            regions_trending=("region_code", "nunique"),
            region_list=(
                "region_code",
                lambda x: sorted(x.dropna().unique().tolist())
            ),
            total_trending_observations=("id", "count"),
            first_seen=("fetch_timestamp", "min"),
            last_seen=("fetch_timestamp", "max")
        )
        .reset_index()
    )

    multi_region_videos = (
        video_region_summary[
            video_region_summary["regions_trending"] >= 2
        ]
        .copy()
    )

    region_specific_videos = (
        video_region_summary[
            video_region_summary["regions_trending"] == 1
        ]
        .copy()
    )

    region_specific_videos["region_code"] = (
        region_specific_videos["region_list"]
        .apply(lambda x: x[0] if len(x) == 1 else None)
    )


# ---------------------------------------------------------
# CROSS-REGION VS REGION-SPECIFIC PERFORMANCE
# ---------------------------------------------------------

    video_region_reach = (
        df.groupby("id")["region_code"]
        .nunique()
        .rename("regions_trending")
        .reset_index()
    )

    global_regional_performance = df.merge(
        video_region_reach,
        on="id",
        how="left"
    )

    global_regional_performance["reach_type"] = "region_specific"

    global_regional_performance.loc[
        global_regional_performance["regions_trending"] >= 2,
        "reach_type"
    ] = "cross_region"

    reach_performance_summary = (
        global_regional_performance
        .groupby("reach_type")
        .agg(
            unique_videos=("id", "nunique"),
            average_regions_trending=("regions_trending", "mean"),
            average_trending_rank=("trending_rank", "mean"),
            median_trending_rank=("trending_rank", "median"),
            average_view_velocity=("view_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean")
        )
        .reset_index()
    )

# ---------------------------------------------------------
# CATEGORY POPULARITY ACROSS REGIONS
# ---------------------------------------------------------

    category_popularity_across_regions = (
        category_region_comparison[
            [
                "category_name",
                "region_code",
                "unique_trending_videos",
                "trending_observations",
                "historical_dominance_pct"
            ]
        ]
        .copy()
    )

    category_popularity_across_regions["popularity_rank"] = (
        category_popularity_across_regions
        .groupby("region_code")["historical_dominance_pct"]
        .rank(method="dense", ascending=False)
    )

    category_popularity_across_regions = (
        category_popularity_across_regions
        .sort_values(
            ["region_code", "popularity_rank"]
        )
    )


# ---------------------------------------------------------
# CHANNEL DOMINANCE ACROSS REGIONS
# ---------------------------------------------------------

    channel_region_dominance = (
        df.dropna(subset=["channel_name"])
        .groupby(["channel_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique")
        )
        .reset_index()
    )

    region_totals = (
        df.groupby("region_code")
        .size()
        .reset_index(name="total_region_observations")
    )

    channel_region_dominance = channel_region_dominance.merge(
        region_totals,
        on="region_code",
        how="left"
    )

    channel_region_dominance["dominance_pct"] = (
        channel_region_dominance["trending_observations"]
        / channel_region_dominance["total_region_observations"]
        * 100
    )

    channel_region_dominance["dominance_rank"] = (
        channel_region_dominance
        .groupby("region_code")["dominance_pct"]
        .rank(method="dense", ascending=False)
    )

# ---------------------------------------------------------
# DATA QUALITY — SNAPSHOT RECORD COUNTS
# ---------------------------------------------------------

    snapshot_record_counts = (
        df.groupby(
            ["region_code", "fetch_timestamp"]
        )
        .size()
        .reset_index(name="records_received")
    )

    print("\nSnapshot record count summary:")

    print(
        snapshot_record_counts
        .groupby("region_code")["records_received"]
        .agg(
            snapshots="count",
            min_records="min",
            max_records="max",
            average_records="mean",
            median_records="median"
        )
        .reset_index()
        .to_string(index=False)
    )

    print("\nLowest-record snapshots:")

    print(
        snapshot_record_counts
        .sort_values("records_received")
        .head(15)
        .to_string(index=False)
    )

# ---------------------------------------------------------
# EXPECTED VS RECEIVED RECORDS
# ---------------------------------------------------------

    EXPECTED_RECORDS_PER_SNAPSHOT = 100

    snapshot_record_counts["expected_records"] = EXPECTED_RECORDS_PER_SNAPSHOT

    snapshot_record_counts["record_difference"] = (
        snapshot_record_counts["records_received"]
        - snapshot_record_counts["expected_records"]
    )

    snapshot_record_counts["completion_pct"] = (
        snapshot_record_counts["records_received"]
        / snapshot_record_counts["expected_records"]
        * 100
    )

    snapshot_record_counts["record_status"] = "complete"

    snapshot_record_counts.loc[
        snapshot_record_counts["records_received"] < EXPECTED_RECORDS_PER_SNAPSHOT,
        "record_status"
    ] = "below_expected"

# ---------------------------------------------------------
# DATA QUALITY — MISSING FIELD COUNTS
# ---------------------------------------------------------

    quality_fields = [
        "id",
        "fetch_timestamp",
        "fetch_date",
        "region_code",
        "trending_rank",
        "category_name",
        "channel_name",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
        "view_count",
        "like_count",
        "comment_count"
    ]

    missing_field_summary = pd.DataFrame({
        "field": quality_fields,
        "missing_count": [
            df[field].isna().sum()
            for field in quality_fields
        ]
    })

    missing_field_summary["missing_pct"] = (
        missing_field_summary["missing_count"]
        / len(df)
        * 100
    )

    missing_field_summary = (
        missing_field_summary
        .sort_values(
            "missing_count",
            ascending=False
        )
        .reset_index(drop=True)
    )

# ---------------------------------------------------------
# DATA QUALITY — DUPLICATE COUNTS
# ---------------------------------------------------------

    duplicate_observation_mask = df.duplicated(
        subset=[
            "id",
            "region_code",
            "fetch_timestamp"
        ],
        keep=False
    )

    duplicate_observation_count = duplicate_observation_mask.sum()

    duplicate_observation_pct = (
        duplicate_observation_count
        / len(df)
        * 100
    )

    duplicate_rows = df[
        duplicate_observation_mask
    ].copy()


# ---------------------------------------------------------
# DATA QUALITY — REGION COMPLETENESS
# ---------------------------------------------------------

    EXPECTED_REGIONS = {"CA", "GB", "IN", "US"}

    observed_regions = set(
        df["region_code"]
        .dropna()
        .unique()
    )

    missing_regions_overall = (
        EXPECTED_REGIONS - observed_regions
    )

    unexpected_regions = (
        observed_regions - EXPECTED_REGIONS
    )

    daily_region_presence = (
        df.groupby("fetch_date")["region_code"]
        .agg(
            lambda x: set(x.dropna().unique())
        )
        .reset_index(name="regions_present")
    )

    daily_region_presence["region_count"] = (
        daily_region_presence["regions_present"]
        .apply(len)
    )

    daily_region_presence["missing_regions"] = (
        daily_region_presence["regions_present"]
        .apply(
            lambda regions:
            sorted(EXPECTED_REGIONS - regions)
        )
    )

    daily_region_presence["all_regions_present"] = (
        daily_region_presence["missing_regions"]
        .apply(lambda x: len(x) == 0)
    )

    incomplete_region_days = (
        daily_region_presence[
            ~daily_region_presence["all_regions_present"]
        ]
        .copy()
    )

# ---------------------------------------------------------
# DATA QUALITY — COLLECTION WINDOW COMPLETENESS
# ---------------------------------------------------------

    EXPECTED_INTERVAL_HOURS = 6
    GAP_TOLERANCE_HOURS = 9

    collection_window_timing = (
        snapshot_record_counts[
            [
                "region_code",
                "fetch_timestamp"
            ]
        ]
        .sort_values(
            ["region_code", "fetch_timestamp"]
        )
        .copy()
    )

    collection_window_timing["previous_snapshot"] = (
        collection_window_timing
        .groupby("region_code")["fetch_timestamp"]
        .shift(1)
    )

    collection_window_timing["gap_hours"] = (
        collection_window_timing["fetch_timestamp"]
        - collection_window_timing["previous_snapshot"]
    ).dt.total_seconds() / 3600

    collection_gaps = collection_window_timing[
        collection_window_timing["gap_hours"]
        > GAP_TOLERANCE_HOURS
    ].copy()

    collection_gaps["estimated_missing_windows"] = (
        (
            collection_gaps["gap_hours"]
            / EXPECTED_INTERVAL_HOURS
        )
        .round()
        .astype(int)
        - 1
    )

    collection_gaps["estimated_missing_windows"] = (
        collection_gaps["estimated_missing_windows"]
        .clip(lower=1)
    )

    collection_window_summary = (
        collection_gaps
        .groupby("region_code")
        .agg(
            detected_large_gaps=(
                "gap_hours",
                "count"
            ),
            estimated_missing_windows=(
                "estimated_missing_windows",
                "sum"
            ),
            largest_gap_hours=(
                "gap_hours",
                "max"
            )
        )
        .reset_index()
    )

# ---------------------------------------------------------
# DATA QUALITY — MISSING REGION/WINDOW SNAPSHOTS
# ---------------------------------------------------------

    missing_window_summary = (
        collection_gaps[
            [
                "region_code",
                "previous_snapshot",
                "fetch_timestamp",
                "gap_hours",
                "estimated_missing_windows"
            ]
        ]
        .copy()
    )

    missing_window_summary["gap_start"] = (
        missing_window_summary["previous_snapshot"]
    )

    missing_window_summary["gap_end"] = (
        missing_window_summary["fetch_timestamp"]
    )

    missing_window_summary["missing_region"] = (
        missing_window_summary["region_code"]
    )

    missing_window_summary = (
        missing_window_summary[
            [
                "missing_region",
                "gap_start",
                "gap_end",
                "gap_hours",
                "estimated_missing_windows"
            ]
        ]
        .sort_values(
            ["gap_start", "missing_region"]
        )
        .reset_index(drop=True)
    )

# ---------------------------------------------------------
# DATA QUALITY — CATEGORY ENRICHMENT COMPLETENESS
# ---------------------------------------------------------

    df["category_enriched"] = (
        df["category_name"].notna()
    )

    category_enrichment_summary = (
        df.groupby("region_code")
        .agg(
            total_observations=("id", "count"),
            enriched_observations=(
                "category_enriched",
                "sum"
            )
        )
        .reset_index()
    )

    category_enrichment_summary[
        "missing_category"
    ] = (
        category_enrichment_summary["total_observations"]
        - category_enrichment_summary["enriched_observations"]
    )

    category_enrichment_summary[
        "category_completeness_pct"
    ] = (
        category_enrichment_summary["enriched_observations"]
        / category_enrichment_summary["total_observations"]
        * 100
    )

    category_missing_by_schema = (
        df.groupby("schema_version")
        .agg(
            total_observations=("id", "count"),
            missing_category=(
                "category_name",
                lambda x: x.isna().sum()
            )
        )
        .reset_index()
    )

    category_missing_by_schema[
        "missing_pct"
    ] = (
        category_missing_by_schema["missing_category"]
        / category_missing_by_schema["total_observations"]
        * 100
    )


# ---------------------------------------------------------
# DATA QUALITY — CHANNEL ENRICHMENT COMPLETENESS
# ---------------------------------------------------------

    channel_enrichment_fields = [
        "channel_name",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count"
    ]

    df["channel_enriched"] = (
        df[channel_enrichment_fields]
        .notna()
        .all(axis=1)
    )

    channel_enrichment_summary = (
        df.groupby("region_code")
        .agg(
            total_observations=("id", "count"),
            enriched_observations=(
                "channel_enriched",
                "sum"
            )
        )
        .reset_index()
    )

    channel_enrichment_summary[
        "missing_channel_enrichment"
    ] = (
        channel_enrichment_summary["total_observations"]
        - channel_enrichment_summary["enriched_observations"]
    )

    channel_enrichment_summary[
        "channel_completeness_pct"
    ] = (
        channel_enrichment_summary["enriched_observations"]
        / channel_enrichment_summary["total_observations"]
        * 100
    )

    channel_missing_by_schema = (
        df.groupby("schema_version")
        .agg(
            total_observations=("id", "count"),
            missing_channel_enrichment=(
                "channel_enriched",
                lambda x: (~x).sum()
            )
        )
        .reset_index()
    )

    channel_missing_by_schema[
        "missing_pct"
    ] = (
        channel_missing_by_schema[
            "missing_channel_enrichment"
        ]
        / channel_missing_by_schema[
            "total_observations"
        ]
        * 100
    )

# ---------------------------------------------------------
# PIPELINE HEALTH — LAST SUCCESSFUL RUN
# ---------------------------------------------------------

    successful_runs = audit_df[
        audit_df["status"] == "SUCCESS"
    ].copy()

    if not successful_runs.empty:

        last_successful_run = (
            successful_runs
            .sort_values("ended_at")
            .iloc[-1]
        )

        last_successful_run_summary = {
            "run_id": last_successful_run["run_id"],
            "started_at": last_successful_run["started_at"],
            "ended_at": last_successful_run["ended_at"],
            "records_fetched": last_successful_run["records_fetched"],
            "records_written": last_successful_run["records_written"],
            "retry_count": last_successful_run["retry_count"]
        }

    else:
        last_successful_run_summary = None

# ---------------------------------------------------------
# PIPELINE HEALTH — FRESHNESS
# ---------------------------------------------------------

    if last_successful_run_summary is not None:

        current_time = pd.Timestamp.now(tz="UTC")

        last_success_time = (
            last_successful_run["ended_at"]
        )

        freshness_hours = (
            current_time - last_success_time
        ).total_seconds() / 3600

        if freshness_hours <= 9:
            freshness_status = "FRESH"
        else:
            freshness_status = "STALE"

        freshness_summary = {
            "checked_at": current_time,
            "last_successful_run": last_success_time,
            "freshness_hours": freshness_hours,
            "freshness_status": freshness_status
        }

    else:
        freshness_summary = None


# ---------------------------------------------------------
# PIPELINE HEALTH — SUCCESS / FAILURE
# ---------------------------------------------------------

    run_status_summary = (
        audit_df["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="run_count")
    )

    total_audit_runs = len(audit_df)

    successful_run_count = (
        audit_df["status"] == "SUCCESS"
    ).sum()

    failed_run_count = (
        audit_df["status"] == "FAILED"
    ).sum()

    success_rate_pct = (
        successful_run_count
        / total_audit_runs
        * 100
        if total_audit_runs > 0
        else 0
    )

    failure_rate_pct = (
        failed_run_count
        / total_audit_runs
        * 100
        if total_audit_runs > 0
        else 0
    )


# ---------------------------------------------------------
# PIPELINE HEALTH — HISTORICAL HEALTH
# ---------------------------------------------------------

    audit_df["run_date"] = (
        audit_df["ended_at"].dt.date
    )

    historical_health = (
        audit_df
        .groupby("run_date")
        .agg(
            total_runs=("run_id", "count"),
            successful_runs=(
                "status",
                lambda x: (x == "SUCCESS").sum()
            ),
            failed_runs=(
                "status",
                lambda x: (x == "FAILED").sum()
            )
        )
        .reset_index()
    )

    historical_health["success_rate_pct"] = (
        historical_health["successful_runs"]
        / historical_health["total_runs"]
        * 100
    )

    historical_health["health_status"] = "HEALTHY"

    historical_health.loc[
        historical_health["failed_runs"] > 0,
        "health_status"
    ] = "DEGRADED"


    # ---------------------------------------------------------
# PIPELINE HEALTH — RETRY COUNT HISTORY
# ---------------------------------------------------------

    audit_df["retry_count"] = pd.to_numeric(
        audit_df["retry_count"],
        errors="coerce"
    ).fillna(0)

    total_retries = audit_df["retry_count"].sum()

    runs_with_retries = (
        audit_df["retry_count"] > 0
    ).sum()

    max_retries_single_run = (
        audit_df["retry_count"].max()
    )

    retry_history = audit_df[
        audit_df["retry_count"] > 0
    ][
        [
            "run_id",
            "ended_at",
            "status",
            "retry_count"
        ]
    ].copy()

    retry_history = retry_history.sort_values(
        "ended_at"
    )

# ---------------------------------------------------------
# PIPELINE HEALTH — RECORDS FETCHED / WRITTEN HISTORY
# ---------------------------------------------------------

    audit_df["records_fetched"] = pd.to_numeric(
        audit_df["records_fetched"],
        errors="coerce"
    ).fillna(0)

    audit_df["records_written"] = pd.to_numeric(
        audit_df["records_written"],
        errors="coerce"
    ).fillna(0)

    audit_df["record_write_difference"] = (
        audit_df["records_fetched"]
        - audit_df["records_written"]
    )

    records_history = audit_df[
        [
            "run_id",
            "ended_at",
            "status",
            "records_fetched",
            "records_written",
            "record_write_difference"
        ]
    ].copy()

    total_records_fetched = (
        audit_df["records_fetched"].sum()
    )

    total_records_written = (
        audit_df["records_written"].sum()
    )

    runs_with_record_mismatch = (
        audit_df["record_write_difference"] != 0
    ).sum()

    record_mismatch_runs = audit_df[
        audit_df["record_write_difference"] != 0
    ][
        [
            "run_id",
            "ended_at",
            "status",
            "records_fetched",
            "records_written",
            "record_write_difference"
        ]
    ].copy()


# ---------------------------------------------------------
# PIPELINE HEALTH — FAILED RUN / ERROR SUMMARY
# ---------------------------------------------------------

    failed_runs = audit_df[
        audit_df["status"] == "FAILED"
    ].copy()

    failed_run_count = len(failed_runs)

    error_type_summary = (
        failed_runs["error_type"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="failure_count")
    )

    failed_run_summary = failed_runs[
        [
            "run_id",
            "ended_at",
            "records_fetched",
            "records_written",
            "retry_count",
            "error_type"
        ]
    ].copy()

# ---------------------------------------------------------
 # VERIFICATION
# ---------------------------------------------------------

    print("\nTop channels by trending frequency:")

    for region in sorted(channel_frequency["region_code"].dropna().unique()):
            print(f"\nRegion: {region}")

            print(
                channel_frequency[
                    channel_frequency["region_code"] == region
                ]
                .head(10)
                .to_string(index=False)
            )
    print("\nChannel dominance verification:")

    for region in sorted(
        channel_daily_dominance["region_code"].dropna().unique()
    ):
            print(f"\nRegion: {region}")

            print(
                channel_daily_dominance[
                    channel_daily_dominance["region_code"] == region
                ]
                .sort_values("dominance_pct", ascending=False)
                .head(10)
                [
                    [
                        "channel_name",
                        "fetch_date",
                        "channel_observations",
                        "total_region_observations",
                        "dominance_pct"
                    ]
                ]
                .to_string(index=False)
            )

    print("\nChannel momentum missing-day verification:")

    print(
            channel_daily_presence[
                (channel_daily_presence["channel_name"] == "2BCProductions2BC")
                & (channel_daily_presence["region_code"] == "CA")
            ][
                [
                    "fetch_date",
                    "previous_trending_observations",
                    "trending_observations",
                    "channel_momentum",
                    "momentum_trend"
                ]
            ].to_string(index=False)
        )


    print("\nChannel subscriber count distribution:")

    print(
        channel_size_data["channel_subscriber_count"]
        .describe()
        .to_string()
    )

    print("\nChannel size groups:")

    print(
        channel_size_data["channel_size"]
        .value_counts()
        .to_string()
    )

    print("\nChannel size thresholds:")
    print(f"25th percentile: {q1:,.0f}")
    print(f"50th percentile: {q2:,.0f}")
    print(f"75th percentile: {q3:,.0f}")


    print("\nLarge vs smaller channel performance:")

    print(
        channel_size_performance
        .sort_values("channel_size")
        .to_string(index=False)
    )

    print("\nSubscriber vs trending performance sample:")

    print(
        channel_performance[
            [
                "channel_name",
                "subscriber_count",
                "unique_trending_videos",
                "trending_observations",
                "average_trending_rank",
                "best_trending_rank"
            ]
        ]
        .sort_values("subscriber_count", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nChannels in subscriber-performance analysis:",
        len(channel_performance)
    )

    print("\nSubscriber count vs trending performance correlations:")

    print(
        subscriber_correlations.loc[
            "subscriber_count",
            [
                "unique_trending_videos",
                "trending_observations",
                "average_trending_rank",
                "best_trending_rank"
            ]
        ].to_string()
    )

    print("\nHistorical competitor/channel comparison:")

    test_channel_comparison = compare_channels(
        df,
        "Cash",
        "CaseOh"
    )

    print(
        test_channel_comparison.to_string(index=False)
    )


    print("\nCategory trends sample:")

    sample_category = (
        category_daily_trends[
            category_daily_trends["region_code"] == "US"
        ]
        ["category_name"]
        .value_counts()
        .index[0]
    )

    print("\nCategory complete-calendar verification:")

    print(
        category_daily_trends[
            (category_daily_trends["category_name"] == "Autos & Vehicles")
            & (category_daily_trends["region_code"] == "CA")
            & (
                category_daily_trends["fetch_date"].between(
                    "2026-09-14",
                    "2026-09-27"
                )
            )
        ][
            [
                "fetch_date",
                "previous_trending_observations",
                "trending_observations",
                "observation_change"
            ]
        ]
        .to_string(index=False)
    )


    print("\nCategory trending frequency:")

    for region in sorted(
        category_frequency["region_code"].dropna().unique()
    ):
        print(f"\nRegion: {region}")

        print(
            category_frequency[
                category_frequency["region_code"] == region
            ]
            .head(10)
            .to_string(index=False)
        )


    print("\nCategory dominance verification:")

    latest_us_date = (
        category_dominance[
            category_dominance["region_code"] == "US"
        ]["fetch_date"]
        .max()
    )

    latest_us_dominance = (
        category_dominance[
            (category_dominance["region_code"] == "US")
            & (category_dominance["fetch_date"] == latest_us_date)
        ]
        .sort_values(
            "dominance_pct",
            ascending=False
        )
    )

    print("US date:", latest_us_date)

    print(
        latest_us_dominance[
            [
                "category_name",
                "category_observations",
                "total_region_observations",
                "dominance_pct"
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\nTotal category dominance:",
        latest_us_dominance["dominance_pct"].sum()
    )


    print("\nCategory comparison between regions:")

    print(
        category_region_comparison[
            category_region_comparison["category_name"].isin(
                ["Gaming", "Music"]
            )
        ][
            [
                "category_name",
                "region_code",
                "unique_trending_videos",
                "trending_observations",
                "active_trending_days",
                "average_trending_rank",
                "best_trending_rank",
                "historical_dominance_pct"
            ]
        ]
        .sort_values(
            ["category_name", "region_code"]
        )
        .to_string(index=False)
    )

   
    print("\nReliable category rank/growth patterns — US:")

    print(
        category_performance_reliable[
            category_performance_reliable["region_code"] == "US"
        ][
            [
                "category_name",
                "unique_trending_videos",
                "average_trending_rank",
                "average_view_velocity",
                "average_like_velocity",
                "average_comment_velocity",
                "average_engagement_rate"
            ]
        ]
        .sort_values(
            "average_trending_rank"
        )
        .to_string(index=False)
    )

    print("\nRegional comparison:")

    print(
        regional_comparison[
            [
                "region_code",
                "historical_observations",
                "unique_trending_videos",
                "unique_channels",
                "active_collection_days",
                "average_trending_rank",
                "median_trending_rank",
                "average_view_velocity",
                "average_engagement_rate"
            ]
        ]
        .sort_values("region_code")
        .to_string(index=False)
        )


    print("\nCross-country rank comparison:")

    sample_multi_region_ids = (
        cross_country_rank
        .groupby("id")["region_code"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    print(
        cross_country_rank[
            cross_country_rank["id"].isin(sample_multi_region_ids)
        ][
            [
                "id",
                "video_title",
                "region_code",
                "average_rank",
                "best_rank",
                "trending_observations"
            ]
        ]
        .sort_values(["id", "region_code"])
        .to_string(index=False)
    )

    print("\nMulti-region trending videos:")

    print(
        multi_region_videos[
            [
                "id",
                "video_title",
                "regions_trending",
                "region_list",
                "total_trending_observations",
                "first_seen",
                "last_seen"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nTotal multi-region videos:",
        len(multi_region_videos)
    )

    print(
        "Videos trending in all collected regions:",
        (multi_region_videos["regions_trending"] == df["region_code"].nunique()).sum()
    )


    print("\nRegion-specific trending videos:")

    print(
        region_specific_videos[
            [
                "id",
                "video_title",
                "region_code",
                "total_trending_observations",
                "first_seen",
                "last_seen"
            ]
        ]
        .sort_values(
            ["region_code", "total_trending_observations"],
            ascending=[True, False]
        )
        .groupby("region_code")
        .head(5)
        .to_string(index=False)
    )

    print("\nRegion-specific video counts:")

    print(
        region_specific_videos[
            "region_code"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTotal region-specific videos:",
        len(region_specific_videos)
    )


    print("\nCross-region vs region-specific performance:")

    print(
        reach_performance_summary.to_string(index=False)
    )

    print(
        "\nUnique videos represented:",
        reach_performance_summary["unique_videos"].sum()
    )

    print(
        "Total unique videos in historical data:",
        df["id"].nunique()
    )



    print("\nTop categories by region:")

    print(
        category_popularity_across_regions[
            category_popularity_across_regions["popularity_rank"] <= 5
        ][
            [
                "region_code",
                "popularity_rank",
                "category_name",
                "unique_trending_videos",
                "trending_observations",
                "historical_dominance_pct"
            ]
        ]
        .to_string(index=False)
    )

    print("\nTop dominant channels by region:")

    print(
        channel_region_dominance[
            channel_region_dominance["dominance_rank"] <= 5
        ][
            [
                "region_code",
                "dominance_rank",
                "channel_name",
                "unique_trending_videos",
                "trending_observations",
                "active_trending_days",
                "dominance_pct"
            ]
        ]
        .sort_values(
            ["region_code", "dominance_rank"]
        )
        .to_string(index=False)
    )


    print("\nExpected vs received record summary:")

    print(
        snapshot_record_counts
        .groupby(["region_code", "record_status"])
        .agg(
            snapshots=("fetch_timestamp", "count"),
            average_received=("records_received", "mean"),
            minimum_received=("records_received", "min"),
            average_completion_pct=("completion_pct", "mean")
        )
        .reset_index()
        .to_string(index=False)
    )

    print(
        "\nSnapshots below expected:",
        (snapshot_record_counts["records_received"] < EXPECTED_RECORDS_PER_SNAPSHOT).sum()
    )

    print(
        "Total snapshots:",
        len(snapshot_record_counts)
    )

    
    print("\nMissing-field quality summary:")

    print(
        missing_field_summary.to_string(
            index=False
        )
    )

    print(
        "\nTotal historical observations:",
        len(df)
    )

    print("\nDuplicate quality summary:")

    print(
        "Duplicate observations:",
        duplicate_observation_count
    )

    print(
        "Duplicate percentage:",
        round(duplicate_observation_pct, 6),
        "%"
    )

    print(
        "Unique observations:",
        len(df) - duplicate_observation_count
    )

    print(
        "Total observations:",
        len(df)
    )

    if duplicate_observation_count > 0:
        print("\nDuplicate rows:")
        print(
            duplicate_rows[
                [
                    "id",
                    "region_code",
                    "fetch_timestamp",
                    "trending_rank"
                ]
            ]
            .sort_values(
                ["id", "region_code", "fetch_timestamp"]
            )
            .head(20)
            .to_string(index=False)
        )

    print("\nRegion completeness summary:")

    print(
        "Expected regions:",
        sorted(EXPECTED_REGIONS)
    )

    print(
        "Observed regions:",
        sorted(observed_regions)
    )

    print(
        "Missing regions overall:",
        sorted(missing_regions_overall)
    )

    print(
        "Unexpected regions:",
        sorted(unexpected_regions)
    )

    print(
        "Total collection days:",
        len(daily_region_presence)
    )

    print(
        "Days with all expected regions:",
        daily_region_presence[
            "all_regions_present"
        ].sum()
    )

    print(
        "Days missing one or more regions:",
        len(incomplete_region_days)
    )

    if len(incomplete_region_days) > 0:
        print("\nIncomplete region days:")

        print(
            incomplete_region_days[
                [
                    "fetch_date",
                    "region_count",
                    "regions_present",
                    "missing_regions"
                ]
            ]
            .to_string(index=False)
        )

    print("\nSnapshot timing pattern:")

    snapshot_timing = (
        snapshot_record_counts[
            [
                "region_code",
                "fetch_timestamp"
            ]
        ]
        .sort_values(
            ["region_code", "fetch_timestamp"]
        )
        .copy()
    )

    snapshot_timing["previous_snapshot"] = (
        snapshot_timing
        .groupby("region_code")["fetch_timestamp"]
        .shift(1)
    )

    snapshot_timing["gap_hours"] = (
        snapshot_timing["fetch_timestamp"]
        - snapshot_timing["previous_snapshot"]
    ).dt.total_seconds() / 3600

    print(
        snapshot_timing
        .groupby("region_code")["gap_hours"]
        .describe()[
            ["count", "mean", "50%", "max"]
        ]
        .to_string()
    )

    print("\nLargest snapshot gaps:")

    print(
        snapshot_timing[
            [
                "region_code",
                "previous_snapshot",
                "fetch_timestamp",
                "gap_hours"
            ]
        ]
        .sort_values(
            "gap_hours",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nCollection-window completeness summary:")

    print(
        collection_window_summary.to_string(index=False)
    )

    print(
        "\nTotal detected large gaps:",
        len(collection_gaps)
    )

    print(
        "Estimated missing collection windows:",
        collection_gaps["estimated_missing_windows"].sum()
    )

    print("\nDetected collection gaps:")

    print(
        collection_gaps[
            [
                "region_code",
                "previous_snapshot",
                "fetch_timestamp",
                "gap_hours",
                "estimated_missing_windows"
            ]
        ]
        .sort_values(
            "gap_hours",
            ascending=False
        )
        .to_string(index=False)
    )

    print("\nMissing region/window snapshot summary:")

    print(
        missing_window_summary
        .groupby("missing_region")
        .agg(
            gap_events=("gap_start", "count"),
            estimated_missing_snapshots=(
                "estimated_missing_windows",
                "sum"
            ),
            largest_gap_hours=("gap_hours", "max")
        )
        .reset_index()
        .to_string(index=False)
    )

    print(
        "\nTotal estimated missing region snapshots:",
        missing_window_summary[
            "estimated_missing_windows"
        ].sum()
    )

    print("\nLargest missing region/window events:")

    print(
        missing_window_summary
        .sort_values(
            "gap_hours",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nCategory enrichment completeness:")

    print(
        category_enrichment_summary
        .to_string(index=False)
    )

    print("\nCategory missingness by schema version:")

    print(
        category_missing_by_schema
        .to_string(index=False)
    )

    print(
        "\nOverall category completeness:",
        round(
            df["category_name"].notna().mean() * 100,
            4
        ),
        "%"
    )

    print(
        "Total missing category values:",
        df["category_name"].isna().sum()
    )

    print("\nChannel enrichment completeness:")

    print(
        channel_enrichment_summary
        .to_string(index=False)
    )

    print("\nChannel missingness by schema version:")

    print(
        channel_missing_by_schema
        .to_string(index=False)
    )

    print(
        "\nOverall channel completeness:",
        round(
            df["channel_enriched"].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "Total observations missing channel enrichment:",
        (~df["channel_enriched"]).sum()
    )

    print("\nLast successful pipeline run:")

    if last_successful_run_summary is not None:
        for key, value in last_successful_run_summary.items():
            print(f"{key}: {value}")
    else:
        print("No successful pipeline runs found.")

    print("\nPipeline freshness:")

    if freshness_summary is not None:

        print(
            "Checked at:",
            freshness_summary["checked_at"]
        )

        print(
            "Last successful run:",
            freshness_summary["last_successful_run"]
        )

        print(
            "Freshness hours:",
            round(
                freshness_summary["freshness_hours"],
                2
            )
        )

        print(
            "Freshness status:",
            freshness_summary["freshness_status"]
        )

    else:
        print(
            "Freshness unavailable: "
            "no successful runs found."
        )
    

    print("\nPipeline success/failure summary:")

    print(
        run_status_summary
        .to_string(index=False)
    )

    print("Total audited runs:", total_audit_runs)
    print("Successful runs:", successful_run_count)
    print("Failed runs:", failed_run_count)

    print(
        "Success rate:",
        round(success_rate_pct, 2),
        "%"
    )

    print(
        "Failure rate:",
        round(failure_rate_pct, 2),
        "%"
    )

    print("\nHistorical pipeline health:")

    print(
        historical_health
        .to_string(index=False)
    )

    degraded_days = historical_health[
        historical_health["failed_runs"] > 0
    ]

    print(
        "\nDays with pipeline failures:",
        len(degraded_days)
    )

    if not degraded_days.empty:

        print("\nDegraded pipeline days:")

        print(
            degraded_days
            .to_string(index=False)
        )

    print("\nPipeline retry history:")

    print("Total retries:", int(total_retries))

    print(
        "Runs requiring retries:",
        int(runs_with_retries)
    )

    print(
        "Maximum retries in a single run:",
        int(max_retries_single_run)
    )

    if not retry_history.empty:

        print("\nRuns with retries:")

        print(
            retry_history
            .to_string(index=False)
        )

    else:
        print("No audited runs required retries.")


    print("\nRecords fetched/written history:")

    print(
        "Total records fetched:",
        int(total_records_fetched)
    )

    print(
        "Total records written:",
        int(total_records_written)
    )

    print(
        "Runs with fetched/written mismatch:",
        int(runs_with_record_mismatch)
    )

    print("\nRecords per run summary:")

    print(
        audit_df[
            ["records_fetched", "records_written"]
        ]
        .describe()
        .to_string()
    )

    if not record_mismatch_runs.empty:

        print("\nRuns with record mismatch:")

        print(
            record_mismatch_runs
            .to_string(index=False)
        )

    else:
        print(
            "\nAll audited runs have matching "
            "fetched/written record counts."
        )


    print("\nFailed-run/error summary:")

    print(
        "Total failed runs:",
        failed_run_count
    )

    print("\nFailures by error type:")

    print(
        error_type_summary
        .to_string(index=False)
    )

    print("\nFailed runs:")

    print(
        failed_run_summary
        .to_string(index=False)
    )