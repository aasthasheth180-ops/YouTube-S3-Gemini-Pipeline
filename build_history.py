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


if __name__ == "__main__":
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