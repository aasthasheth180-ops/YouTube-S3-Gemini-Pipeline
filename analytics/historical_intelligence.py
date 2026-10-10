import pandas as pd
import numpy as np


def prepare_historical_intelligence(df):
    """
    Calculate reusable historical intelligence from YouTube snapshots.

    IMPORTANT:
    This layer calculates facts only.
    It does not call Gemini and does not generate natural-language claims.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------------

    df["fetch_timestamp"] = pd.to_datetime(
        df["fetch_timestamp"],
        utc=True,
        errors="coerce"
    )

    df["fetch_date"] = pd.to_datetime(
        df["fetch_date"],
        errors="coerce"
    ).dt.normalize()

    numeric_columns = [
        "trending_rank",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Extract nested statistics.
    df["view_count"] = df["statistics"].apply(
        lambda x: x.get("viewCount")
        if isinstance(x, dict)
        else None
    )

    df["like_count"] = df["statistics"].apply(
        lambda x: x.get("likeCount")
        if isinstance(x, dict)
        else None
    )

    df["comment_count"] = df["statistics"].apply(
        lambda x: x.get("commentCount")
        if isinstance(x, dict)
        else None
    )

    for column in [
        "view_count",
        "like_count",
        "comment_count",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Video title is useful for Gemini explanations.
    df["video_title"] = df["snippet"].apply(
        lambda x: x.get("title")
        if isinstance(x, dict)
        else None
    )

    df = df.sort_values(
        ["id", "region_code", "fetch_timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # RANK INTELLIGENCE
    # ---------------------------------------------------------

    group = df.groupby(
        ["id", "region_code"],
        sort=False
    )

    df["previous_rank"] = group["trending_rank"].shift(1)

    # Positive movement = rising toward rank #1.
    df["rank_movement"] = (
        df["previous_rank"] - df["trending_rank"]
    )

    df["absolute_rank_movement"] = (
        df["rank_movement"].abs()
    )

    # ---------------------------------------------------------
    # VIDEO LIFECYCLE
    # ---------------------------------------------------------

    df["first_seen"] = group[
        "fetch_timestamp"
    ].transform("min")

    df["last_seen"] = group[
        "fetch_timestamp"
    ].transform("max")

    df["trending_observations"] = (
        group.cumcount() + 1
    )

    df["hours_trending"] = (
        df["fetch_timestamp"] - df["first_seen"]
    ).dt.total_seconds() / 3600

    df["days_trending"] = (
        df["hours_trending"] / 24
    )

    # ---------------------------------------------------------
    # VIEW GROWTH / VELOCITY
    # ---------------------------------------------------------

    df["previous_view_count"] = (
        group["view_count"].shift(1)
    )

    df["view_growth"] = (
        df["view_count"]
        - df["previous_view_count"]
    )

    df["previous_fetch_timestamp"] = (
        group["fetch_timestamp"].shift(1)
    )

    df["elapsed_hours"] = (
        df["fetch_timestamp"]
        - df["previous_fetch_timestamp"]
    ).dt.total_seconds() / 3600

    df["view_velocity"] = (
        df["view_growth"]
        / df["elapsed_hours"]
    )

    df.loc[
        df["elapsed_hours"] <= 0,
        "view_velocity"
    ] = np.nan

    df["previous_view_velocity"] = (
        group["view_velocity"].shift(1)
    )

    df["view_velocity_change"] = (
        df["view_velocity"]
        - df["previous_view_velocity"]
    )

    # ---------------------------------------------------------
    # LIKE / COMMENT VELOCITY
    # ---------------------------------------------------------

    df["previous_like_count"] = (
        group["like_count"].shift(1)
    )

    df["like_growth"] = (
        df["like_count"]
        - df["previous_like_count"]
    )

    df["like_velocity"] = (
        df["like_growth"]
        / df["elapsed_hours"]
    )

    df.loc[
        df["elapsed_hours"] <= 0,
        "like_velocity"
    ] = np.nan

    df["previous_comment_count"] = (
        group["comment_count"].shift(1)
    )

    df["comment_growth"] = (
        df["comment_count"]
        - df["previous_comment_count"]
    )

    df["comment_velocity"] = (
        df["comment_growth"]
        / df["elapsed_hours"]
    )

    df.loc[
        df["elapsed_hours"] <= 0,
        "comment_velocity"
    ] = np.nan

    # ---------------------------------------------------------
    # ENGAGEMENT
    # ---------------------------------------------------------

    df["engagement_rate"] = (
        (df["like_count"] + df["comment_count"])
        / df["view_count"]
    )

    df.loc[
        df["view_count"] <= 0,
        "engagement_rate"
    ] = np.nan

    df["previous_engagement_rate"] = (
        group["engagement_rate"].shift(1)
    )

    df["engagement_rate_movement"] = (
        df["engagement_rate"]
        - df["previous_engagement_rate"]
    )

    # ---------------------------------------------------------
    # CHANNEL TRENDS
    # ---------------------------------------------------------

    channel_trends = (
        df.dropna(subset=["channel_name"])
        .groupby(["channel_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique"),
            average_rank=("trending_rank", "mean"),
            best_rank=("trending_rank", "min"),
            average_view_velocity=("view_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # CATEGORY TRENDS
    # ---------------------------------------------------------

    category_trends = (
        df.dropna(subset=["category_name"])
        .groupby(["category_name", "region_code"])
        .agg(
            unique_trending_videos=("id", "nunique"),
            trending_observations=("id", "count"),
            active_trending_days=("fetch_date", "nunique"),
            average_rank=("trending_rank", "mean"),
            best_rank=("trending_rank", "min"),
            average_view_velocity=("view_velocity", "mean"),
            average_like_velocity=("like_velocity", "mean"),
            average_comment_velocity=("comment_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # REGIONAL TRENDS
    # ---------------------------------------------------------

    regional_trends = (
        df.groupby("region_code")
        .agg(
            historical_observations=("id", "count"),
            unique_trending_videos=("id", "nunique"),
            unique_channels=("channel_name", "nunique"),
            active_collection_days=("fetch_date", "nunique"),
            average_rank=("trending_rank", "mean"),
            average_view_velocity=("view_velocity", "mean"),
            average_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # GLOBAL VS REGIONAL REACH
    # ---------------------------------------------------------

    video_region_reach = (
        df.groupby("id")["region_code"]
        .nunique()
        .rename("regions_trending")
        .reset_index()
    )

    global_regional_df = df.merge(
        video_region_reach,
        on="id",
        how="left"
    )

    global_regional_df["reach_type"] = np.where(
        global_regional_df["regions_trending"] >= 2,
        "cross_region",
        "region_specific"
    )

    global_vs_regional = (
        global_regional_df
        .groupby("reach_type")
        .agg(
            unique_videos=("id", "nunique"),
            average_regions_trending=(
                "regions_trending",
                "mean"
            ),
            average_rank=("trending_rank", "mean"),
            average_view_velocity=(
                "view_velocity",
                "mean"
            ),
            average_engagement_rate=(
                "engagement_rate",
                "mean"
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # LATEST STATE FOR VIDEO EXPLANATIONS
    # ---------------------------------------------------------

    latest_video_state = (
        df.sort_values("fetch_timestamp")
        .groupby(
            ["id", "region_code"],
            as_index=False
        )
        .tail(1)
        .copy()
    )

    # Strongest currently observed rank rises.
    rising_videos = (
        latest_video_state[
            latest_video_state["rank_movement"] > 0
        ]
        .sort_values(
            [
                "rank_movement",
                "view_velocity",
            ],
            ascending=[False, False]
        )
        .copy()
    )

    return {
        "video_history": df,
        "latest_video_state": latest_video_state,
        "rising_videos": rising_videos,
        "channel_trends": channel_trends,
        "category_trends": category_trends,
        "regional_trends": regional_trends,
        "global_vs_regional": global_vs_regional,
    }