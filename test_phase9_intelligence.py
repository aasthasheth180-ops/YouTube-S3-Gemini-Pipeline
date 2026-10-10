import pandas as pd

from build_history import (
    list_historical_snapshots,
    combine_historical_snapshots
)

from analytics.historical_intelligence import (
    prepare_historical_intelligence
)

keys = list_historical_snapshots()
records = combine_historical_snapshots(keys)

df = pd.DataFrame(records)

intel = prepare_historical_intelligence(df)

print("\n=== PHASE 9 PYTHON INTELLIGENCE TEST ===")
print("Historical rows:", len(intel["video_history"]))
print("Latest video-region states:", len(intel["latest_video_state"]))
print("Rising videos:", len(intel["rising_videos"]))
print("Channel trend rows:", len(intel["channel_trends"]))
print("Category trend rows:", len(intel["category_trends"]))
print("Regional trend rows:", len(intel["regional_trends"]))

print("\n=== REGIONS ===")
print(
    intel["regional_trends"].to_string(index=False)
)

print("\n=== GLOBAL VS REGIONAL ===")
print(
    intel["global_vs_regional"].to_string(index=False)
)

print("\n=== TOP 10 CURRENT RISING VIDEOS ===")
print(
    intel["rising_videos"][
        [
            "video_title",
            "region_code",
            "previous_rank",
            "trending_rank",
            "rank_movement",
            "view_velocity",
            "like_velocity",
            "comment_velocity",
            "engagement_rate",
            "days_trending",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
