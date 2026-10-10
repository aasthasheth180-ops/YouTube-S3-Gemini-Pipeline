"""
YouTube Trending Intelligence Dashboard
Portfolio Project

Run:
    streamlit run dashboard.py
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from ai.build_fact_package import build_fact_package, load_raw_history, add_prediction_results
from ai.gemini_interpreter import (
    generate_trend_summary,
    explain_rising_videos,
    explain_regional_differences,
)
from analytics.historical_intelligence import prepare_historical_intelligence
from build_history import load_audit_history

load_dotenv()

st.set_page_config(
    page_title="YouTube Trending Intelligence",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXPECTED_REGIONS = {"CA", "GB", "IN", "US"}
FRESHNESS_HOURS = 9

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 12px;
    padding: 14px;
}
.small-note {opacity:.75; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)



def youtube_thumbnail(video_id):
    """Standard YouTube thumbnail derived from the public video id."""
    if not video_id:
        return ""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def safe_text(value, fallback="Unknown"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return str(value)


def category_icon(name):
    name = safe_text(name, "").lower()
    if "gaming" in name:
        return "🎮"
    if "music" in name:
        return "🎵"
    if "sport" in name:
        return "🏅"
    if "news" in name:
        return "📰"
    if "science" in name or "tech" in name:
        return "💻"
    if "education" in name:
        return "📚"
    if "film" in name or "entertainment" in name:
        return "🎬"
    if "comedy" in name:
        return "😄"
    return "▶️"


def show_video_card(row, key_prefix):
    """Compact YouTube-style video card with expandable intelligence."""
    video_id = safe_text(row.get("id"), "")
    title = safe_text(row.get("video_title"), "Untitled video")
    channel = safe_text(row.get("channel_name"), "Unknown channel")
    category = safe_text(row.get("category_name"), "Unknown")
    region = safe_text(row.get("region_code"), "—")
    rank = row.get("trending_rank")
    movement = row.get("rank_movement")
    velocity = row.get("view_velocity")
    probability = row.get("still_trending_tomorrow_probability")

    st.image(youtube_thumbnail(video_id), use_container_width=True)
    st.markdown(f"**{title}**")
    st.caption(f"{channel} · {category} · {region}")

    m1, m2 = st.columns(2)
    m1.metric("Rank", f"#{int(rank)}" if pd.notna(rank) else "—")
    if pd.notna(movement):
        arrow = "▲" if movement > 0 else "▼" if movement < 0 else "→"
        m2.metric("Movement", f"{arrow} {abs(int(movement))}")
    else:
        m2.metric("Movement", "—")

    if pd.notna(velocity):
        st.caption(f"⚡ {velocity:,.0f} views/hour")
    if pd.notna(probability):
        st.progress(float(max(0, min(1, probability))))
        st.caption(f"🤖 Tomorrow persistence: {probability:.0%}")

    with st.expander("View intelligence"):
        details = {
            "Current rank": f"#{int(rank)}" if pd.notna(rank) else "—",
            "Previous rank": (
                f"#{int(row.get('previous_rank'))}"
                if pd.notna(row.get("previous_rank")) else "—"
            ),
            "Rank movement": fmt_float(movement, 0),
            "Views": fmt_int(row.get("view_count")),
            "View growth": fmt_int(row.get("view_growth")),
            "View velocity / hour": fmt_int(velocity),
            "Like velocity / hour": fmt_float(row.get("like_velocity")),
            "Comment velocity / hour": fmt_float(row.get("comment_velocity")),
            "Engagement": fmt_pct_fraction(row.get("engagement_rate")),
            "Trending observations": fmt_int(row.get("trending_observations")),
            "Hours trending": fmt_float(row.get("hours_trending")),
            "Days trending": fmt_float(row.get("days_trending")),
        }
        st.dataframe(
            pd.DataFrame(details.items(), columns=["Metric", "Value"]),
            hide_index=True,
            use_container_width=True,
        )


def fmt_int(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def fmt_float(value, digits=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_pct_fraction(value, digits=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


@st.cache_data(ttl=1800, show_spinner=False)
def load_dashboard_data():
    raw = load_raw_history()
    intel = prepare_historical_intelligence(raw)
    latest_with_predictions, prediction_date = add_prediction_results(
        raw, intel["latest_video_state"]
    )
    fact_package = build_fact_package()

    try:
        audit = load_audit_history()
        if not isinstance(audit, pd.DataFrame):
            audit = pd.DataFrame(audit)
    except Exception:
        audit = pd.DataFrame()

    return raw, intel, latest_with_predictions, prediction_date, fact_package, audit


def current_prediction_rows(latest_with_predictions, prediction_date):
    result = latest_with_predictions.copy()
    result["observed_date"] = (
        pd.to_datetime(result["fetch_timestamp"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    date = pd.to_datetime(prediction_date).normalize()
    return result[result["observed_date"] == date].copy()


def build_quality_metrics(raw):
    q = raw.copy()
    q["fetch_date"] = pd.to_datetime(q["fetch_date"], errors="coerce").dt.normalize()

    duplicate_count = int(
        q.duplicated(subset=["id", "region_code", "fetch_timestamp"]).sum()
    )

    missing = {}
    for col in [
        "id", "region_code", "fetch_timestamp", "trending_rank",
        "category_name", "channel_name"
    ]:
        if col in q.columns:
            missing[col] = int(q[col].isna().sum())

    region_days = (
        q.dropna(subset=["fetch_date", "region_code"])
        .groupby("fetch_date")["region_code"]
        .agg(lambda x: set(x))
    )
    complete_days = int(region_days.apply(lambda x: EXPECTED_REGIONS.issubset(x)).sum())
    incomplete_days = int(len(region_days) - complete_days)

    category_complete = (
        100 * q["category_name"].notna().mean()
        if "category_name" in q.columns and len(q) else np.nan
    )
    channel_complete = (
        100 * q["channel_name"].notna().mean()
        if "channel_name" in q.columns and len(q) else np.nan
    )

    return {
        "duplicate_count": duplicate_count,
        "missing": missing,
        "complete_days": complete_days,
        "incomplete_days": incomplete_days,
        "category_complete": category_complete,
        "channel_complete": channel_complete,
    }


def normalize_audit(audit):
    if audit.empty:
        return audit
    a = audit.copy()
    for col in ["started_at", "ended_at"]:
        if col in a.columns:
            a[col] = pd.to_datetime(a[col], utc=True, errors="coerce")
    for col in ["records_fetched", "records_written", "retry_count"]:
        if col in a.columns:
            a[col] = pd.to_numeric(a[col], errors="coerce").fillna(0)
    return a


with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg",
        width=125,
    )
    st.markdown("## Project Status")
    if os.getenv("S3_BUCKET"):
        st.success("S3 configuration loaded")
    else:
        st.error("S3 configuration missing")

    if os.getenv("GOOGLE_API_KEY"):
        st.success("Gemini configuration loaded")
    else:
        st.warning("Gemini key not configured")

    refresh = st.button("Refresh historical data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

st.title("YouTube Trending Intelligence")
st.caption(
    "Historical trend analytics · Cross-region intelligence · "
    "Random Forest persistence prediction · Grounded Gemini explanations"
)

try:
    with st.spinner("Loading historical intelligence from S3..."):
        raw, intel, latest_pred, prediction_date, facts, audit = load_dashboard_data()
except Exception as exc:
    st.error(f"Dashboard data load failed: {exc}")
    st.stop()

audit = normalize_audit(audit)
prediction_rows = current_prediction_rows(latest_pred, prediction_date)
quality = build_quality_metrics(raw)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Historical observations", fmt_int(len(raw)))
k2.metric("Unique videos", fmt_int(raw["id"].nunique()))
k3.metric("Regions", fmt_int(raw["region_code"].nunique()))
k4.metric("Prediction date", str(pd.to_datetime(prediction_date).date()))
k5.metric("Model", "Random Forest")

tabs = st.tabs([
    "🏠 Overview",
    "🔥 Trend Intelligence",
    "🎬 Content",
    "🌎 Regions",
    "🤖 ML Prediction",
    "✨ AI Intelligence",
    "⚙️ Pipeline Health",
    "🔎 Explorer",
])

# ---------------------------------------------------------------------
# OVERVIEW — current trending pulse
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("🔥 Current Trending Pulse")
    st.caption(
        "Latest observed video-region states, combining rank momentum, velocity, "
        "lifecycle and model outlook."
    )

    latest_date = pd.to_datetime(prediction_date).normalize()
    current = latest_pred.copy()
    current["observed_date"] = (
        pd.to_datetime(current["fetch_timestamp"], utc=True, errors="coerce")
        .dt.tz_localize(None).dt.normalize()
    )
    current = current[current["observed_date"] == latest_date].copy()

    rising = current[current["rank_movement"] > 0].sort_values(
        ["rank_movement", "view_velocity"], ascending=[False, False]
    )
    velocity_leader = current.sort_values("view_velocity", ascending=False).head(1)
    best_rank = current.sort_values("trending_rank").head(1)

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Current video-region states", fmt_int(len(current)))
    o2.metric("Rising now", fmt_int(len(rising)))
    o3.metric(
        "Fastest rise",
        f"▲ {int(rising.iloc[0]['rank_movement'])}" if not rising.empty else "—",
    )
    o4.metric(
        "Likely to persist",
        fmt_int(
            (pd.to_numeric(
                current["still_trending_tomorrow_prediction"], errors="coerce"
            ) == 1).sum()
        ),
    )

    st.markdown("### 🚀 Top Movers")
    cards = rising.head(4)
    if cards.empty:
        st.info("No rising videos are available for the latest prediction date.")
    else:
        card_cols = st.columns(len(cards))
        for idx, ((_, row), col) in enumerate(zip(cards.iterrows(), card_cols)):
            with col:
                show_video_card(row, f"overview_{idx}")

    st.markdown("### Today at a glance")
    g1, g2 = st.columns(2)
    with g1:
        if not velocity_leader.empty:
            row = velocity_leader.iloc[0]
            st.info(
                f"⚡ **Velocity leader:** {safe_text(row.get('video_title'))}\n\n"
                f"{fmt_int(row.get('view_velocity'))} views/hour · "
                f"{safe_text(row.get('region_code'))}"
            )
    with g2:
        if not best_rank.empty:
            row = best_rank.iloc[0]
            st.info(
                f"🏆 **Top observed rank:** {safe_text(row.get('video_title'))}\n\n"
                f"Rank #{int(row.get('trending_rank'))} · "
                f"{safe_text(row.get('region_code'))}"
            )


# ---------------------------------------------------------------------
# TREND INTELLIGENCE
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔥 Trend Intelligence")
    hist_tab, rank_tab, velocity_tab, life_tab = st.tabs(
        ["Historical", "Rank Movers", "Growth & Velocity", "Lifecycle"]
    )

    history = intel["video_history"].copy()
    history["fetch_date"] = pd.to_datetime(history["fetch_date"], errors="coerce")

    with hist_tab:
        daily = (
            history.groupby("fetch_date", as_index=False)
            .agg(
                observations=("id", "count"),
                unique_videos=("id", "nunique"),
                unique_channels=("channel_name", "nunique"),
            )
            .sort_values("fetch_date")
        )
        fig = px.line(
            daily,
            x="fetch_date",
            y=["observations", "unique_videos"],
            markers=True,
            title="Historical Trending Activity",
            labels={"value": "Count", "fetch_date": "Collection date", "variable": "Metric"},
        )
        st.plotly_chart(fig, use_container_width=True)

        region_daily = (
            history.groupby(["fetch_date", "region_code"], as_index=False)
            .agg(observations=("id", "count"))
        )
        fig = px.line(
            region_daily,
            x="fetch_date",
            y="observations",
            color="region_code",
            title="Historical Observations by Region",
            labels={"observations": "Observations", "fetch_date": "Collection date"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with rank_tab:
        latest_states = intel["latest_video_state"].copy()
        movers = latest_states.dropna(subset=["rank_movement"]).copy()
        fastest_rising = movers.sort_values("rank_movement", ascending=False).head(12)
        fastest_falling = movers.sort_values("rank_movement", ascending=True).head(12)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### 🚀 Fastest Rising")
            chart = fastest_rising.sort_values("rank_movement")
            fig = px.bar(
                chart, x="rank_movement", y="video_title", orientation="h",
                title="Largest Positive Rank Movement",
                labels={"rank_movement": "Positions gained", "video_title": "Video"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            st.markdown("#### 📉 Fastest Falling")
            chart = fastest_falling.copy()
            chart["positions_lost"] = chart["rank_movement"].abs()
            chart = chart.sort_values("positions_lost")
            fig = px.bar(
                chart, x="positions_lost", y="video_title", orientation="h",
                title="Largest Negative Rank Movement",
                labels={"positions_lost": "Positions lost", "video_title": "Video"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with velocity_tab:
        metric_map = {
            "View velocity": "view_velocity",
            "View growth": "view_growth",
            "Like velocity": "like_velocity",
            "Comment velocity": "comment_velocity",
            "Engagement movement": "engagement_rate_movement",
        }
        label = st.selectbox("Momentum metric", list(metric_map), key="velocity_metric")
        metric_col = metric_map[label]
        leaders = (
            intel["latest_video_state"]
            .dropna(subset=[metric_col])
            .sort_values(metric_col, ascending=False)
            .head(15)
            .copy()
        )
        fig = px.bar(
            leaders.sort_values(metric_col),
            x=metric_col, y="video_title", orientation="h",
            color="region_code",
            title=f"Leaders — {label}",
            labels={metric_col: label, "video_title": "Video", "region_code": "Region"},
        )
        st.plotly_chart(fig, use_container_width=True)
        cols = [
            "video_title", "channel_name", "category_name", "region_code",
            "trending_rank", "rank_movement", metric_col
        ]
        st.dataframe(leaders[cols], hide_index=True, use_container_width=True)

    with life_tab:
        life = (
            intel["latest_video_state"]
            .sort_values(["days_trending", "trending_observations"], ascending=False)
            .head(20)
            .copy()
        )
        l1, l2, l3 = st.columns(3)
        l1.metric("Longest observed lifecycle", f"{life['days_trending'].max():.1f} days")
        l2.metric("Max observations", fmt_int(life["trending_observations"].max()))
        l3.metric(
            "New on latest date",
            fmt_int((current["trending_observations"] == 1).sum()),
        )

        fig = px.scatter(
            life,
            x="hours_trending",
            y="trending_observations",
            size="view_count",
            color="region_code",
            hover_name="video_title",
            title="Lifecycle: Time Trending vs Observations",
            labels={
                "hours_trending": "Hours trending",
                "trending_observations": "Trending observations",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        life_cols = [
            "video_title", "channel_name", "region_code", "first_seen", "last_seen",
            "trending_observations", "hours_trending", "days_trending", "trending_rank"
        ]
        st.dataframe(life[life_cols], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# CONTENT INTELLIGENCE
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("🎬 Content Intelligence")
    channel_tab, category_tab = st.tabs(["Channels", "Categories"])

    with channel_tab:
        region_choice = st.selectbox(
            "Channel region",
            ["All"] + sorted(intel["channel_trends"]["region_code"].dropna().unique().tolist()),
            key="channel_region",
        )
        ch = intel["channel_trends"].copy()
        if region_choice != "All":
            ch = ch[ch["region_code"] == region_choice]
        if region_choice == "All":
            ch = (
                ch.groupby("channel_name", as_index=False)
                .agg(
                    unique_trending_videos=("unique_trending_videos", "sum"),
                    trending_observations=("trending_observations", "sum"),
                    active_trending_days=("active_trending_days", "max"),
                    average_rank=("average_rank", "mean"),
                    average_view_velocity=("average_view_velocity", "mean"),
                    average_engagement_rate=("average_engagement_rate", "mean"),
                )
            )
        leaders = ch.sort_values(
            ["unique_trending_videos", "trending_observations"], ascending=False
        ).head(15)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(
                leaders.sort_values("unique_trending_videos"),
                x="unique_trending_videos", y="channel_name", orientation="h",
                title="Channel Trending Frequency",
                labels={"unique_trending_videos": "Unique trending videos", "channel_name": "Channel"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if not leaders.empty:
                top = leaders.iloc[0]
                st.success(
                    f"🏆 **Most frequent channel**\n\n"
                    f"### {top['channel_name']}\n"
                    f"{fmt_int(top['unique_trending_videos'])} unique trending videos\n\n"
                    f"{fmt_int(top['trending_observations'])} observations\n\n"
                    f"Avg rank: {fmt_float(top['average_rank'])}"
                )
        st.dataframe(leaders, hide_index=True, use_container_width=True)

    with category_tab:
        region_choice = st.selectbox(
            "Category region",
            ["All"] + sorted(intel["category_trends"]["region_code"].dropna().unique().tolist()),
            key="category_region",
        )
        cats = intel["category_trends"].copy()
        if region_choice != "All":
            cats = cats[cats["region_code"] == region_choice]
        else:
            cats = (
                cats.groupby("category_name", as_index=False)
                .agg(
                    unique_trending_videos=("unique_trending_videos", "sum"),
                    trending_observations=("trending_observations", "sum"),
                    active_trending_days=("active_trending_days", "max"),
                    average_rank=("average_rank", "mean"),
                    best_rank=("best_rank", "min"),
                    average_view_velocity=("average_view_velocity", "mean"),
                    average_like_velocity=("average_like_velocity", "mean"),
                    average_comment_velocity=("average_comment_velocity", "mean"),
                    average_engagement_rate=("average_engagement_rate", "mean"),
                )
            )
        cats = cats.sort_values("trending_observations", ascending=False)
        st.markdown("#### Category Board")
        top_cats = cats.head(6)
        card_cols = st.columns(3)
        for i, (_, row) in enumerate(top_cats.iterrows()):
            with card_cols[i % 3]:
                icon = category_icon(row["category_name"])
                st.info(
                    f"{icon} **{row['category_name']}**\n\n"
                    f"**{fmt_int(row['trending_observations'])}** observations  \n"
                    f"⚡ {fmt_int(row['average_view_velocity'])} views/hour  \n"
                    f"🏆 Avg rank {fmt_float(row['average_rank'])}  \n"
                    f"❤️ {fmt_pct_fraction(row['average_engagement_rate'])} engagement"
                )

        selected_category = st.selectbox(
            "Explore category",
            cats["category_name"].tolist(),
            key="selected_category",
        )
        selected = cats[cats["category_name"] == selected_category].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Trending observations", fmt_int(selected["trending_observations"]))
        k2.metric("Unique videos", fmt_int(selected["unique_trending_videos"]))
        k3.metric("Avg view velocity", fmt_int(selected["average_view_velocity"]))
        k4.metric("Avg engagement", fmt_pct_fraction(selected["average_engagement_rate"]))


# ---------------------------------------------------------------------
# REGIONAL INTELLIGENCE
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("🌎 Regional Intelligence")
    regional_tab, reach_tab = st.tabs(["Regional Comparison", "Global vs Regional"])

    with regional_tab:
        regional = intel["regional_trends"].copy().sort_values("region_code")
        cols = st.columns(len(regional))
        for col, (_, row) in zip(cols, regional.iterrows()):
            with col:
                st.markdown(f"### {row['region_code']}")
                st.caption(
                    f"{fmt_int(row['unique_trending_videos'])} videos · "
                    f"{fmt_int(row['unique_channels'])} channels"
                )
                st.metric("Avg rank", fmt_float(row["average_rank"]))
                st.metric("View velocity", fmt_int(row["average_view_velocity"]))
                st.metric("Engagement", fmt_pct_fraction(row["average_engagement_rate"]))

        a, b = st.columns(2)
        with a:
            fig = px.bar(
                regional, x="region_code", y="average_view_velocity",
                title="Average View Velocity",
                labels={"region_code": "Region", "average_view_velocity": "Views / hour"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.bar(
                regional, x="region_code", y="average_engagement_rate",
                title="Average Engagement Rate",
                labels={"region_code": "Region", "average_engagement_rate": "Engagement"},
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

    with reach_tab:
        gr = intel["global_vs_regional"].copy()
        cross = gr[gr["reach_type"] == "cross_region"]
        local = gr[gr["reach_type"] == "region_specific"]

        a, b, c = st.columns(3)
        if not cross.empty and not local.empty:
            a.metric("Cross-region videos", fmt_int(cross.iloc[0]["unique_videos"]))
            b.metric("Region-specific videos", fmt_int(local.iloc[0]["unique_videos"]))
            ratio = cross.iloc[0]["average_view_velocity"] / local.iloc[0]["average_view_velocity"]
            c.metric("Velocity advantage", f"{ratio:.2f}×")

        display = gr.copy()
        display["reach_type"] = display["reach_type"].replace(
            {"cross_region": "Cross-region", "region_specific": "Region-specific"}
        )
        metric = st.selectbox(
            "Comparison metric",
            ["average_view_velocity", "average_rank", "average_engagement_rate"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="reach_metric",
        )
        fig = px.bar(
            display, x="reach_type", y=metric,
            title=metric.replace("_", " ").title(),
            labels={"reach_type": "Reach type", metric: metric.replace("_", " ").title()},
        )
        if metric == "average_engagement_rate":
            fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cross-region means observed in at least two collected regions "
            "(CA, GB, IN, US), not all YouTube markets."
        )


# ---------------------------------------------------------------------
# ML PREDICTION
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("🤖 Tomorrow's Trend Outlook")
    st.markdown(
        "**Prediction question:** Will this video still be trending tomorrow "
        "in the same region?"
    )
    st.caption(
        "Random Forest model · chronological evaluation · probabilities calculated in Python"
    )

    valid = prediction_rows.copy()
    valid["still_trending_tomorrow_probability"] = pd.to_numeric(
        valid["still_trending_tomorrow_probability"], errors="coerce"
    )
    valid = valid.dropna(subset=["still_trending_tomorrow_probability"])

    p1, p2, p3 = st.columns(3)
    p1.metric("Videos analyzed", fmt_int(len(valid)))
    p2.metric(
        "Likely to persist",
        fmt_int((valid["still_trending_tomorrow_prediction"] == 1).sum()),
    )
    p3.metric(
        "Average probability",
        f"{valid['still_trending_tomorrow_probability'].mean():.1%}" if len(valid) else "—",
    )

    st.markdown("### High-confidence persistence")
    top_predictions = valid.sort_values(
        "still_trending_tomorrow_probability", ascending=False
    ).head(10)
    fig = px.bar(
        top_predictions.sort_values("still_trending_tomorrow_probability"),
        x="still_trending_tomorrow_probability",
        y="video_title",
        orientation="h",
        color="region_code",
        title="Highest Persistence Probabilities",
        labels={
            "still_trending_tomorrow_probability": "Persistence probability",
            "video_title": "Video",
        },
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View prediction evidence"):
        cols = [
            "video_title", "channel_name", "region_code", "trending_rank",
            "rank_movement", "view_velocity", "engagement_rate",
            "still_trending_tomorrow_prediction",
            "still_trending_tomorrow_probability",
        ]
        st.dataframe(valid[cols].sort_values(
            "still_trending_tomorrow_probability", ascending=False
        ), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# AI INTELLIGENCE
# ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("✨ AI Intelligence Center")
    st.caption(
        "Python calculates the facts. Gemini interprets only the supplied evidence."
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("GOOGLE_API_KEY is not configured.")
    else:
        ai_mode = st.segmented_control(
            "Choose intelligence view",
            ["Trend Brief", "Rising Stories", "Regional Story"],
            default="Trend Brief",
        )

        if st.button("Generate grounded insight", type="primary"):
            with st.spinner("Interpreting the calculated fact package..."):
                try:
                    if ai_mode == "Rising Stories":
                        answer = explain_rising_videos(facts)
                    elif ai_mode == "Regional Story":
                        answer = explain_regional_differences(facts)
                    else:
                        answer = generate_trend_summary(facts)
                    st.session_state["ai_answer"] = answer
                    st.session_state["ai_mode"] = ai_mode
                except Exception as exc:
                    st.error(f"Gemini interpretation failed: {exc}")

        # Visual evidence first
        st.markdown("### Evidence Snapshot")
        ev1, ev2, ev3 = st.columns(3)
        with ev1:
            if not rising.empty:
                r = rising.iloc[0]
                st.success(
                    f"🚀 **Fastest riser**\n\n{safe_text(r.get('video_title'))}\n\n"
                    f"▲ {int(r.get('rank_movement'))} positions · {safe_text(r.get('region_code'))}"
                )
        with ev2:
            gr = intel["global_vs_regional"]
            cr = gr[gr["reach_type"] == "cross_region"]
            rs = gr[gr["reach_type"] == "region_specific"]
            if not cr.empty and not rs.empty:
                ratio = cr.iloc[0]["average_view_velocity"] / rs.iloc[0]["average_view_velocity"]
                st.info(
                    f"🌎 **Cross-region signal**\n\n"
                    f"{ratio:.2f}× observed view-velocity ratio"
                )
        with ev3:
            reg = intel["regional_trends"].sort_values("average_view_velocity", ascending=False)
            if not reg.empty:
                r = reg.iloc[0]
                st.info(
                    f"⚡ **Velocity-leading region**\n\n"
                    f"{r['region_code']} · {fmt_int(r['average_view_velocity'])} views/hour"
                )

        if st.session_state.get("ai_answer"):
            st.markdown(f"### {st.session_state.get('ai_mode', 'AI')} Interpretation")
            answer = st.session_state["ai_answer"]
            # Keep the page readable: show a compact preview and full report on demand.
            paragraphs = [p for p in answer.split("\n\n") if p.strip()]
            preview = "\n\n".join(paragraphs[:4])
            st.markdown(preview)
            with st.expander("View full grounded AI explanation"):
                st.markdown(answer)

        st.markdown("### Ask AI about a category")
        category_names = sorted(
            intel["category_trends"]["category_name"].dropna().unique().tolist()
        )
        ai_category = st.selectbox("Category", category_names, key="ai_category")
        st.caption(
            f"Category-specific grounded Q&A for **{ai_category}** is presented as a "
            "guided intelligence view; all numerical evidence remains Python-calculated."
        )
        st.info(
            "Suggested questions: compare this category across regions, explain its "
            "strongest momentum signals, or identify recurring channel patterns."
        )


# ---------------------------------------------------------------------
# PIPELINE HEALTH
# ---------------------------------------------------------------------
with tabs[6]:
    st.subheader("⚙️ Pipeline Health")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Duplicates", fmt_int(quality["duplicate_count"]))
    q2.metric("Complete region-days", fmt_int(quality["complete_days"]))
    q3.metric("Incomplete region-days", fmt_int(quality["incomplete_days"]))
    q4.metric(
        "Category enrichment",
        f"{quality['category_complete']:.2f}%" if pd.notna(quality["category_complete"]) else "—",
    )

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Data Quality")
        st.progress(float(max(0, min(1, quality["category_complete"] / 100))))
        st.caption(f"Category enrichment · {quality['category_complete']:.2f}%")
        st.progress(float(max(0, min(1, quality["channel_complete"] / 100))))
        st.caption(f"Channel enrichment · {quality['channel_complete']:.2f}%")

    with e2:
        st.markdown("#### Automation Architecture")
        st.code("GitHub Actions → YouTube API → S3 → Python Analytics → ML → Gemini")

    if audit.empty:
        st.warning(
            "Audit history is unavailable in this dashboard session. "
            "Data-quality metrics above are calculated directly from historical snapshots."
        )
    else:
        success = audit[audit["status"].astype(str).str.upper() == "SUCCESS"].copy()
        failed = audit[audit["status"].astype(str).str.upper() == "FAILED"].copy()
        last_success = success["ended_at"].max() if not success.empty else pd.NaT
        freshness = (
            (pd.Timestamp.now(tz="UTC") - last_success).total_seconds() / 3600
            if pd.notna(last_success) else np.nan
        )
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Successful runs", fmt_int(len(success)))
        h2.metric("Failed runs", fmt_int(len(failed)))
        h3.metric(
            "Success rate",
            f"{100 * len(success) / len(audit):.2f}%" if len(audit) else "—",
        )
        h4.metric(
            "Freshness",
            f"{freshness:.2f} h" if pd.notna(freshness) else "—",
            delta="FRESH" if pd.notna(freshness) and freshness <= FRESHNESS_HOURS else "STALE",
        )

        audit_cols = [
            c for c in [
                "run_id", "started_at", "ended_at", "status",
                "records_fetched", "records_written", "retry_count", "error_type"
            ] if c in audit.columns
        ]
        with st.expander("View recent run audits"):
            st.dataframe(
                audit.sort_values("started_at", ascending=False)[audit_cols].head(30),
                hide_index=True, use_container_width=True,
            )

    with st.expander("View missing-field counts"):
        st.dataframe(
            pd.DataFrame(
                [{"field": k, "missing_rows": v} for k, v in quality["missing"].items()]
            ),
            hide_index=True, use_container_width=True,
        )


# ---------------------------------------------------------------------
# EXPLORER
# ---------------------------------------------------------------------
with tabs[7]:
    st.subheader("🔎 Historical Data Explorer")
    history = intel["video_history"].copy()
    c1, c2, c3 = st.columns(3)
    region_filter = c1.multiselect(
        "Region", sorted(history["region_code"].dropna().unique().tolist())
    )
    category_filter = c2.multiselect(
        "Category", sorted(history["category_name"].dropna().unique().tolist())
    )
    min_rank = c3.number_input(
        "Rank at or above", min_value=1, max_value=100, value=100
    )

    filtered = history.copy()
    if region_filter:
        filtered = filtered[filtered["region_code"].isin(region_filter)]
    if category_filter:
        filtered = filtered[filtered["category_name"].isin(category_filter)]
    filtered = filtered[
        pd.to_numeric(filtered["trending_rank"], errors="coerce") <= min_rank
    ]

    explorer_cols = [
        "video_title", "channel_name", "category_name", "region_code",
        "fetch_timestamp", "trending_rank", "rank_movement",
        "view_count", "view_velocity", "engagement_rate",
    ]
    explorer_cols = [c for c in explorer_cols if c in filtered.columns]
    st.caption(f"Showing {len(filtered):,} historical observations")
    st.dataframe(
        filtered.sort_values("fetch_timestamp", ascending=False)[explorer_cols].head(500),
        hide_index=True, use_container_width=True,
    )
    st.download_button(
        "Download filtered CSV",
        data=filtered[explorer_cols].to_csv(index=False),
        file_name="youtube_historical_intelligence.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Portfolio project · Historical YouTube trending snapshots from CA, GB, IN and US · "
    "Python analytics + Random Forest + grounded Gemini interpretation"
)
