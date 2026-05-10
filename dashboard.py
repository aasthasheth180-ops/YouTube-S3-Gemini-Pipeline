"""
YouTube Trending Data Analysis Dashboard
Aastha Sheth | Portfolio Project

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
import json
import google.generativeai as genai
import os
import anthropic
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


# MUST be here, before any other logic
load_dotenv() 

# Now the variables will have values instead of 'None'
s3_bucket = os.getenv("S3_BUCKET")

# Initialize memory if it's the first time opening the app

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'channel_stats' not in st.session_state:
    st.session_state.channel_stats = None
if 'weekly' not in st.session_state:
    st.session_state.weekly = None
if 'view_r2' not in st.session_state:
    st.session_state.view_r2 = None
if 'view_preds' not in st.session_state:
    st.session_state.view_preds = None
if 'like_preds' not in st.session_state:
    st.session_state.like_preds = None
if 'view_model' not in st.session_state:
    st.session_state.view_model = None
if 'x_num' not in st.session_state:
    st.session_state.x_num = None
if 'future_views' not in st.session_state:
    st.session_state.future_views = None
if 'future_dates' not in st.session_state:
    st.session_state.future_dates = None
if 'like_r2' not in st.session_state:
    st.session_state.like_r2 = None

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Analytics + AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Keep your existing styles */
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #02809090;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .insight-box {
        background: #1E2235; /* CHANGED to dark background to match your dashboard */
        color: #FFFFFF;      /* Force text to be white so it's always visible */
        border-radius: 10px;
        padding: 1rem 1.2rem; 
        border-left: 4px solid #1E2761;
        margin: 0.5rem 0; 
        font-size: 0.93rem; 
        line-height: 1.6;
    }
    .ai-badge {
        background: linear-gradient(90deg,#1E2761,#028090);
        color: white; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    h1 { color: #1E2761; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }

    /* ADD the new selection fixes here */
    .insight-box ::selection {
        background: #FF0000;
        color: #FFFFFF;
    }
    .insight-box ::-moz-selection {
        background: #FF0000;
        color: #FFFFFF;
    }
    ::selection {
        background: #FF000066;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — CONFIG
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # 1. Keep your branding
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=120)
    st.markdown("## Project Status")

    # 2. SILENT LOADING: Pull values from .env instead of UI
    # We use 'os.getenv' to get the values you saved in your .env file
    s3_bucket     = os.getenv("S3_BUCKET")
    s3_prefix     = "raw/trending/" # Standardized folder path
    aws_region    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    google_api_key = os.getenv("GOOGLE_API_KEY") # Use your new Gemini Key

    # 3. Keep only the useful UI controls
    # These don't need to be hidden because they control the charts, not security
    top_n = st.slider("Top N Channels to Compare", 3, 15, 8)
    
    # 4. Status Indicator (Shows you it's working)
    if s3_bucket and google_api_key:
        st.success("Credentials Loaded")
    else:
        st.error("Missing .env keys")

    st.divider()
    
    # 5. The Action Button
    load_btn = st.button("Load & Analyze", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Aastha Sheth** \nData Analysis Portfolio \n[GitHub](https://github.com/aasthasheth180-ops)")
# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
CAT_MAP = {'1':'Film & Animation','2':'Autos & Vehicles','10':'Music',
           '15':'Pets & Animals','17':'Sports','20':'Gaming',
           '22':'People & Blogs','23':'Comedy','24':'Entertainment',
           '25':'News & Politics','26':'How-to & Style','27':'Education',
           '28':'Science & Tech','29':'Non-profits'}

@st.cache_data(ttl=1800, show_spinner=False)
def load_from_s3(bucket, prefix, region):
    s3 = boto3.client('s3', region_name=region)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    records = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.json'):
            raw  = s3.get_object(Bucket=bucket, Key=obj['Key'])
            data = json.loads(raw['Body'].read().decode('utf-8'))
            items = data if isinstance(data, list) else data.get('items', [])
            records.extend(items)
    return records

def parse_records(records):
    rows = []
    for r in records:
        sn = r.get('snippet', {})
        st_ = r.get('statistics', {})
        rows.append({
            'video_id'      : r.get('id',''),
            'title'         : sn.get('title',''),
            'channel_title' : sn.get('channelTitle',''),
            'channel_id'    : sn.get('channelId',''),
            'published_at'  : sn.get('publishedAt',''),
            'category_id'   : sn.get('categoryId',''),
            'view_count'    : int(st_.get('viewCount', 0)),
            'like_count'    : int(st_.get('likeCount', 0)),
            'comment_count' : int(st_.get('commentCount', 0)),
            'fetch_date'    : r.get('fetch_date', pd.Timestamp.today().strftime('%Y-%m-%d')),
        })
    df = pd.DataFrame(rows)
    df['published_at']  = pd.to_datetime(df['published_at'], errors='coerce')
    df['fetch_date']    = pd.to_datetime(df['fetch_date'], errors='coerce')
    df['engagement_rate'] = ((df['like_count'] + df['comment_count'])
                              / df['view_count'].replace(0, np.nan) * 100).round(3)
    df['category'] = df['category_id'].astype(str).map(CAT_MAP).fillna('Other')
    df = df.dropna(subset=['view_count','published_at']).drop_duplicates('video_id')
    df = df[df['view_count'] > 0].reset_index(drop=True)
    return df

def compute_channel_stats(df, top_n):
    return (df.groupby('channel_title')
              .agg(total_views    =('view_count','sum'),
                   median_views   =('view_count','median'),
                   total_likes    =('like_count','sum'),
                   avg_engagement =('engagement_rate','mean'),
                   video_count    =('video_id','count'),
                   top_category   =('category', lambda x: x.mode()[0]))
              .reset_index()
              .assign(views_per_video=lambda d: (d.total_views/d.video_count).round(0))
              .sort_values('total_views', ascending=False)
              .head(top_n)
              .reset_index(drop=True))

def fit_trend(x_dates, y_values, degree=3):
    x_num = (x_dates - x_dates.min()).dt.days.values.reshape(-1,1)
    model = Pipeline([('poly', PolynomialFeatures(degree)),('reg', LinearRegression())])
    model.fit(x_num, y_values)
    preds = model.predict(x_num)
    return preds, r2_score(y_values, preds), model, x_num

def forecast(model, x_num, weeks=8):
    last_x = x_num[-1][0]
    future_x = np.array([[last_x + 7*i] for i in range(1, weeks+1)])
    return model.predict(future_x)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────


st.title("YouTube Trending Analysis + AI Insights")
st.markdown("**Competitor channel comparison · Trend prediction · Gemini-powered insights**")

# ── NEW DATA LOADING LOGIC (Paste here) ────────────────────────────────────────────────
if load_btn:
    with st.spinner("Loading data from S3..."):
        try:
            records = load_from_s3(s3_bucket, s3_prefix, aws_region)
            df = parse_records(records)
            channel_stats = compute_channel_stats(df, top_n)

            # Clean week column for plotly (Fixes the messy X-axis)
            df['week'] = (df['published_at']
                          .dt.to_period('W')
                          .dt.start_time
                          .dt.normalize()
                          .dt.date)
            df['week'] = pd.to_datetime(df['week'])

            weekly = (df.groupby('week')
                        .agg(avg_views  =('view_count', 'median'),
                             avg_likes  =('like_count',  'median'),
                             video_count=('video_id',    'count'))
                        .reset_index()
                        .sort_values('week'))
            
            has_enough = len(weekly) >= 4

            # Trend Fitting Logic
            if has_enough:
                view_preds, view_r2, view_model, x_num = fit_trend(weekly['week'], weekly['avg_views'])
                like_preds, like_r2, _,           _    = fit_trend(weekly['week'], weekly['avg_likes'])
                future_dates = pd.date_range(weekly['week'].max() + pd.Timedelta('7D'), periods=8, freq='W')
                future_views = forecast(view_model, x_num, 8)
            else:
                view_preds, like_preds = weekly['avg_views'].values, weekly['avg_likes'].values
                view_r2, like_r2 = float('nan'), float('nan')
                future_views = np.array([weekly['avg_views'].iloc[-1]] * 8)
                future_dates = pd.date_range(weekly['week'].max() + pd.Timedelta('7D'), periods=8, freq='W')

            # --- CRITICAL: SAVE TO SESSION STATE ---
            st.session_state.df            = df
            st.session_state.channel_stats = channel_stats
            st.session_state.data_loaded   = True
            st.session_state.weekly        = weekly
            st.session_state.view_r2       = view_r2
            st.session_state.future_views  = future_views
            st.session_state.future_dates  = future_dates
            st.session_state.like_r2 = like_r2
            st.session_state.like_preds = like_preds

            st.success(f"Loaded {len(df):,} videos")
        except Exception as e:
            st.error(f"S3 load failed: {e}")
            st.stop()

# ── GATE: Check if data is in memory ──
if not st.session_state.get('data_loaded', False):
    st.info("Configure credentials in the sidebar, then click **Load & Analyze**.")
    st.stop()

# ── RETRIEVE: Pull data back out for the Tabs ──
df            = st.session_state.df
channel_stats = st.session_state.channel_stats
weekly        = st.session_state.weekly
view_preds    = st.session_state.get('view_preds')
view_r2       = st.session_state.view_r2
future_views  = st.session_state.future_views
future_dates  = st.session_state.future_dates
like_preds = st.session_state.get('like_preds', np.array([]))
like_r2    = st.session_state.get('like_r2', 0.0) # Fallback to 0.0 instead of None
has_enough = st.session_state.get('has_enough', False)

top_n          = len(channel_stats) if channel_stats is not None else 8


# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
 
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Videos",    f"{len(df):,}")
k2.metric("Unique Channels", f"{df['channel_title'].nunique():,}")
k3.metric("Median Views",    f"{df['view_count'].median()/1e3:.1f}K")
k4.metric("Avg Engagement",  f"{df['engagement_rate'].mean():.2f}%")
 
# Safe R² display — show "—" when not enough data instead of nan
r2_display = f"{view_r2:.3f}" if (view_r2 == view_r2) else "—"   # nan != nan is True
r2_delta   = ("Strong" if view_r2 > 0.7
               else "Moderate" if view_r2 > 0.4
               else "Weak" if view_r2 == view_r2
               else "Need 4+ weeks")
k5.metric("View Trend R²", r2_display, delta=r2_delta)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Prediction",
    "Competitor Comparison",
    "AI Insights",
    "Raw Data"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — TREND PREDICTION
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# TAB 1 — TREND PREDICTION  (FIXED)
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("View & Like Trends Over Time")
 
    # ── GUARD: need at least 4 weeks to draw a meaningful trend ──
    has_enough_data = len(weekly) >= 4
 
    col1, col2 = st.columns([3, 1])
    with col2:
        show_forecast = st.checkbox("Show 8-week forecast", value=True)
        show_ci       = st.checkbox("Show confidence band",  value=True)
        metric_sel    = st.radio("Metric", ["Views", "Likes", "Both"], index=0)
 
    with col1:
        if not has_enough_data:
            st.info(
        f"Only **{len(weekly)} week(s)** of data. "
        "Run `fetch_to_s3.py` daily for 4+ weeks to unlock trend lines and forecasts."
            )
            if len(weekly) > 0:
                weekly_plot = weekly.copy()
                weekly_plot['week_label'] = pd.to_datetime(weekly_plot['week']).dt.strftime('%b %d, %Y')
 
            fig_bar = go.Figure(go.Bar(
                x=weekly_plot['week_label'],   # string labels, not datetime — avoids x-axis bug
                y=weekly_plot['avg_views'],
                marker_color='#FF0000',
                text=weekly_plot['avg_views'].apply(
                    lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"),
                textposition='outside'
            ))
            fig_bar.update_layout(
                height=350,
                title="Weekly Median Views (collecting history — run fetch daily)",
                template='plotly_dark',
                paper_bgcolor='#1E2235',
                plot_bgcolor='#1E2235',
                xaxis_title="Week",
                yaxis_title="Median Views",
                showlegend=False,
                yaxis=dict(tickformat='.2s', showgrid=True, gridcolor='#2D3348'),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # ── full trend chart ──
            fig = go.Figure()
 
            if metric_sel in ["Views", "Both"]:
                fig.add_trace(go.Scatter(
                    x=weekly['week'], y=weekly['avg_views'],
                    mode='lines+markers', name='Actual Views',
                    line=dict(color='#FFFFFF', width=2),
                    marker=dict(size=5, color='#FFFFFF')))
                fig.add_trace(go.Scatter(
                    x=weekly['week'], y=view_preds,
                    mode='lines', name=f'Trend (R²={view_r2:.2f})',
                    line=dict(color='#02C39A', width=2.5, dash='dash')))
                if show_forecast:
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=future_views,
                        mode='lines+markers', name='8-wk Forecast',
                        line=dict(color='#F59E0B', width=2, dash='dot'),
                        marker=dict(size=6, symbol='square', color='#F59E0B')))
                    if show_ci:
                        fig.add_trace(go.Scatter(
                            x=list(future_dates) + list(future_dates[::-1]),
                            y=list(future_views * 1.15) + list((future_views * 0.85)[::-1]),
                            fill='toself',
                            fillcolor='rgba(245,158,11,0.12)',
                            line=dict(color='rgba(0,0,0,0)'),
                            name='±15% CI',
                            showlegend=True))
 
            if metric_sel in ["Likes", "Both"]:
                fig.add_trace(go.Scatter(
                    x=weekly['week'], y=weekly['avg_likes'],
                    mode='lines+markers', name='Actual Likes',
                    line=dict(color='#A78BFA', width=2),
                    marker=dict(size=5, color='#A78BFA')))
                fig.add_trace(go.Scatter(
                    x=weekly['week'], y=like_preds,
                    mode='lines', name=f'Like Trend (R²={like_r2:.2f})',
                    line=dict(color='#EC4899', width=2.5, dash='dash')))
 
            fig.update_layout(
                height=420,
                template='plotly_dark',
                paper_bgcolor='#1E2235',
                plot_bgcolor='#1E2235',
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.25, font=dict(size=11)),
                xaxis=dict(showgrid=True, gridcolor='#2D3348'),
                yaxis=dict(showgrid=True, gridcolor='#2D3348',
                           tickformat='.2s')  # auto K/M formatting
            )
            st.plotly_chart(fig, use_container_width=True)
 
    # ── 8-Week Forecast Table ─────────────────────────────────
    if show_forecast and has_enough_data:
        st.markdown("**8-Week View Forecast**")
 
        # FIX: direction was broken because enumerate starts at 0
        # future_views[i-1] when i=0 wraps to future_views[-1] (last element!)
        # Correct logic: compare each week to the previous week properly
        directions = []
        for i, v in enumerate(future_views):
            if i == 0:
                # compare first forecast week to last actual week
                prev = weekly['avg_views'].iloc[-1]
            else:
                prev = future_views[i - 1]
            directions.append("▲ Up" if v > prev else "▼ Down" if v < prev else "→ Flat")
 
        forecast_df = pd.DataFrame({
            'Week'           : future_dates.strftime('%b %d, %Y'),
            'Forecast Views' : [f"{v:,.0f}" for v in future_views],
            'vs Prev Week'   : directions
        })
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)
 
    elif show_forecast and not has_enough_data:
        st.caption("⏳ Forecast will appear once 4+ weeks of data are available.")
 
    # ── Views by Category ────────────────────────────────────
    st.subheader("Views by Category")
 
    # Check if we have enough date spread for a line chart
    n_weeks = df['published_at'].dt.to_period('W').nunique()
 
    if n_weeks >= 3:
        df['week_dt'] = df['published_at'].dt.to_period('W').dt.start_time
        top_cats = df['category'].value_counts().head(6).index
        cat_weekly = (df[df['category'].isin(top_cats)]
                      .groupby(['week_dt', 'category'])['view_count']
                      .median().reset_index())
        fig_cat = px.line(
            cat_weekly, x='week_dt', y='view_count',
            color='category',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'view_count': 'Median Views', 'week_dt': 'Week', 'category': 'Category'}
        )
        fig_cat.update_layout(
            height=360,
            paper_bgcolor='#1E2235',
            plot_bgcolor='#1E2235',
            legend=dict(orientation='h', y=-0.25),
            xaxis=dict(showgrid=True, gridcolor='#2D3348'),
            yaxis=dict(showgrid=True, gridcolor='#2D3348', tickformat='.2s')
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        # ── fallback: bar chart of total views by category (works with 1 day of data)
        cat_totals = (df.groupby('category')['view_count']
                      .sum().sort_values(ascending=False).head(8).reset_index())
        fig_cat = px.bar(
            cat_totals, x='view_count', y='category',
            orientation='h',
            template='plotly_dark',
            color='view_count',
            color_continuous_scale='Teal',
            labels={'view_count': 'Total Views', 'category': 'Category'},
            title="Total Views by Category (weekly trend available after 3+ weeks)"
        )
        fig_cat.update_layout(
            height=380,
            paper_bgcolor='#1E2235',
            plot_bgcolor='#1E2235',
            showlegend=False,
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig_cat.update_traces(
            text=cat_totals['view_count'].apply(
                lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"),
            textposition='outside'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════
# TAB 2 — COMPETITOR COMPARISON  (FIXED)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Top {top_n} Channels — Competitor Analysis")
 
    colors_list = px.colors.qualitative.Set2
    channels    = channel_stats['channel_title'].tolist()
 
    # ── ROW 1: Total Views + Engagement side by side ──────────
    r1c1, r1c2 = st.columns(2)
 
    with r1c1:
        fig_views = go.Figure(go.Bar(
            x=channel_stats['total_views'],
            y=channels,
            orientation='h',
            marker_color=colors_list[:len(channels)],
            text=channel_stats['total_views'].apply(
                lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"),
            textposition='outside'
        ))
        fig_views.update_layout(
            title="Total Views",
            height=380, template='plotly_dark',
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            showlegend=False, margin=dict(l=10, r=60, t=40, b=10),
            xaxis=dict(tickformat='.2s', showgrid=True, gridcolor='#2D3348'),
            yaxis=dict(autorange='reversed', showgrid=False)
        )
        st.plotly_chart(fig_views, use_container_width=True)
 
    with r1c2:
        fig_eng = go.Figure(go.Bar(
            x=channel_stats['avg_engagement'].round(2),
            y=channels,
            orientation='h',
            marker_color=colors_list[:len(channels)],
            text=channel_stats['avg_engagement'].round(2).astype(str) + '%',
            textposition='outside'
        ))
        fig_eng.update_layout(
            title="Avg Engagement Rate (%)",
            height=380, template='plotly_dark',
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            showlegend=False, margin=dict(l=10, r=60, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor='#2D3348', ticksuffix='%'),
            yaxis=dict(autorange='reversed', showgrid=False)
        )
        st.plotly_chart(fig_eng, use_container_width=True)
 
    # ── ROW 2: Views per Video + Video Count ──────────────────
    r2c1, r2c2 = st.columns(2)
 
    with r2c1:
        fig_eff = go.Figure(go.Bar(
            x=channel_stats['views_per_video'],
            y=channels,
            orientation='h',
            marker_color=colors_list[:len(channels)],
            text=channel_stats['views_per_video'].apply(
                lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"),
            textposition='outside'
        ))
        fig_eff.update_layout(
            title="Views per Video (Efficiency)",
            height=380, template='plotly_dark',
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            showlegend=False, margin=dict(l=10, r=60, t=40, b=10),
            xaxis=dict(tickformat='.2s', showgrid=True, gridcolor='#2D3348'),
            yaxis=dict(autorange='reversed', showgrid=False)
        )
        st.plotly_chart(fig_eff, use_container_width=True)
 
    with r2c2:
        fig_cnt = go.Figure(go.Bar(
            x=channel_stats['video_count'],
            y=channels,
            orientation='h',
            marker_color=colors_list[:len(channels)],
            text=channel_stats['video_count'].astype(str) + ' videos',
            textposition='outside'
        ))
        fig_cnt.update_layout(
            title="Video Count (videos fetched)",
            height=380, template='plotly_dark',
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            showlegend=False, margin=dict(l=10, r=60, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor='#2D3348'),
            yaxis=dict(autorange='reversed', showgrid=False)
        )
        # Add note if all channels have video_count = 1
        if channel_stats['video_count'].max() == 1:
            st.caption(
                "⚠️ Each channel shows 1 video — YouTube's `mostPopular` endpoint returns "
                "the top trending video per run. Run `fetch_to_s3.py` daily to accumulate more."
            )
        st.plotly_chart(fig_cnt, use_container_width=True)
 
    # ── Monthly Trend Lines ───────────────────────────────────
    st.subheader("Monthly View Trends — Head to Head")
 
    n_weeks_ch = df[df['channel_title'].isin(channels)]['published_at'].dt.to_period('W').nunique()
 
    if n_weeks_ch >= 3:
        df['week_dt'] = df['published_at'].dt.to_period('W').dt.start_time
        ch_weekly = (df[df['channel_title'].isin(channels)]
                     .groupby(['week_dt', 'channel_title'])['view_count']
                     .median().reset_index())
        fig_trend = px.line(
            ch_weekly, x='week_dt', y='view_count',
            color='channel_title',
            template='plotly_dark',
            color_discrete_sequence=colors_list,
            labels={'view_count': 'Median Views', 'channel_title': 'Channel', 'week_dt': 'Week'}
        )
        fig_trend.update_layout(
            height=400,
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            legend=dict(orientation='h', y=-0.25),
            xaxis=dict(showgrid=True, gridcolor='#2D3348'),
            yaxis=dict(showgrid=True, gridcolor='#2D3348', tickformat='.2s')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(
            "📅 Head-to-head trend lines need **3+ weeks** of data. "
            "Currently showing a snapshot comparison instead."
        )
        # Snapshot bar — useful even with 1 day of data
        fig_snap = px.bar(
            channel_stats.sort_values('total_views', ascending=False),
            x='channel_title', y='total_views',
            color='channel_title',
            template='plotly_dark',
            color_discrete_sequence=colors_list,
            labels={'total_views': 'Total Views', 'channel_title': 'Channel'},
            title="Current Snapshot — Total Views by Channel"
        )
        fig_snap.update_layout(
            height=380,
            paper_bgcolor='#1E2235', plot_bgcolor='#1E2235',
            showlegend=False,
            xaxis=dict(tickangle=-30),
            yaxis=dict(tickformat='.2s')
        )
        st.plotly_chart(fig_snap, use_container_width=True)
 
    # ── Stats Table ───────────────────────────────────────────
    st.subheader("Channel Stats Table")
    display_df = channel_stats.copy()
    display_df['total_views']     = display_df['total_views'].apply(
        lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    display_df['avg_engagement']  = display_df['avg_engagement'].apply(lambda x: f"{x:.2f}%")
    display_df['views_per_video'] = display_df['views_per_video'].apply(
        lambda x: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    st.dataframe(
        display_df[['channel_title', 'total_views', 'avg_engagement',
                    'views_per_video', 'video_count', 'top_category']],
        hide_index=True, use_container_width=True,
        column_config={
            'channel_title'  : st.column_config.TextColumn("Channel"),
            'total_views'    : st.column_config.TextColumn("Total Views"),
            'avg_engagement' : st.column_config.TextColumn("Engagement Rate"),
            'views_per_video': st.column_config.TextColumn("Views / Video"),
            'video_count'    : st.column_config.NumberColumn("Videos Fetched", format="%d"),
            'top_category'   : st.column_config.TextColumn("Top Category"),
        }
    )
# ══════════════════════════════════════════════════════════════
# TAB 3 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="ai-badge">Powered by Gemini 1.5 Flash</span>',
                unsafe_allow_html=True)
    st.subheader("AI-Powered Analysis")
 
    if not google_api_key:
        st.warning("Enter your Gemini API key in the sidebar to enable AI insights.")
        st.stop()
 
    # ── FIX 1: Much richer summary so Gemini can give real answers ──
    def build_summary():
        top5 = channel_stats.head(5)
 
        # Per-category stats
        cat_stats = (df.groupby('category')
                       .agg(total_views   =('view_count', 'sum'),
                            median_views  =('view_count', 'median'),
                            video_count   =('video_id',   'count'),
                            avg_engagement=('engagement_rate', 'mean'))
                       .sort_values('total_views', ascending=False)
                       .head(6)
                       .round(1))
 
        # Top 3 individual videos
        top_videos = (df.sort_values('view_count', ascending=False)
                        [['title', 'channel_title', 'category',
                          'view_count', 'like_count', 'engagement_rate']]
                        .head(3)
                        .to_string(index=False))
 
        # Weekly trend summary (if available)
        if len(weekly) >= 2:
            first_week_views = weekly['avg_views'].iloc[0]
            last_week_views  = weekly['avg_views'].iloc[-1]
            pct_change = ((last_week_views - first_week_views)
                          / first_week_views * 100) if first_week_views > 0 else 0
            trend_summary = (
                f"Views changed {pct_change:+.1f}% from first to latest week "
                f"({first_week_views:,.0f} → {last_week_views:,.0f} median views). "
                f"Trend R²: {view_r2:.3f} ({'strong' if view_r2 > 0.7 else 'moderate' if view_r2 > 0.4 else 'weak'} fit). "
                f"8-week forecast direction: {'upward ↑' if has_enough and future_views[-1] > future_views[0] else 'downward ↓' if has_enough else 'not enough data yet'}."
            )
        else:
            trend_summary = "Only 1 week of data collected so far — trend not yet meaningful."
 
        # Engagement leaders
        eng_leaders = (df.groupby('channel_title')
                         .filter(lambda x: len(x) >= 1)
                         .groupby('channel_title')['engagement_rate']
                         .mean()
                         .sort_values(ascending=False)
                         .head(5)
                         .round(2)
                         .to_string())
 
        return f"""
=== YOUTUBE TRENDING DATA ANALYSIS ===
Dataset: {len(df)} trending videos | {df['channel_title'].nunique()} unique channels
Collection date range: {df['published_at'].min().date()} to {df['published_at'].max().date()}
 
--- OVERALL METRICS ---
Median views per video : {df['view_count'].median():,.0f}
Mean views per video   : {df['view_count'].mean():,.0f}
Median likes per video : {df['like_count'].median():,.0f}
Avg engagement rate    : {df['engagement_rate'].mean():.2f}%
Most common category   : {df['category'].mode()[0]}
 
--- TREND ANALYSIS ---
{trend_summary}
 
--- TOP 5 CHANNELS BY TOTAL VIEWS ---
{top5[['channel_title','total_views','avg_engagement','views_per_video','video_count','top_category']].to_string(index=False)}
 
--- TOP 5 ENGAGEMENT RATE LEADERS ---
{eng_leaders}
 
--- CATEGORY PERFORMANCE (top 6) ---
{cat_stats[['total_views','median_views','video_count','avg_engagement']].to_string()}
 
--- TOP 3 INDIVIDUAL VIDEOS ---
{top_videos}
"""
 
    # ── Analysis type selector ────────────────────────────────
    analysis_type = st.radio(
        "Choose analysis focus:",
        ["Full Analysis", "Trend Deep-Dive", "Competitor Strategy"],
        horizontal=True
    )
 
    if st.button("Generate AI Insights", type="primary"):
        summary = build_summary()
        type_map = {
            "Full Analysis"      : "full",
            "Trend Deep-Dive"    : "trends",
            "Competitor Strategy": "competitors"
        }
        selected = type_map[analysis_type]
 
        # FIX: much more specific prompts with explicit output format
        prompts = {
            "full": f"""You are a senior YouTube data analyst.
Analyze the data below and respond with EXACTLY these 5 sections:
 
**1. KEY FINDINGS**
- [3 bullet points with specific numbers from the data]
 
**2. CATEGORY INSIGHTS**
- [Which categories dominate and why, with view/engagement numbers]
 
**3. TREND INTERPRETATION**
- [What the view trend means, is growth accelerating or slowing?]
 
**4. TOP CHANNEL STRATEGY**
- [What makes the #1 channel win — views vs engagement tradeoff]
 
**5. ACTIONABLE RECOMMENDATIONS**
- [3 specific tactics a new creator should copy, with data backing]
 
DATA:
{summary}""",
 
            "trends": f"""You are a data analyst specializing in YouTube growth.
Answer these 4 questions using ONLY the numbers in the data:
 
**Q1. Is overall viewership growing or declining?**
[Use the trend R² and % change figures. Be specific.]
 
**Q2. Which category is growing fastest?**
[Compare category view totals and video counts from the data.]
 
**Q3. Is the 8-week forecast reliable?**
[Comment specifically on the R² score — is it strong enough to trust?]
 
**Q4. What should a creator do right now?**
[3 specific actions based on the trend data above.]
 
DATA:
{summary}""",
 
            "competitors": f"""You are a competitive intelligence analyst for YouTube.
Answer these 4 questions using the channel data provided:
 
**Q1. Which channel has the best overall strategy?**
[Compare top channel's views vs engagement — high views + low engagement vs high engagement + fewer views]
 
**Q2. Who is punching above their weight?**
[Find the channel with highest engagement rate despite fewer views — this is the hidden threat]
 
**Q3. What content categories are underserved?**
[Look at category breakdown — which has few videos but high median views per video?]
 
**Q4. Two tactics to steal immediately**
[Pick 2 specific strategies from top performers a new channel should copy]
 
DATA:
{summary}"""
        }
 
        with st.spinner("Gemini is analyzing your data..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('models/gemini-2.5-flash')
                response = model.generate_content(prompts[selected])
                insight  = response.text
 
                # Render with proper markdown (not raw HTML replace)
                st.markdown("---")
                st.markdown(insight)
                st.markdown("---")
 
                st.download_button(
                    "⬇️ Download Insights",
                    data=json.dumps({
                        "analysis_type": analysis_type,
                        "insight"      : insight,
                        "data_summary" : summary
                    }, indent=2),
                    file_name="ai_insights.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Gemini API error: {e}")
 
    # ── Custom Question ───────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask a Custom Question")
    user_q = st.text_input(
        "Ask anything about your YouTube data:",
        placeholder="e.g. Which channel has the best growth trajectory?"
    )
 
    if user_q and st.button("Ask Gemini"):
        summary = build_summary()
        with st.spinner("Thinking..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel('models/gemini-2.5-flash')
 
                # Better prompt — forces specific answer using the data
                full_prompt = f"""You are a YouTube data analyst.
Answer the question below using ONLY the data provided.
Be specific — cite actual numbers. Keep answer under 150 words.
 
DATA:
{summary}
 
QUESTION: {user_q}
 
ANSWER:"""
 
                response = model.generate_content(full_prompt)
 
                # Use st.markdown — not HTML — so selected text is always readable
                st.markdown("**Gemini says:**")
                st.info(response.text)   # st.info gives a styled box with proper contrast
 
            except Exception as e:
                st.error(f"Gemini API error: {e}")
 

# ══════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Raw Data Explorer")
    col1, col2, col3 = st.columns(3)
    cat_filter = col1.multiselect("Filter by Category", df['category'].unique(), default=[])
    ch_filter  = col2.multiselect("Filter by Channel",  df['channel_title'].unique()[:20], default=[])
    min_views  = col3.number_input("Min Views", value=0, step=1000)

    filtered = df.copy()
    if cat_filter: filtered = filtered[filtered['category'].isin(cat_filter)]
    if ch_filter:  filtered = filtered[filtered['channel_title'].isin(ch_filter)]
    filtered = filtered[filtered['view_count'] >= min_views]

    st.caption(f"Showing {len(filtered):,} videos")
    st.dataframe(
        filtered[['title','channel_title','category','published_at',
                  'view_count','like_count','engagement_rate']]
                .sort_values('view_count', ascending=False),
        use_container_width=True, hide_index=True
    )
    st.download_button("Download CSV",
        data=filtered.to_csv(index=False),
        file_name="youtube_analysis.csv", mime="text/csv")
