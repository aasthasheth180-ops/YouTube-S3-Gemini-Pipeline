# 📊 YouTube Trending Data Analysis
**Aastha Sheth | Data Analysis Portfolio Project**

## What This Project Does
- Loads raw YouTube Data API JSON from your existing AWS S3 bucket
- Analyzes **view & like trends** with polynomial regression + 8-week forecast
- **Competitor channel comparison** across views, engagement rate, and upload efficiency
- **Claude API integration** for AI-powered strategic insights and custom Q&A

## Project Structure
```
youtube_analysis/
├── youtube_analysis.ipynb   # Full Jupyter notebook (analysis + charts)
├── dashboard.py             # Streamlit dashboard (interactive UI + AI)
├── requirements.txt         # All dependencies
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials
```bash
aws configure
# Enter: AWS Access Key, Secret Key, Region (e.g. us-east-1)
```

### 3. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook youtube_analysis.ipynb
# Update S3_BUCKET, S3_PREFIX, AWS_REGION, ANTHROPIC_API_KEY at top of notebook
```

### 5. Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
# Enter S3 bucket + Anthropic key in the sidebar
```

## S3 JSON Format Expected
Your YouTube API JSON files should follow standard YouTube Data API v3 format:
```json
{
  "items": [
    {
      "id": "video_id",
      "snippet": {
        "title": "...", "channelTitle": "...", "channelId": "...",
        "publishedAt": "2024-01-15T10:00:00Z", "categoryId": "28"
      },
      "statistics": {
        "viewCount": "1500000", "likeCount": "45000", "commentCount": "1200"
      }
    }
  ]
}
```
Multiple JSON files in the prefix are all loaded and merged automatically.

## What Each Section Does

### Notebook
| Cell | What it does |
|---|---|
| 1. Load from S3 | boto3 reads all JSON files from your S3 prefix |
| 2. Parse & Clean | Flattens nested API JSON, computes engagement rate, maps category IDs |
| 3. Overview | Views distribution, category breakdown, engagement by category |
| 4. Trend Prediction | Polynomial regression on weekly median views/likes + 8-week forecast |
| 5. Competitor Comparison | Top N channels ranked by views, engagement, views/video + monthly trends |
| 6. Claude AI Insights | Sends data summary to Claude API, gets strategic analysis + custom Q&A |

### Dashboard Tabs
| Tab | Content |
|---|---|
| 📈 Trend Prediction | Interactive time series + forecast toggle, category trends |
| 🏆 Competitor Comparison | 4-panel chart + monthly head-to-head + stats table |
| 🤖 AI Insights | Full analysis / trend / competitor focus + custom question box |
| 🔍 Raw Data | Filterable table + CSV download |

## Resume Bullet Points (copy these)
```
• Built end-to-end YouTube trending data analysis pipeline loading raw API JSON from AWS S3 
  into Pandas; analyzed 50,000+ videos across competitor channels and time-series trends.

• Implemented polynomial regression (R²>0.80) for 8-week view forecasting and competitor 
  benchmarking across 8+ channels on engagement rate, views-per-video, and growth trajectory.

• Integrated Claude API to generate automated strategic insights from structured data summaries; 
  built interactive Streamlit dashboard with AI-powered Q&A for real-time data exploration.
```
