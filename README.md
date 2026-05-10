
---

# Youtube Trending -S3-Gemini-Pipeline

An end-to-end Data Engineering and AI-driven analytics pipeline. This project extracts trending YouTube data into **AWS S3**, performs time-series forecasting using **Polynomial Regression**, and generates strategic insights using the **Google Gemini 2.5 Flash** model.

## 🚀 What This Project Does

* **ETL Pipeline**: Automates the extraction of raw YouTube Data API v3 JSON directly to an AWS S3 bucket.
* **Predictive Analytics**: Analyzes view and like trends with polynomial regression and an 8-week growth forecast.
* **Competitive Intelligence**: Compares channels across engagement rates, upload efficiency, and category dominance.
* **GenAI Insights**: Leverages the Gemini API for automated strategic analysis and a custom Q&A dashboard.

## 📂 Project Structure

```bash
youtube_analysis/
├── fetch_to_s3.py         # Data Engineering: Fetches YouTube API data to AWS S3
├── dashboard.py           # Streamlit application (Interactive UI + Gemini AI Layer)
├── .env                   # Environment variables (S3 bucket, API keys)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

```

## 🛠️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

```

### 2. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key, Secret Key, and Region (e.g., us-east-1)

```

### 3. Set your Gemini API Key

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY="your-gemini-api-key-here"
S3_BUCKET="your-bucket-name"

```

### 4. Run the Pipeline & Dashboard

**Fetch data to S3:**

```bash
python fetch_to_s3.py

```

**Launch the Dashboard:**

```bash
streamlit run dashboard.py

```

## 🧠 Data Processing & AI Insights

| Component | Functionality |
| --- | --- |
| **Data Extraction** | Boto3 reads partitioned JSON files from S3; Pandas flattens nested API structures. |
| **Trend Prediction** | Polynomial regression on weekly median metrics with $R^2$ accuracy tracking. |
| **Gemini AI Integration** | Sends structured data summaries to `gemini-2.5-flash` for automated reporting. |

### Dashboard Tabs

* **📈 Trend Prediction**: Interactive time-series charts with forecast toggles.
* **🏆 Competitor Comparison**: Head-to-head stats on engagement and views-per-video.
* **🤖 AI Insights**: Context-aware strategic analysis and real-time Q&A.

---

**Developed by Aastha Sheth**
