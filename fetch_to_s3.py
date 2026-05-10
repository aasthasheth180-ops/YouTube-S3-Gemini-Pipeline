import os
import json
import datetime
import boto3
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load all keys from your .env file
load_dotenv()

# --- CONFIGURATION ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
S3_BUCKET       = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION      = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def fetch_and_upload():
    # 1. Initialize YouTube Client
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # 2. Fetch Trending Videos
    print("Fetching trending videos from YouTube...")
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode="US",
        maxResults=50
    )
    response = request.execute()

    # 3. Initialize S3 Client
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    # 4. Create a unique filename based on today's date
    today = datetime.date.today().isoformat()
    s3_key = f"raw/trending/trending_{today}.json"

    # 5. Upload to S3
    print(f"Uploading to S3: {S3_BUCKET}/{s3_key}")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(response),
        ContentType='application/json'
    )
    
    print("Success! Data is now in your S3 bucket.")

if __name__ == "__main__":
    fetch_and_upload()