import os
import json
import logging
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

REGION_CODE = os.getenv("YOUTUBE_REGION", "US")
MAX_RESULTS = 200


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_environment():
    """
    Make sure required environment variables exist
    before calling external services.
    """

    missing = []

    if not YOUTUBE_API_KEY:
        missing.append("YOUTUBE_API_KEY")

    if not S3_BUCKET:
        missing.append("S3_BUCKET")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


# ---------------------------------------------------------
# YOUTUBE EXTRACTION
# ---------------------------------------------------------

def fetch_trending_videos(region_code):
    """
    Fetch the current top trending videos for a region.
    """

    logger.info(
        "Fetching trending videos for region=%s",
        region_code
    )

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )

    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=MAX_RESULTS
    )

    max_retries = 3

    retryable_status_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):

        try:
            response = request.execute()
            break

        except HttpError as exc:
            status_code = exc.resp.status

            if status_code not in retryable_status_codes:
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            logger.warning(
                "YouTube API request failed, retrying in %s seconds...",
                wait_time
            )

            time.sleep(wait_time)

    items = response.get("items", [])
# ---------------------------------------------------------
# SNAPSHOT ENRICHMENT
# ---------------------------------------------------------

def create_snapshot(items, region_code):
    """
    Add ingestion metadata to every video record.
    """

    fetched_at = datetime.now(timezone.utc)

    snapshot_timestamp = fetched_at.isoformat()

    enriched_items = []

    for rank, item in enumerate(items, start=1):

        enriched_item = item.copy()

        enriched_item["fetch_timestamp"] = snapshot_timestamp
        enriched_item["fetch_date"] = fetched_at.date().isoformat()
        enriched_item["region_code"] = region_code
        enriched_item["trending_rank"] = rank

        enriched_items.append(enriched_item)

    return enriched_items, fetched_at


# ---------------------------------------------------------
# DATA VALIDATION
# ---------------------------------------------------------

def validate_snapshot(records):
    """
    Perform basic quality checks before writing data to S3.
    """

    if not records:
        raise ValueError("Snapshot contains zero records.")

    required_fields = [
        "id",
        "snippet",
        "statistics",
        "fetch_timestamp",
        "region_code",
        "trending_rank"
    ]

    for index, record in enumerate(records):

        missing = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"Record {index} missing fields: {missing}"
            )

    video_ids = [
        record["id"]
        for record in records
    ]

    duplicate_count = (
        len(video_ids)
        - len(set(video_ids))
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate video IDs "
            "inside the same snapshot."
        )

    logger.info(
        "Validation passed: %s records",
        len(records)
    )


# ---------------------------------------------------------
# S3 KEY
# ---------------------------------------------------------

def build_s3_key(fetched_at, region_code):
    """
    Generate partitioned S3 path.

    Example:

    raw/trending/
        year=2026/
        month=08/
        day=20/
        region=US/
        trending_20260820T140000Z.json
    """

    timestamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")

    return (
        f"raw/trending/"
        f"year={fetched_at.year}/"
        f"month={fetched_at.month:02d}/"
        f"day={fetched_at.day:02d}/"
        f"region={region_code}/"
        f"trending_{timestamp}.json"
    )


# ---------------------------------------------------------
# LOAD TO S3
# ---------------------------------------------------------

def upload_to_s3(records, fetched_at, region_code):

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    s3_key = build_s3_key(
        fetched_at,
        region_code
    )

    payload = {
        "metadata": {
            "source": "youtube_data_api_v3",
            "region_code": region_code,
            "fetched_at": fetched_at.isoformat(),
            "record_count": len(records)
        },
        "items": records
    }

    logger.info(
        "Uploading snapshot to s3://%s/%s",
        S3_BUCKET,
        s3_key
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json"
    )

    return s3_key


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

def run_pipeline():

    validate_environment()

    try:

        items = fetch_trending_videos(
            REGION_CODE
        )

        records, fetched_at = create_snapshot(
            items,
            REGION_CODE
        )

        validate_snapshot(records)

        s3_key = upload_to_s3(
            records,
            fetched_at,
            REGION_CODE
        )

        logger.info(
            "Pipeline completed successfully."
        )

        logger.info(
            "Records uploaded: %s",
            len(records)
        )

        logger.info(
            "S3 location: s3://%s/%s",
            S3_BUCKET,
            s3_key
        )

    except HttpError as exc:

        logger.exception(
            "YouTube API request failed: %s",
            exc
        )

        raise

    except (BotoCoreError, ClientError) as exc:

        logger.exception(
            "AWS S3 operation failed: %s",
            exc
        )

        raise

    except Exception as exc:

        logger.exception(
            "Pipeline failed: %s",
            exc
        )

        raise


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()