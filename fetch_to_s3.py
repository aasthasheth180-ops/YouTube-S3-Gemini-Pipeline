import os
import json
import logging
import time
import uuid

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

REGION_CODES = os.getenv(
    "YOUTUBE_REGIONS",
    "US,IN,GB,CA"
).split(",")
MAX_RESULTS = 50


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
            f"Environment configuration failed: "
            f"Missing required variables: {', '.join(missing)}"
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


    max_retries = 3
    retry_count = 0
    retryable_status_codes = {429, 500, 502, 503, 504}

    all_items = []
    page_token = None
    MAX_PAGES = 2

    for page_number in range(MAX_PAGES):
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=50,
            pageToken=page_token
        )

        for attempt in range(max_retries):
            try:
                response = request.execute()
                break
            except HttpError as exc:
                status_code = exc.resp.status

                if status_code not in retryable_status_codes:
                    raise

                if attempt == max_retries - 1:
                    exc.retry_count = retry_count
                    raise

                retry_count += 1

                wait_time = 2 ** attempt

                logger.warning(
                    "YouTube API request failed, retrying in %s seconds...",
                    wait_time
                )

                time.sleep(wait_time)

        page_items = response.get("items", [])
        all_items.extend(page_items)

        logger.info(
            "Fetched page %s for region=%s with %s videos",
            page_number + 1,
            region_code,
            len(page_items)
        )

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    if not all_items:
        raise ValueError(
            f"YouTube API returned zero videos for region {region_code}"
        )

    return all_items, retry_count
#---------------

#---------------
def fetch_video_categories(region_code):
    """
    Fetch YouTube video category IDs and names for a region.
    """

    logger.info(
        "Fetching video categories for region=%s",
        region_code
    )

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )

    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code
    )

    response = request.execute()

    category_map = {}

    for item in response.get("items", []):
        category_id = item["id"]
        category_name = item["snippet"]["title"]

        category_map[category_id] = category_name

    return category_map

#---------------------------------------------------------

#--------------------------------------------------------
def fetch_channel_data(channel_ids):
    """
    Fetch channel-level statistics for a list of channel IDs.
    """

    logger.info(
        "Fetching channel data for %s channels",
        len(channel_ids)
    )

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )

    channel_map = {}

    for start in range(0, len(channel_ids), 50):

        batch = channel_ids[start:start + 50]

        logger.info(
            "Fetching channel batch with %s channels",
            len(batch)
        )


        request = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch)
        )

        response = request.execute()


        for item in response.get("items", []):
            channel_id = item["id"]
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            channel_map[channel_id] = {
                "channel_name": snippet.get("title"),
                "subscriber_count": statistics.get("subscriberCount"),
                "channel_view_count": statistics.get("viewCount"),
                "channel_video_count": statistics.get("videoCount"),
            }

    return channel_map
# ---------------------------------------------------------
# SNAPSHOT ENRICHMENT
# ---------------------------------------------------------

def create_snapshot(items,region_code,category_map,channel_map):
    """
    Add ingestion metadata and category enrichment
    to every video record.
    """

    fetched_at = datetime.now(timezone.utc)

    snapshot_timestamp = fetched_at.isoformat()

    enriched_items = []

    for rank, item in enumerate(items, start=1):

        enriched_item = item.copy()

        category_id = item.get(
            "snippet",
            {}
        ).get("categoryId")

        category_name = category_map.get(
            category_id,
            "Unknown"
        )
        channel_id = item.get(
            "snippet",
            {}
        ).get("channelId")

        channel_data = channel_map.get(
            channel_id,
            {}
        )
        
        enriched_item["category_name"] = category_name
        enriched_item["channel_name"] = (channel_data.get("channel_name"))
        enriched_item["channel_subscriber_count"] = (channel_data.get("subscriber_count"))
        enriched_item["channel_view_count"] = (channel_data.get("channel_view_count"))
        enriched_item["channel_video_count"] = (channel_data.get("channel_video_count"))
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
        raise ValueError("Snapshot validation failed: zero records.")

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
                f"Snapshot validation failed:"
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
def get_collection_window_start_hour(fetched_at):
    return (fetched_at.hour // 6) * 6

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
    window_start_hour = get_collection_window_start_hour(
        fetched_at
    )

    return (
        f"raw/trending/"
        f"year={fetched_at.year}/"
        f"month={fetched_at.month:02d}/"
        f"day={fetched_at.day:02d}/"
        f"region={region_code}/"
        f"window={window_start_hour:02d}/"
        f"trending_{timestamp}.json"
    )


#----------------------------------------------------

#---------------------------------------------------------
def snapshot_exists_for_window(fetched_at, region_code):

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    window_start_hour = get_collection_window_start_hour(
        fetched_at
    )


    prefix = (
        f"raw/trending/"
        f"year={fetched_at.year}/"
        f"month={fetched_at.month:02d}/"
        f"day={fetched_at.day:02d}/"
        f"region={region_code}/"
        f"window={window_start_hour:02d}/"
    )

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=prefix,
        MaxKeys=1
    )

    return response.get("KeyCount", 0) > 0
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
            "schema_version": "1.0",
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
#---------------------------------------------------------

#---------------------------------------------------------

def upload_audit_record(audit_record):

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    ended_at = datetime.fromisoformat(
        audit_record["ended_at"]
    )

    s3_key = (
        f"audit/runs/"
        f"year={ended_at.year}/"
        f"month={ended_at.month:02d}/"
        f"day={ended_at.day:02d}/"
        f"run_{audit_record['run_id']}.json"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(audit_record).encode("utf-8"),
        ContentType="application/json"
    )

    logger.info(
        "Audit record uploaded to s3://%s/%s",
        S3_BUCKET,
        s3_key
    )



# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

def run_pipeline():

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    status = "FAILED"
    records_fetched = 0
    records_written = 0
    retry_count = 0
    error_type = None
    error_message = None

    logger.info(
        "Pipeline started. run_id=%s, started_at=%s",
        run_id,
        started_at.isoformat()
    )

    try:
        
        validate_environment()

        for region_code in REGION_CODES:
            items,region_retry_count = fetch_trending_videos(
                region_code
            )

            retry_count += region_retry_count
            records_fetched += len(items)

            category_map = fetch_video_categories(region_code)

            channel_ids = list({
                item["snippet"]["channelId"]
                for item in items
                if item.get("snippet", {}).get("channelId")
            })


            channel_map = fetch_channel_data(channel_ids)

            records, fetched_at = create_snapshot(
                items,
                region_code,
                category_map,
                channel_map
            )

            validate_snapshot(records)
            if snapshot_exists_for_window(
                    fetched_at,
                    region_code
                ):
                    window_start_hour = get_collection_window_start_hour(
                        fetched_at
                        )

                    logger.warning(
                        "Snapshot already exists for date=%s region=%s window=%02d. Skipping upload.",
                        fetched_at.date(),
                        region_code,
                        window_start_hour
                    )
        
                    continue

            s3_key = upload_to_s3(
                records,
                fetched_at,
                region_code
            )

            records_written += len(records)

        status = "SUCCESS"



        logger.info(
            "Pipeline completed successfully."
        )

        logger.info("Total records uploaded: %s", records_written)


    except HttpError as exc:
        error_type = "HttpError"
        error_message = str(exc)    
        retry_count = getattr(
            exc,
            "retry_count",
            0
        )

        logger.exception(
            "YouTube API request failed: %s",
            exc
        )

        raise

    except ClientError as exc:
        error_type = "ClientError"
        error_message = str(exc)
        error_code = exc.response["Error"]["Code"]

        logger.exception(
            "AWS S3 client error. Bucket= %s ErrorCode = %s Error = %s",
            S3_BUCKET,
            error_code,
            exc
        )
        raise

    except BotoCoreError as exc:
        error_type = "BotoCoreError"
        error_message = str(exc)
        logger.exception(
            "AWS SDK error: %s",
            exc
        )
        raise

    except ValueError as exc:
        error_type = "ValueError"
        error_message = str(exc)
        logger.exception(
            "Validation or configuration failed: %s",
            exc
        )

        raise
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)

        logger.exception(
            "Pipeline failed: %s",
            exc
        )

        raise

    finally:
        ended_at = datetime.now(timezone.utc)
        audit_record = {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "status": status,
            "records_fetched": records_fetched,
            "records_written": records_written,
            "retry_count": retry_count,
            "error_type": error_type,
            "error_message": error_message,
            }

        logger.info(
            "Pipeline audit record: %s",
            audit_record
        )

        try:
            upload_audit_record(audit_record)

        except Exception as audit_exc:
            logger.exception(
                "Failed to upload pipeline audit record: %s",
                audit_exc
            )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

