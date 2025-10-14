"""Utility for downloading Singapore LTA traffic camera images on a schedule.

This script polls the Land Transport Authority (LTA) Traffic Images v2 API at a
fixed interval and downloads images for the set of cameras described in a CSV
file. The default behaviour is to follow the `reference/camera_info.csv`
configuration from this project and continue the polling loop for a full week.

Images are only fetched during the active window of 05:00-24:00 Singapore time
by default and can optionally be synchronised to an AWS S3 bucket after each
download.

Example usage:

```
python scripts/fetch_lta_camera_images.py \
    --api-key $LTA_API_KEY \
    --camera-csv reference/camera_info.csv \
    --output-dir data/lta_images \
    --interval-minutes 5
```

The script stores the images in `<output-dir>/<camera-id>/<timestamp>.jpg`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for Python <3.9
    ZoneInfo = None  # type: ignore[assignment]

import requests


LOGGER = logging.getLogger(__name__)
LTA_TRAFFIC_IMAGES_URL = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
SINGAPORE_TZ = ZoneInfo("Asia/Singapore") if ZoneInfo is not None else None
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class Camera:
    """Represents the basic information required to poll a camera."""

    camera_id: str
    latitude: float | None = None
    longitude: float | None = None


def load_cameras(csv_path: Path) -> List[Camera]:
    """Load camera information from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with at least a `CameraID` column.
    """

    cameras: List[Camera] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "CameraID" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'CameraID' column")

        for row in reader:
            camera_id = row["CameraID"].strip()
            if not camera_id:
                continue
            try:
                latitude = float(row.get("Latitude")) if row.get("Latitude") else None
                longitude = float(row.get("Longitude")) if row.get("Longitude") else None
            except ValueError:
                LOGGER.warning("Invalid coordinates for camera %s; ignoring lat/lon", camera_id)
                latitude = longitude = None
            cameras.append(Camera(camera_id=camera_id, latitude=latitude, longitude=longitude))

    if not cameras:
        raise ValueError(f"No camera entries found in {csv_path}")

    return cameras


def fetch_camera_metadata(session: requests.Session, api_key: str) -> Sequence[Dict[str, object]]:
    """Fetch the current metadata for all LTA cameras."""

    headers = {
        "AccountKey": api_key,
        "accept": "application/json",
    }
    response = session.get(LTA_TRAFFIC_IMAGES_URL, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "value" not in data:
        raise RuntimeError("Unexpected response format from LTA API: missing 'value'")
    return data["value"]


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_time_of_day(value: str) -> int:
    """Convert an HH:MM string into seconds since midnight."""

    value = value.strip()
    if value == "24:00":
        return SECONDS_PER_DAY

    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError("Time must be in HH:MM format") from exc

    if hour == 24 and minute == 0:
        return SECONDS_PER_DAY
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise argparse.ArgumentTypeError("Hour must be 0-24 and minute 0-59")
    return hour * 60 * 60 + minute * 60


def within_active_window(current_seconds: int, start_seconds: int, end_seconds: int) -> bool:
    """Return True if the current second-of-day is inside the active window."""

    if start_seconds <= end_seconds:
        return start_seconds <= current_seconds < end_seconds
    # Window wraps past midnight.
    return current_seconds >= start_seconds or current_seconds < end_seconds


def seconds_until_window(current_seconds: int, start_seconds: int, end_seconds: int) -> int:
    """Return seconds until the next active window begins."""

    if within_active_window(current_seconds, start_seconds, end_seconds):
        return 0
    if start_seconds <= end_seconds:
        if current_seconds < start_seconds:
            return start_seconds - current_seconds
        return SECONDS_PER_DAY - current_seconds + start_seconds
    # Wraps past midnight.
    if current_seconds < start_seconds:
        return start_seconds - current_seconds
    return SECONDS_PER_DAY - current_seconds + start_seconds


def download_image(
    session: requests.Session,
    camera: Camera,
    image_link: str,
    output_dir: Path,
    timestamp: datetime,
) -> Path:
    """Download an image for a camera and save it to disk."""

    response = session.get(image_link, timeout=30)
    response.raise_for_status()

    # Use the suffix from the URL if available, otherwise default to .jpg.
    suffix = Path(image_link).suffix or ".jpg"
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")
    destination_dir = output_dir / camera.camera_id
    _ensure_directory(destination_dir)

    destination = destination_dir / f"{timestamp_str}{suffix}"
    destination.write_bytes(response.content)
    return destination


class S3Uploader:
    """Simple helper to push downloaded files to an S3 bucket."""

    def __init__(self, bucket: str, prefix: str = "", profile: Optional[str] = None, region: Optional[str] = None):
        try:  # Import lazily so users without boto3 can still run local downloads.
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("boto3 must be installed to upload to S3") from exc

        session_kwargs = {}
        if profile:
            session_kwargs["profile_name"] = profile
        if region:
            session_kwargs["region_name"] = region

        self._session = boto3.session.Session(**session_kwargs)
        self._client = self._session.client("s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._boto_core_error = BotoCoreError
        self._client_error = ClientError

    def upload(self, file_path: Path, camera: Camera) -> None:
        relative_key = f"{camera.camera_id}/{file_path.name}"
        if self._prefix:
            key = f"{self._prefix}/{relative_key}"
        else:
            key = relative_key

        try:
            self._client.upload_file(str(file_path), self._bucket, key)
            LOGGER.info("Uploaded %s to s3://%s/%s", file_path, self._bucket, key)
        except (self._boto_core_error, self._client_error) as exc:
            LOGGER.error("Failed to upload %s to s3://%s/%s: %s", file_path, self._bucket, key, exc)


def poll_and_download(
    cameras: Sequence[Camera],
    api_key: str,
    output_dir: Path,
    interval: timedelta,
    duration: timedelta,
    active_start_seconds: int,
    active_end_seconds: int,
    active_timezone: tzinfo,
    upload_callback: Optional[Callable[[Path, Camera], None]] = None,
) -> None:
    """Poll the LTA API for the given duration and download camera images."""

    camera_lookup = {camera.camera_id: camera for camera in cameras}
    end_time = datetime.now(timezone.utc) + duration

    with requests.Session() as session:
        while True:
            loop_start = datetime.now(timezone.utc)
            if loop_start >= end_time:
                LOGGER.info("Reached end of requested duration; stopping fetch loop")
                break

            local_time = loop_start.astimezone(active_timezone)
            seconds_since_midnight = (
                local_time.hour * 60 * 60 + local_time.minute * 60 + local_time.second
            )
            if not within_active_window(seconds_since_midnight, active_start_seconds, active_end_seconds):
                wait_seconds = seconds_until_window(seconds_since_midnight, active_start_seconds, active_end_seconds)
                if wait_seconds <= 0:
                    continue
                remaining = (end_time - loop_start).total_seconds()
                if remaining <= 0:
                    LOGGER.info("Reached end of requested duration while waiting for active window")
                    break
                sleep_seconds = min(wait_seconds, remaining)
                LOGGER.debug(
                    "Current time %s outside active window; sleeping %.0f seconds until next window",
                    local_time.isoformat(),
                    sleep_seconds,
                )
                time.sleep(max(0, sleep_seconds))
                continue

            try:
                metadata = fetch_camera_metadata(session, api_key)
            except requests.HTTPError as exc:
                LOGGER.error("HTTP error from LTA API: %s", exc, exc_info=True)
                metadata = []
            except requests.RequestException as exc:
                LOGGER.error("Network error when contacting LTA API: %s", exc, exc_info=True)
                metadata = []
            except Exception:  # pragma: no cover - unexpected errors logged for visibility
                LOGGER.exception("Unexpected error when fetching camera metadata")
                metadata = []

            found_cameras = set()
            timestamp = datetime.now(timezone.utc)
            for entry in metadata:
                camera_id = str(entry.get("CameraID"))
                if camera_id not in camera_lookup:
                    continue
                image_link = entry.get("ImageLink")
                if not isinstance(image_link, str) or not image_link:
                    LOGGER.warning("No image link available for camera %s", camera_id)
                    continue

                camera = camera_lookup[camera_id]
                try:
                    destination = download_image(session, camera, image_link, output_dir, timestamp)
                    LOGGER.info("Downloaded camera %s image to %s", camera_id, destination)
                    found_cameras.add(camera_id)
                    if upload_callback is not None:
                        upload_callback(destination, camera)
                except requests.HTTPError as exc:
                    LOGGER.warning("Failed to download image for camera %s: %s", camera_id, exc)
                except requests.RequestException as exc:
                    LOGGER.warning(
                        "Network error when downloading image for camera %s: %s", camera_id, exc
                    )

            missing = set(camera_lookup) - found_cameras
            if missing:
                LOGGER.warning(
                    "Did not receive data for %d cameras in this cycle: %s", len(missing), ", ".join(sorted(missing))
                )

            # Sleep until the next scheduled interval, taking into account the time spent so far.
            elapsed = datetime.now(timezone.utc) - loop_start
            sleep_seconds = interval.total_seconds() - elapsed.total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-csv",
        type=Path,
        default=Path("reference/camera_info.csv"),
        help="Path to the CSV file containing camera details.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lta_images"),
        help="Directory to store downloaded images.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=positive_float,
        default=5.0,
        help="Polling interval in minutes (default: 5)",
    )
    parser.add_argument(
        "--duration-days",
        type=positive_float,
        default=7.0,
        help="Total duration of polling in days (default: 7)",
    )
    parser.add_argument(
        "--active-start",
        default="05:00",
        help="Daily start of the active polling window in HH:MM (default: 05:00)",
    )
    parser.add_argument(
        "--active-end",
        default="24:00",
        help="Daily end of the active polling window in HH:MM, exclusive (default: 24:00)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LTA_API_KEY"),
        help="LTA API key. Defaults to the LTA_API_KEY environment variable.",
    )
    parser.add_argument(
        "--s3-bucket",
        help="If provided, upload images to the specified AWS S3 bucket.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="Optional prefix for S3 object keys when uploading.",
    )
    parser.add_argument(
        "--aws-profile",
        help="Optional AWS profile name used when creating the boto3 session.",
    )
    parser.add_argument(
        "--aws-region",
        help="Optional AWS region name used when creating the boto3 session.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    if not args.api_key:
        LOGGER.error("An API key must be provided via --api-key or the LTA_API_KEY environment variable")
        return 1

    if SINGAPORE_TZ is None:
        LOGGER.error("zoneinfo module not available; Python 3.9+ is required for timezone support")
        return 1

    try:
        active_start_seconds = parse_time_of_day(args.active_start)
        active_end_seconds = parse_time_of_day(args.active_end)
    except argparse.ArgumentTypeError as exc:
        LOGGER.error("Invalid active window configuration: %s", exc)
        return 1

    if active_start_seconds == active_end_seconds:
        LOGGER.error("Active window start and end times cannot be identical")
        return 1

    uploader = None
    if args.s3_bucket:
        try:
            uploader = S3Uploader(
                bucket=args.s3_bucket,
                prefix=args.s3_prefix,
                profile=args.aws_profile,
                region=args.aws_region,
            )
        except Exception as exc:  # pragma: no cover - relies on optional dependency
            LOGGER.error("Unable to configure S3 uploader: %s", exc)
            return 1

    try:
        cameras = load_cameras(args.camera_csv)
    except Exception as exc:
        LOGGER.error("Unable to load camera data: %s", exc)
        return 1

    try:
        poll_and_download(
            cameras=cameras,
            api_key=args.api_key,
            output_dir=args.output_dir,
            interval=timedelta(minutes=args.interval_minutes),
            duration=timedelta(days=args.duration_days),
            active_start_seconds=active_start_seconds,
            active_end_seconds=active_end_seconds,
            active_timezone=SINGAPORE_TZ,
            upload_callback=uploader.upload if uploader else None,
        )
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user; exiting")
    except Exception:
        LOGGER.exception("Unexpected error during polling loop")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
