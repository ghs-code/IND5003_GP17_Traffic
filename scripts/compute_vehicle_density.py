"""Analyse traffic camera images with YOLO to estimate vehicle density.

This utility loads camera images that were previously downloaded with
``fetch_lta_camera_images.py`` (or any directory containing images organised
by ``<camera-id>/<timestamp>.jpg``) and runs an Ultralytics YOLO model to count
vehicles.  The resulting per-image statistics – total vehicle count and a
simple density metric (vehicles per megapixel) – are saved to a CSV file, which
can optionally be uploaded back to the source S3 bucket for downstream use.

If your images were uploaded to S3 via ``fetch_lta_camera_images.py`` you can
point the script at the corresponding bucket/prefix and it will download the
objects locally before running inference.

Example usage
-------------

```bash
python scripts/compute_vehicle_density.py \
    --image-root data/lta_images \
    --output-csv data/vehicle_density.csv \
    --model yolov8n.pt
```

By default the script counts detections belonging to the ``car``, ``bus``,
``truck`` and ``motorcycle`` classes.  You can override the classes using the
``--classes`` flag to match the class names in the YOLO model you are using.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
    raise SystemExit(
        "The 'ultralytics' package is required to run this script. "
        "Install it via 'pip install ultralytics'."
    ) from exc

LOGGER = logging.getLogger(__name__)
DEFAULT_CLASSES = ("car", "bus", "truck", "motorcycle")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ImageStats:
    """Stores model output for a single image."""

    camera_id: str
    timestamp: str
    total_vehicles: int
    vehicles_per_mpx: float
    counts_by_class: Dict[str, int]

    def to_csv_row(self, class_order: Sequence[str]) -> List[str]:
        """Serialise the statistics into a CSV row respecting ``class_order``."""

        row = [
            self.camera_id,
            self.timestamp,
            str(self.total_vehicles),
            f"{self.vehicles_per_mpx:.4f}",
        ]
        for class_name in class_order:
            row.append(str(self.counts_by_class.get(class_name, 0)))
        return row


def iter_image_files(image_root: Path) -> Iterable[Path]:
    """Yield all supported image files below ``image_root`` sorted by path."""

    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            yield path


def _create_s3_client(
    profile: Optional[str],
    region: Optional[str],
) -> Tuple[Any, Tuple[Any, Any]]:
    """Return an S3 client along with the relevant exception classes."""

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The 'boto3' package is required when using --s3-bucket. "
            "Install it via 'pip install boto3'."
        ) from exc

    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    if region:
        session_kwargs["region_name"] = region

    session = boto3.session.Session(**session_kwargs)
    client = session.client("s3")
    return client, (BotoCoreError, ClientError)


def download_images_from_s3(
    bucket: str,
    prefix: str,
    destination: Path,
    profile: Optional[str],
    region: Optional[str],
    max_images: Optional[int],
) -> List[Path]:
    """Download camera images from an S3 bucket into ``destination``.

    The S3 objects are expected to follow the structure created by
    ``fetch_lta_camera_images.py``: ``<prefix>/<camera-id>/<timestamp>.jpg``.
    """

    client, (BotoCoreError, ClientError) = _create_s3_client(profile, region)

    cleaned_prefix = prefix.strip("/")
    prefix_with_sep = f"{cleaned_prefix}/" if cleaned_prefix else ""

    paginator = client.get_paginator("list_objects_v2")
    downloaded: List[Path] = []

    paginate_kwargs = {"Bucket": bucket}
    if prefix_with_sep:
        paginate_kwargs["Prefix"] = prefix_with_sep

    for page in paginator.paginate(**paginate_kwargs):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key:
                continue
            if key.endswith("/"):
                continue

            if prefix_with_sep and key.startswith(prefix_with_sep):
                relative_key = key[len(prefix_with_sep) :]
            else:
                relative_key = key

            suffix = Path(relative_key).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            local_path = destination / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                client.download_file(bucket, key, str(local_path))
                downloaded.append(local_path)
                LOGGER.debug("Downloaded %s to %s", key, local_path)
            except (BotoCoreError, ClientError) as exc:
                LOGGER.warning("Failed to download s3://%s/%s: %s", bucket, key, exc)
                continue

            if max_images is not None and len(downloaded) >= max_images:
                LOGGER.info("Reached --max-images limit while downloading from S3")
                return downloaded

    return downloaded


def upload_file_to_s3(
    bucket: str,
    key: str,
    file_path: Path,
    profile: Optional[str],
    region: Optional[str],
) -> None:
    """Upload ``file_path`` to ``s3://bucket/key``."""

    client, (BotoCoreError, ClientError) = _create_s3_client(profile, region)

    try:
        client.upload_file(str(file_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Failed to upload {file_path} to s3://{bucket}/{key}: {exc}") from exc


def _normalise_timestamp(raw_token: str) -> str:
    """Return an ISO-8601-ish timestamp if the token matches the fetch naming scheme."""

    token = raw_token.strip()
    if len(token) == 16 and token.endswith("Z"):
        try:
            parsed = datetime.strptime(token, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return token
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return token


def parse_camera_and_timestamp(image_path: Path) -> tuple[str, str]:
    """Infer the camera identifier and capture timestamp from the image path."""

    try:
        camera_id = image_path.parent.name
        stem = image_path.stem
    except AttributeError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unable to derive camera/timestamp from {image_path}") from exc

    if not camera_id:
        raise ValueError(f"Missing camera identifier in path: {image_path}")
    if not stem:
        raise ValueError(f"Missing timestamp in filename: {image_path}")

    if "_" in stem:
        timestamp_token = stem.split("_", 1)[0]
    else:
        timestamp_token = stem

    timestamp = _normalise_timestamp(timestamp_token)
    return camera_id, timestamp


def run_inference(
    model: YOLO,
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    conf_threshold: float,
    device: Optional[str],
    road_processor: AutoImageProcessor,
    road_model: SegformerForSemanticSegmentation,
    road_class_id: int,
    road_threshold: Optional[float],
) -> Iterable[ImageStats]:
    """Run YOLO inference on ``image_paths`` and yield ``ImageStats`` objects."""

    class_name_set = set(class_names)
    call_kwargs = {"verbose": False, "conf": conf_threshold}
    if device:
        call_kwargs["device"] = device
    for result in model(image_paths, **call_kwargs):
        if result.path is None:
            LOGGER.warning("Received result without an image path; skipping")
            continue
        image_path = Path(result.path)
        camera_id, timestamp = parse_camera_and_timestamp(image_path)

        with Image.open(image_path) as img:
            pil_image = img.convert("RGB")
            height, width = pil_image.height, pil_image.width
            inputs = road_processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = road_model(**inputs).logits  # shape: (1, num_labels, h, w)
        logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        if road_threshold is None:
            road_mask = (logits.argmax(dim=1) == road_class_id).squeeze(0)
        else:
            probs = logits.softmax(dim=1)
            road_mask = (probs[:, road_class_id, :, :] >= road_threshold).squeeze(0)
        road_pixels = int(road_mask.sum().item())

        counts, vehicle_pixels = _compute_vehicle_stats(result, class_name_set)
        total = sum(counts.values())
        density = (vehicle_pixels / road_pixels) if road_pixels > 0 else 0.0
        if road_pixels == 0:
            LOGGER.debug("Road segmentation produced zero pixels for %s; vehicle density set to 0", image_path)

        yield ImageStats(
            camera_id=camera_id,
            timestamp=timestamp,
            total_vehicles=total,
            vehicles_per_mpx=density,
            counts_by_class=dict(counts),
        )


def write_csv(
    stats: Iterable[ImageStats],
    output_csv: Path,
    class_order: Sequence[str],
) -> None:
    """Persist ``stats`` to ``output_csv`` with consistent columns."""

    _ensure_parent_dir(output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = [
            "camera_id",
            "timestamp",
            "total_vehicles",
            "vehicles_per_mpx",
        ] + [f"count_{name}" for name in class_order]
        writer.writerow(header)
        for stat in stats:
            writer.writerow(stat.to_csv_row(class_order))


def _ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_label_id(config: Any, target_label: str) -> int:
    """Resolve the numeric label id for a given class name."""

    target = target_label.lower()
    if getattr(config, "label2id", None):
        for label_name, idx in config.label2id.items():
            if label_name.lower() == target:
                return int(idx)
    if getattr(config, "id2label", None):
        for idx, label_name in config.id2label.items():
            if label_name.lower() == target:
                return int(idx)
    raise ValueError(f"Unable to resolve label '{target_label}' from segmentation model config")


def _load_road_segmentation(model_name: str, device: str, road_label: str) -> tuple[AutoImageProcessor, SegformerForSemanticSegmentation, int]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    road_class_id = _resolve_label_id(model.config, road_label)
    return processor, model, road_class_id


def _compute_vehicle_stats(
    result: Any,
    class_name_set: set[str],
    threshold: float = 0.5,
) -> tuple[Counter[str], int]:
    """Return per-class counts and total pixel area for detected vehicles."""

    counts: Counter[str] = Counter()
    total_pixels = 0

    if result.boxes is None or result.boxes.cls is None:
        return counts, total_pixels
    if len(result.boxes) == 0:
        return counts, total_pixels

    if result.masks is None or result.masks.data is None:
        raise RuntimeError(
            "Segmentation masks were not found in YOLO results despite detections being present. "
            "Ensure that a YOLOv8 segmentation model (e.g. yolov8n-seg.pt) is used."
        )

    mask_data = result.masks.data.to(dtype=torch.float32)  # shape: (num_instances, mask_h, mask_w)
    mask_data = mask_data.unsqueeze(1)  # (N,1,H,W)
    resized = F.interpolate(
        mask_data,
        size=result.masks.orig_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    mask_bools = (resized > threshold).to(torch.bool).cpu()

    class_indices = result.boxes.cls.to(dtype=torch.int64).cpu().tolist()
    for idx, cls_idx in enumerate(class_indices):
        cls_name = result.names[int(cls_idx)]
        if cls_name not in class_name_set:
            continue
        pixel_count = int(mask_bools[idx].sum().item())
        total_pixels += pixel_count
        counts[cls_name] += 1

    return counts, total_pixels


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and torch.backends.mps.is_available())


def resolve_device(device_arg: Optional[str]) -> str:
    """Return a runtime device string, applying graceful fallbacks when needed."""

    requested = (device_arg or "").lower()
    if requested in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        LOGGER.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"

    if requested == "mps":
        if _mps_available():
            return "mps"
        LOGGER.warning("MPS requested but not available; falling back to CPU")
        return "cpu"

    return requested


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory containing per-camera image folders or staging area for S3 downloads",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV file for vehicle statistics",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model name or path recognised by ultralytics.YOLO",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(DEFAULT_CLASSES),
        help="Class names to count as vehicles",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Name of the S3 bucket containing camera images",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="",
        help="Prefix within the S3 bucket where images are stored",
    )
    parser.add_argument(
        "--csv-s3-key",
        type=str,
        default=None,
        help=(
            "S3 object key to use for the output CSV (defaults to <s3-prefix>/"
            "<output filename>). Applies only when --s3-bucket is provided."
        ),
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile to use when downloading from S3",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region to use when downloading from S3",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence score for detections (default: 0.25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to process in each batch (default: 16)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images to process",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for YOLO inference (e.g. auto, cpu, cuda, mps, 0, etc.)",
    )
    parser.add_argument(
        "--road-model",
        type=str,
        default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        help="Hugging Face model identifier or path for road semantic segmentation",
    )
    parser.add_argument(
        "--road-class",
        type=str,
        default="road",
        help="Label name in the road segmentation model corresponding to roadway pixels",
    )
    parser.add_argument(
        "--road-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a pixel as road (set <0 to use argmax)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.s3_bucket:
        args.image_root.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Downloading images from s3://%s/%s to %s",
            args.s3_bucket,
            args.s3_prefix,
            args.image_root,
        )
        downloaded = download_images_from_s3(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            destination=args.image_root,
            profile=args.aws_profile,
            region=args.aws_region,
            max_images=args.max_images,
        )
        if not downloaded:
            LOGGER.error(
                "No images downloaded from s3://%s/%s", args.s3_bucket, args.s3_prefix
            )
            return 1
        LOGGER.info("Downloaded %d images from S3", len(downloaded))
    elif not args.image_root.exists():
        LOGGER.error("Image root %s does not exist", args.image_root)
        return 1

    device = resolve_device(args.device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    LOGGER.info("Using %s device for YOLO inference", device)

    LOGGER.info("Loading YOLO model %s", args.model)
    model = YOLO(args.model)
    try:
        model.to(device)
    except AttributeError:
        try:
            model.model.to(device)
        except AttributeError:
            LOGGER.debug("YOLO model instance does not expose '.to'; relying on call-time device specification")

    LOGGER.info("Loading road segmentation model %s", args.road_model)
    road_processor, road_model, road_class_id = _load_road_segmentation(args.road_model, device, args.road_class)
    road_threshold = args.road_threshold if args.road_threshold is not None else 0.5
    if road_threshold < 0:
        road_threshold = None

    image_paths = list(iter_image_files(args.image_root))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        LOGGER.error("No images found in %s", args.image_root)
        return 1

    LOGGER.info("Processing %d images", len(image_paths))
    stats: List[ImageStats] = []
    for start in range(0, len(image_paths), args.batch_size):
        batch = image_paths[start : start + args.batch_size]
        LOGGER.debug("Running inference for batch %d-%d", start, start + len(batch))
        stats.extend(
            run_inference(
                model,
                batch,
                args.classes,
                args.conf_threshold,
                device,
                road_processor,
                road_model,
                road_class_id,
                road_threshold,
            )
        )

    write_csv(stats, args.output_csv, args.classes)
    LOGGER.info("Wrote vehicle statistics to %s", args.output_csv)

    if args.s3_bucket:
        csv_key = args.csv_s3_key
        if not csv_key:
            cleaned_prefix = args.s3_prefix.strip("/")
            if cleaned_prefix:
                csv_key = f"{cleaned_prefix}/{args.output_csv.name}"
            else:
                csv_key = args.output_csv.name

        LOGGER.info("Uploading vehicle statistics CSV to s3://%s/%s", args.s3_bucket, csv_key)
        upload_file_to_s3(
            bucket=args.s3_bucket,
            key=csv_key,
            file_path=args.output_csv,
            profile=args.aws_profile,
            region=args.aws_region,
        )
        LOGGER.info("Uploaded CSV to s3://%s/%s", args.s3_bucket, csv_key)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
