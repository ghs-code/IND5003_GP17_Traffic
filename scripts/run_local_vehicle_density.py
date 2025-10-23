#!/usr/bin/env python3
"""Convenience wrapper to analyse locally stored LTA camera images.

The script assumes images are organised as ``<image-root>/<camera-id>/<file>`` –
matching the layout produced by ``fetch_lta_camera_images.py``.  It simply
invokes ``compute_vehicle_density.py`` with the appropriate arguments, so the
full YOLO-based counting logic is reused without needing to pull data from S3.

Example
-------

```bash
python scripts/run_local_vehicle_density.py \
    --image-root reference/pictures \
    --output-csv outputs/local_vehicle_density.csv \
    --model yolov8n.pt
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

# Ensure the sibling module can be imported when running from the repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compute_vehicle_density  # noqa: E402

DEFAULT_IMAGE_ROOT = SCRIPT_DIR.parent / "reference" / "pictures"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR.parent / "outputs" / "local_vehicle_density.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Directory containing per-camera image folders (default: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Ultralytics YOLO model identifier or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use for inference (auto, cpu, cuda, mps, ...)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Optional override for vehicle classes to count (defaults to compute_vehicle_density defaults)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Minimum confidence threshold for detections",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of images to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for YOLO batch size",
    )
    parser.add_argument(
        "--road-model",
        type=str,
        default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        help="Road segmentation model identifier to forward to compute_vehicle_density",
    )
    parser.add_argument(
        "--road-class",
        type=str,
        default="road",
        help="Road class label to use when computing area ratio",
    )
    parser.add_argument(
        "--road-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for road pixels (set <0 to use argmax)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        help="Input image size for YOLO inference (e.g., 1280). If unset, uses model default.",
    )
    parser.add_argument(
        "--yolo-augment",
        action="store_true",
        help="Enable test-time augmentation in YOLO inference for higher recall (slower).",
    )
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Enable Ultralytics 'retina_masks' for higher-quality instance masks.",
    )
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        help="Enable manual horizontal-flip TTA (merge by choosing higher-count result).",
    )
    parser.add_argument(
        "--refine-with-sam",
        action="store_true",
        help="Refine vehicle masks using SAM given YOLO detections (slower, better separation).",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Path to SAM checkpoint (.pth). Required when --refine-with-sam is set.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_h",
        help="SAM model type (e.g., vit_h, vit_l, vit_b).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU threshold for YOLO NMS (ultralytics 'iou' param). If unset, uses model default.",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=None,
        help="Maximum number of YOLO detections per image. If unset, uses model default.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--save-viz-dir",
        type=Path,
        default=None,
        help="Optional directory to save visualization images with road/vehicle overlays",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=None,
        help="Per-camera ROI config (JSON or YAML) to restrict road area by camera_id",
    )
    parser.add_argument(
        "--roi-labelme-dir",
        type=Path,
        default=None,
        help="Directory of LabelMe JSON files; each filename (without extension) is treated as camera_id",
    )
    parser.add_argument(
        "--unify-vehicle-counts",
        action="store_true",
        help="Write a single 'count_vehicles' column (sum of selected classes) instead of per-class columns",
    )
    return parser.parse_args(argv)


def build_compute_args(args: argparse.Namespace) -> List[str]:
    """Translate wrapper arguments into compute_vehicle_density CLI args."""

    cli_args: List[str] = [
        "--image-root",
        str(args.image_root),
        "--output-csv",
        str(args.output_csv),
        "--model",
        args.model,
    ]

    if args.device:
        cli_args.extend(["--device", args.device])
    if args.classes:
        cli_args.append("--classes")
        cli_args.extend(args.classes)
    if args.conf_threshold is not None:
        cli_args.extend(["--conf-threshold", str(args.conf_threshold)])
    if args.max_images is not None:
        cli_args.extend(["--max-images", str(args.max_images)])
    if args.batch_size is not None:
        cli_args.extend(["--batch-size", str(args.batch_size)])
    if args.road_model:
        cli_args.extend(["--road-model", args.road_model])
    if args.road_class:
        cli_args.extend(["--road-class", args.road_class])
    if args.road_threshold is not None:
        cli_args.extend(["--road-threshold", str(args.road_threshold)])
    if args.save_viz_dir is not None:
        cli_args.extend(["--save-viz-dir", str(args.save_viz_dir)])
    if args.roi_config is not None:
        cli_args.extend(["--roi-config", str(args.roi_config)])
    if args.roi_labelme_dir is not None:
        cli_args.extend(["--roi-labelme-dir", str(args.roi_labelme_dir)])
    if args.yolo_imgsz is not None:
        cli_args.extend(["--yolo-imgsz", str(args.yolo_imgsz)])
    if args.yolo_augment:
        cli_args.append("--yolo-augment")
    if args.iou_threshold is not None:
        cli_args.extend(["--iou-threshold", str(args.iou_threshold)])
    if args.yolo_max_det is not None:
        cli_args.extend(["--yolo-max-det", str(args.yolo_max_det)])
    if args.retina_masks:
        cli_args.append("--retina-masks")
    if args.flip_tta:
        cli_args.append("--flip-tta")
    if args.refine_with_sam:
        cli_args.append("--refine-with-sam")
    if args.sam_checkpoint is not None:
        cli_args.extend(["--sam-checkpoint", str(args.sam_checkpoint)])
    if args.sam_model_type:
        cli_args.extend(["--sam-model-type", args.sam_model_type])
    if args.unify_vehicle_counts:
        cli_args.append("--unify-vehicle-counts")

    return cli_args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.image_root.exists():
        logging.error("Image root %s does not exist", args.image_root)
        return 1

    if args.output_csv.parent and not args.output_csv.parent.exists():
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    compute_args = build_compute_args(args)
    logging.debug("Delegating to compute_vehicle_density with args: %s", compute_args)
    return compute_vehicle_density.main(compute_args)


if __name__ == "__main__":
    raise SystemExit(main())
