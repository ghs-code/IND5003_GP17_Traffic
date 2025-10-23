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
import numpy as np

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
    """Yield all supported image files below ``image_root`` sorted by path.

    过滤掉 macOS 资源分叉等非真实图像文件（例如以 '._' 开头的文件）。
    """

    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        # 跳过 macOS 生成的资源分叉和隐藏元数据文件
        if name.startswith("._") or name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            yield path


def _load_roi_config(path: Optional[Path]) -> Dict[str, Dict[str, List[List[Tuple[float, float]]]]]:
    """加载 ROI 配置（JSON 或 YAML），返回标准化后的 dict。

    期望结构（示例）：
    {
      "2701": {"include": [ [[x,y],...], ...], "exclude": [ [[x,y],...], ...] },
      "2702": {"include": [...]} , ...
    }
    """

    cfg: Dict[str, Dict[str, List[List[Tuple[float, float]]]]] = {}
    if path is None:
        return cfg
    if not path.exists():
        LOGGER.warning("ROI 配置文件不存在: %s", path)
        return cfg
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("无法读取 ROI 配置文件 %s: %s", path, exc)
        return cfg

    data_obj: Any = None
    # 先尝试 JSON
    try:
        import json

        data_obj = json.loads(text)
    except Exception:
        # 再尝试 YAML（如已安装 PyYAML）
        try:
            import yaml  # type: ignore

            data_obj = yaml.safe_load(text)
        except Exception as exc:
            LOGGER.warning("无法解析 ROI 配置为 JSON 或 YAML: %s", exc)
            return cfg

    if not isinstance(data_obj, dict):
        LOGGER.warning("ROI 配置顶层应为对象/dict，实际为: %s", type(data_obj).__name__)
        return cfg

    def _norm_poly(poly: Any) -> Optional[List[Tuple[float, float]]]:
        try:
            pts = []
            for pt in poly:
                x, y = pt
                pts.append((float(x), float(y)))
            if len(pts) >= 3:
                return pts
        except Exception:
            return None
        return None

    for cam, spec in data_obj.items():
        if not isinstance(spec, dict):
            continue
        inc_list = spec.get("include", [])
        exc_list = spec.get("exclude", [])
        inc_polys: List[List[Tuple[float, float]]] = []
        exc_polys: List[List[Tuple[float, float]]] = []
        if isinstance(inc_list, list):
            for poly in inc_list:
                norm = _norm_poly(poly)
                if norm:
                    inc_polys.append(norm)
        if isinstance(exc_list, list):
            for poly in exc_list:
                norm = _norm_poly(poly)
                if norm:
                    exc_polys.append(norm)
        cfg[str(cam)] = {"include": inc_polys, "exclude": exc_polys}
    return cfg


def _rasterize_roi(polygons: List[List[Tuple[float, float]]], size: Tuple[int, int]) -> np.ndarray:
    """将多边形列表栅格化为布尔掩膜（H,W）。size=(W,H)。"""

    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    w, h = size
    mask = _Image.new("L", (w, h), 0)
    draw = _ImageDraw.Draw(mask)
    for poly in polygons:
        try:
            draw.polygon([(float(x), float(y)) for x, y in poly], outline=1, fill=1)
        except Exception:
            continue
    return (np.array(mask) > 0)


def _load_roi_from_labelme_dir(dir_path: Optional[Path]) -> Dict[str, Dict[str, List[List[Tuple[float, float]]]]]:
    """从 LabelMe 标注目录加载 ROI。

    规则：
    - 遍历目录下的 .json 文件，文件名（不含后缀）作为 camera_id。
    - 仅解析 'shapes' 列表中 shape_type 为 'polygon' 的标注。
    - 标签（label）映射：
      * include_labels = {'road','roi','include'} 视为包含多边形
      * exclude_labels = {'exclude','water','ignore','mask_out'} 视为排除多边形
      * 若标签不在上述集合，但存在多边形，也默认视为 include（宽松容错）
    """

    cfg: Dict[str, Dict[str, List[List[Tuple[float, float]]]]] = {}
    if dir_path is None or not dir_path.exists():
        return cfg

    include_labels = {"road", "roi", "include"}
    exclude_labels = {"exclude", "water", "ignore", "mask_out"}

    for json_path in sorted(dir_path.glob("*.json")):
        cam_id = json_path.stem
        try:
            import json

            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        shapes = data.get("shapes", [])
        base_w = data.get("imageWidth")
        base_h = data.get("imageHeight")
        inc_polys: List[List[Tuple[float, float]]] = []
        exc_polys: List[List[Tuple[float, float]]] = []
        for shp in shapes:
            if not isinstance(shp, dict):
                continue
            if shp.get("shape_type") != "polygon":
                continue
            points = shp.get("points", [])
            try:
                poly = [(float(x), float(y)) for x, y in points]
            except Exception:
                continue
            if len(poly) < 3:
                continue
            label = str(shp.get("label", "")).strip().lower()
            if label in exclude_labels:
                exc_polys.append(poly)
            else:
                # 默认走 include，兼容 road/roi/include 或其他任意标签
                inc_polys.append(poly)
        if inc_polys or exc_polys:
            entry: Dict[str, Any] = {"include": inc_polys, "exclude": exc_polys}
            try:
                if isinstance(base_w, int) and isinstance(base_h, int) and base_w > 0 and base_h > 0:
                    entry["base_size"] = (int(base_w), int(base_h))
            except Exception:
                pass
            cfg[cam_id] = entry  # type: ignore[assignment]
    return cfg


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
    """Infer the camera identifier and capture timestamp from the image path.

    Updated behaviour:
    - If filename matches the expected multi-part pattern produced by the fetch script,
      e.g. ``<UTC>_<camid>_<HHMM>_<other>...``, then compose the output timestamp using:
        - Date from the first token's YYYYMMDD (UTC prefix), and
        - Time from the third token's HHMM (interpreted as local HH:MM minutes-level)
      The resulting format is ``YYYY-MM-DD HH:MM`` (no seconds).
    - Otherwise, fall back to the previous behaviour which uses the first token and
      normalises it when it resembles ``%Y%m%dT%H%M%SZ``.
    """

    try:
        camera_id = image_path.parent.name
        stem = image_path.stem
    except AttributeError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unable to derive camera/timestamp from {image_path}") from exc

    if not camera_id:
        raise ValueError(f"Missing camera identifier in path: {image_path}")
    if not stem:
        raise ValueError(f"Missing timestamp in filename: {image_path}")

    parts = stem.split("_") if "_" in stem else [stem]

    # Preferred path: use date from first token and HH:MM from the third token (HHMM)
    if len(parts) >= 3:
        first_token = parts[0]
        third_token = parts[2]
        ymd = first_token[:8]
        hhmm = third_token[:4]
        if len(ymd) == 8 and ymd.isdigit() and len(hhmm) == 4 and hhmm.isdigit():
            date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
            time_str = f"{hhmm[0:2]}:{hhmm[2:4]}"
            return camera_id, f"{date_str} {time_str}"

    # Fallback: keep using the first token and normalise if it matches the UTC pattern
    timestamp_token = parts[0]
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
    save_viz_dir: Optional[Path] = None,
    roi_config: Optional[Dict[str, Dict[str, Any]]] = None,
    flip_tta: bool = False,
    sam_predictor: Any = None,
) -> Iterable[ImageStats]:
    """Run YOLO inference on ``image_paths`` and yield ``ImageStats`` objects."""

    class_name_set = set(class_names)
    call_kwargs = {"verbose": False, "conf": conf_threshold}
    # Optional Ultralytics YOLO arguments are read from top-level parsed args via environment
    # variables set by main(). To avoid threading the whole args object, we read from env.
    # Keys: YOLO_IOU, YOLO_IMGSZ, YOLO_AUGMENT, YOLO_MAX_DET, YOLO_RETINA
    try:
        iou_str = os.environ.get("YOLO_IOU")
        if iou_str:
            call_kwargs["iou"] = float(iou_str)
    except Exception:
        pass
    imgsz_str = os.environ.get("YOLO_IMGSZ")
    if imgsz_str:
        try:
            call_kwargs["imgsz"] = int(imgsz_str)
        except Exception:
            LOGGER.warning("Invalid YOLO_IMGSZ=%s; ignoring", imgsz_str)
    if os.environ.get("YOLO_AUGMENT") == "1":
        call_kwargs["augment"] = True
    max_det_str = os.environ.get("YOLO_MAX_DET")
    if max_det_str:
        try:
            call_kwargs["max_det"] = int(max_det_str)
        except Exception:
            LOGGER.warning("Invalid YOLO_MAX_DET=%s; ignoring", max_det_str)
    if os.environ.get("YOLO_RETINA") == "1":
        call_kwargs["retina_masks"] = True
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

        # ROI 替换：若为该 camera_id 配置了 ROI，则使用 include - exclude 作为 road_mask
        if roi_config and camera_id in roi_config:
            spec = roi_config.get(camera_id, {})
            includes = spec.get("include", [])
            excludes = spec.get("exclude", [])
            base_size = spec.get("base_size")
            # 若标注基准尺寸与当前图片尺寸不同，则按比例缩放
            if includes:
                inc_polys = includes
                exc_polys = excludes or []
                if isinstance(base_size, (list, tuple)) and len(base_size) == 2:
                    try:
                        base_w, base_h = int(base_size[0]), int(base_size[1])
                        if base_w > 0 and base_h > 0 and (base_w != width or base_h != height):
                            sx, sy = width / float(base_w), height / float(base_h)
                            def _scale(ps):
                                return [[(x * sx, y * sy) for (x, y) in poly] for poly in ps]
                            inc_polys = _scale(inc_polys)
                            if exc_polys:
                                exc_polys = _scale(exc_polys)
                    except Exception:
                        pass
                inc_mask = _rasterize_roi(inc_polys, (width, height))
                roi_bool = inc_mask
                if exc_polys:
                    exc_mask = _rasterize_roi(exc_polys, (width, height))
                    roi_bool = np.logical_and(roi_bool, np.logical_not(exc_mask))
                road_mask = torch.from_numpy(roi_bool).to(device=device)
        road_pixels = int(road_mask.sum().item())

        # 计算原图结果
        counts, vehicle_pixels = _compute_vehicle_stats(result, class_name_set, roi_mask=road_mask)

        # 可选：水平翻转 TTA，选择计数更大的那一侧
        if flip_tta:
            try:
                flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                flip_res_list = model(flipped, **call_kwargs)
                if flip_res_list:
                    flip_res = flip_res_list[0]
                    # 将掩膜翻回原坐标（水平反转）
                    if getattr(flip_res, "masks", None) is not None and getattr(flip_res.masks, "data", None) is not None:
                        flip_res.masks.data = torch.flip(flip_res.masks.data, dims=[2])
                        flip_res.masks.orig_shape = (height, width)
                    f_counts, f_pixels = _compute_vehicle_stats(flip_res, class_name_set, roi_mask=road_mask)
                    if sum(f_counts.values()) > sum(counts.values()):
                        counts, vehicle_pixels = f_counts, f_pixels
            except Exception as exc:
                LOGGER.debug("Flip-TTA failed for %s: %s", image_path, exc)

        # 可选：SAM 二阶段细化（用 bbox 生成更细掩膜）
        if sam_predictor is not None and result.boxes is not None and result.boxes.cls is not None:
            try:
                s_counts, s_pixels = _compute_vehicle_stats_with_sam(
                    pil_image,
                    result,
                    class_name_set,
                    road_mask,
                    sam_predictor,
                    device,
                )
                # 取更高计数的结果
                if sum(s_counts.values()) >= sum(counts.values()):
                    counts, vehicle_pixels = s_counts, s_pixels
            except Exception as exc:
                LOGGER.warning("SAM refinement failed for %s: %s", image_path, exc)
        total = sum(counts.values())
        density = (vehicle_pixels / road_pixels) if road_pixels > 0 else 0.0
        if road_pixels == 0:
            LOGGER.debug("Road segmentation produced zero pixels for %s; vehicle density set to 0", image_path)

        if save_viz_dir is not None:
            try:
                _save_visualization(
                    pil_image,
                    road_mask,
                    result,
                    class_name_set,
                    save_viz_dir / image_path.parent.name / image_path.name,
                    counts,
                    camera_id,
                    timestamp,
                )
            except Exception as exc:
                LOGGER.warning("Failed to save visualization for %s: %s", image_path, exc)

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
    unify_vehicle_counts: bool = False,
) -> None:
    """Persist ``stats`` to ``output_csv`` with consistent columns."""

    _ensure_parent_dir(output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        base_header = [
            "camera_id",
            "timestamp",
            "total_vehicles",
            "vehicles_per_mpx",
        ]
        if unify_vehicle_counts:
            header = base_header + ["count_vehicles"]
        else:
            header = base_header + [f"count_{name}" for name in class_order]
        writer.writerow(header)
        for stat in stats:
            if unify_vehicle_counts:
                total = sum(stat.counts_by_class.get(name, 0) for name in class_order)
                row = [
                    stat.camera_id,
                    stat.timestamp,
                    str(stat.total_vehicles),
                    f"{stat.vehicles_per_mpx:.4f}",
                    str(total),
                ]
                writer.writerow(row)
            else:
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
    """Load the road segmentation model, preferring safetensors to avoid torch.load CVE.

    Transformers >=4.43 enforces torch>=2.6 when loading .bin via torch.load due to
    CVE-2025-32434. If the model provides safetensors weights, we request them to
    bypass the restriction on environments still pinned to older torch.
    """

    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True,
        )
    except ValueError as exc:
        # Provide a clearer error when safetensors are unavailable and torch<2.6
        message = str(exc)
        if "upgrade torch to at least v2.6" in message or "CVE-2025-32434" in message:
            raise SystemExit(
                "无法加载道路分割模型：当前环境的 torch 版本过低且该模型未提供 safetensors 权重。\n"
                "解决方案之一：升级 PyTorch 至 2.6+；或选择带有 safetensors 权重的分割模型；"
                "或将运行设备切换到 CPU 版 2.6+ 后再试。"
            ) from exc
        raise

    model.to(device)
    model.eval()
    road_class_id = _resolve_label_id(model.config, road_label)
    return processor, model, road_class_id


def _compute_vehicle_stats(
    result: Any,
    class_name_set: set[str],
    threshold: float = 0.5,
    roi_mask: Optional[torch.Tensor] = None,
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
    mask_bools = (resized > threshold).to(torch.bool)
    if roi_mask is not None:
        try:
            # roi_mask expected shape (H, W)
            roi_b = roi_mask.to(dtype=torch.bool)
            if roi_b.ndim == 2:
                mask_bools = mask_bools & roi_b.unsqueeze(0).to(mask_bools.device)
        except Exception:
            pass
    mask_bools = mask_bools.cpu()

    class_indices = result.boxes.cls.to(dtype=torch.int64).cpu().tolist()
    for idx, cls_idx in enumerate(class_indices):
        cls_name = result.names[int(cls_idx)]
        if cls_name not in class_name_set:
            continue
        pixel_count = int(mask_bools[idx].sum().item())
        if pixel_count <= 0:
            continue
        total_pixels += pixel_count
        counts[cls_name] += 1

    return counts, total_pixels


def _save_visualization(
    pil_image: Image.Image,
    road_mask: torch.Tensor,
    result: Any,
    class_name_set: set[str],
    out_path: Path,
    counts: Dict[str, int],
    camera_id: str,
    timestamp: str,
) -> None:
    """Save an overlay image showing road and detected vehicles.

    - Road area: semi-transparent green
    - Vehicles: colored by class, semi-transparent
    - Adds a small legend text with camera_id, timestamp and counts
    """

    img_np = np.array(pil_image).astype(np.uint8)  # (H,W,3)
    overlay = img_np.copy()

    # Prepare masks
    road_mask_np = road_mask.detach().cpu().numpy().astype(bool)

    # Resize YOLO masks to image size and threshold to boolean masks
    if result.masks is not None and result.masks.data is not None:
        mask_data = result.masks.data.to(dtype=torch.float32)  # (N,h,w)
        mask_data = mask_data.unsqueeze(1)  # (N,1,h,w)
        resized = F.interpolate(
            mask_data,
            size=result.masks.orig_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        inst_masks = (resized > 0.5).to(torch.bool).cpu()  # (N,H,W)
        class_indices = result.boxes.cls.to(dtype=torch.int64).cpu().tolist() if result.boxes is not None else []
    else:
        inst_masks = torch.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=torch.bool)
        class_indices = []

    # Colors for classes
    colors = {
        "car": (0, 0, 255),
        "bus": (255, 255, 0),
        "truck": (255, 0, 255),
        "motorcycle": (0, 255, 255),
    }
    default_color = (255, 0, 0)

    # Blend function
    def blend(mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.4) -> None:
        if not mask.any():
            return
        overlay[mask] = (
            (1 - alpha) * overlay[mask].astype(np.float32)
            + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    # Apply road mask
    blend(road_mask_np, (0, 255, 0), alpha=0.25)

    # Apply instance masks per class
    for i, cls_idx in enumerate(class_indices):
        cls_name = result.names[int(cls_idx)]
        if cls_name not in class_name_set:
            continue
        mask_np = inst_masks[i].numpy()
        color = colors.get(cls_name, default_color)
        blend(mask_np, color, alpha=0.45)

    # Compose final image
    out_np = overlay

    # Draw legend text
    try:
        draw = Image.fromarray(out_np)
        from PIL import ImageDraw, ImageFont

        d = ImageDraw.Draw(draw)
        text = f"cam: {camera_id}  ts: {timestamp}\n" + \
               " ".join([f"{k}:{v}" for k, v in counts.items()])
        d.rectangle([5, 5, 5 + 8 * max(20, len(text)), 35], fill=(0, 0, 0, 128))
        d.text((10, 10), text, fill=(255, 255, 255))
        out_img = draw
    except Exception:
        out_img = Image.fromarray(out_np)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def _load_sam_predictor(checkpoint: Path, model_type: str, device: str) -> Any:
    try:
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "启用 --refine-with-sam 需要安装 'segment-anything' 包，并提供 --sam-checkpoint." 
            "请执行: pip install segment-anything opencv-python"
        ) from exc

    if not checkpoint.exists():
        raise SystemExit(f"SAM 权重不存在: {checkpoint}")
    model = sam_model_registry.get(model_type)
    if model is None:
        raise SystemExit(f"未知的 SAM model_type: {model_type}")
    sam = model(checkpoint=str(checkpoint))
    sam.to(device)
    return SamPredictor(sam)


def _compute_vehicle_stats_with_sam(
    pil_image: Image.Image,
    yolo_result: Any,
    class_name_set: set[str],
    roi_mask: torch.Tensor,
    sam_predictor: Any,
    device: str,
) -> tuple[Counter[str], int]:
    """使用 SAM 基于 YOLO 的 bbox 细化车辆实例掩膜并统计。"""

    img_np = np.array(pil_image).astype(np.uint8)
    sam_predictor.set_image(img_np)

    counts: Counter[str] = Counter()
    total_pixels = 0

    if yolo_result.boxes is None or yolo_result.boxes.cls is None:
        return counts, total_pixels

    boxes = yolo_result.boxes.xyxy.detach().cpu().numpy()  # (N,4)
    classes = yolo_result.boxes.cls.to(dtype=torch.int64).detach().cpu().tolist()
    h, w = img_np.shape[0], img_np.shape[1]

    for idx, (box, cls_idx) in enumerate(zip(boxes, classes)):
        cls_name = yolo_result.names[int(cls_idx)]
        if cls_name not in class_name_set:
            continue
        try:
            masks, scores, _ = sam_predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )  # masks: (1,H,W)
            mask = masks[0].astype(bool)
            # 与 ROI 相交
            roi_np = roi_mask.detach().cpu().numpy().astype(bool)
            mask = np.logical_and(mask, roi_np)
            pixel_count = int(mask.sum())
            if pixel_count <= 0:
                continue
            total_pixels += pixel_count
            counts[cls_name] += 1
        except Exception:
            continue
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
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU threshold for NMS (ultralytics 'iou' param). If unset, uses model default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to process in each batch (default: 16)",
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
        "--yolo-max-det",
        type=int,
        default=None,
        help="Maximum number of detections per image for YOLO. If unset, uses model default.",
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
        "--unify-vehicle-counts",
        action="store_true",
        help=(
            "If set, CSV will include a single 'count_vehicles' column equal to the sum of"
            " selected classes (e.g., cars + buses + trucks + motorcycles), instead of per-class columns."
        ),
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
        help="Per-camera ROI 配置（JSON 或 YAML）。包含 include/exclude 多边形列表。",
    )
    parser.add_argument(
        "--roi-labelme-dir",
        type=Path,
        default=None,
        help=(
            "包含多份 LabelMe JSON 的目录。文件名（去扩展名）将作为 camera_id。"
            "每个 JSON 内 'shapes' 中的多边形将被解析为 include/exclude 多边形。"
        ),
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

    # Expose select YOLO runtime options to run_inference via environment for simplicity
    if args.iou_threshold is not None:
        os.environ["YOLO_IOU"] = str(args.iou_threshold)
    if args.yolo_imgsz is not None:
        os.environ["YOLO_IMGSZ"] = str(args.yolo_imgsz)
    if args.yolo_augment:
        os.environ["YOLO_AUGMENT"] = "1"
    if args.yolo_max_det is not None:
        os.environ["YOLO_MAX_DET"] = str(args.yolo_max_det)
    if args.retina_masks:
        os.environ["YOLO_RETINA"] = "1"

    LOGGER.info("Processing %d images", len(image_paths))
    stats: List[ImageStats] = []
    roi_cfg = _load_roi_config(args.roi_config)
    # 合并 LabelMe 目录中的 ROI，目录内容覆盖同名 camera_id 的配置
    lm_cfg = _load_roi_from_labelme_dir(args.roi_labelme_dir)
    if lm_cfg:
        roi_cfg.update(lm_cfg)

    # 可选加载 SAM 预测器
    sam_predictor = None
    if args.refine_with_sam:
        if not args.sam_checkpoint:
            raise SystemExit("使用 --refine-with-sam 需要提供 --sam-checkpoint 路径")
        sam_predictor = _load_sam_predictor(args.sam_checkpoint, args.sam_model_type, device)
    for start in range(0, len(image_paths), args.batch_size):
        batch = image_paths[start : start + args.batch_size]
        LOGGER.debug("Running inference for batch %d-%d", start, start + len(batch))
        # Build dynamic YOLO kwargs for this run
        _ = model  # keep reference local
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
                args.save_viz_dir,
                roi_cfg,
                args.flip_tta,
                sam_predictor,
            )
        )
    
    write_csv(stats, args.output_csv, args.classes, unify_vehicle_counts=args.unify_vehicle_counts)
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
