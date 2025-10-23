# IND5003_GP17_Traffic

## LTA traffic camera image downloader

The repository includes a utility script that repeatedly downloads images from
the Land Transport Authority (LTA) Traffic Images v2 API for the cameras
defined in `reference/camera_info.csv`. By default the script polls the API
every five minutes and continues for one week, saving the images locally. Image
downloads only occur between 05:00 and 24:00 Singapore time each day and the
files can optionally be copied to an AWS S3 bucket after each download.

### Prerequisites

1. Request an API key from the [LTA DataMall](https://datamall.lta.gov.sg/).
2. Export the API key as an environment variable or pass it as a command-line
   argument.

### Usage

```bash
export LTA_API_KEY="<your-api-key>"
python scripts/fetch_lta_camera_images.py \
  --camera-csv reference/camera_info.csv \
  --output-dir data/lta_images \
  --interval-minutes 5 \
  --duration-days 7 \
  --active-start 05:00 \
  --active-end 24:00
```

The script stores the downloaded files under `data/lta_images/<CameraID>/` with
UTC timestamps in the filenames. Adjust the `--interval-minutes` and
`--duration-days` flags to change how frequently the script runs and how long it
continues polling. Use `--active-start` and `--active-end` to modify the active
hours (in HH:MM, Asia/Singapore time). Passing `--s3-bucket my-bucket` (and
optionally `--s3-prefix`, `--aws-profile`, or `--aws-region`) uploads every
downloaded image to the specified S3 location while also keeping a local copy.

### Vehicle density CSV output

The `scripts/compute_vehicle_density.py` script (and the wrapper
`scripts/run_local_vehicle_density.py`) aggregate per-image detections into a CSV
with the following columns:

- `camera_id`: Camera identifier inferred from the directory name (matches `CameraID` in the CSV).
- `timestamp`: Capture time extracted from the filename. By default it uses the date from the first filename token and HH:MM from the third token (minute-level, e.g. `2025-10-15 21:39`).
- `total_vehicles`: Sum of all counted detections for the configured vehicle classes during that capture.
- `vehicles_per_mpx`: Vehicle density measured as the ratio between total vehicle pixel area (from YOLO masks) and road pixel area (from the semantic segmentation model).
- `count_<class>`: A column per class listed in `--classes` (defaults to `car`, `bus`, `truck`, `motorcycle`)
  showing the individual detection counts.

The script first loads images organised under `<image-root>/<camera-id>/`, runs a YOLO model via the Ultralytics
API, and populates each row using the detections produced for that frame.

### Image analysis CLI reference

The analysis entrypoints are `scripts/compute_vehicle_density.py` and the convenience wrapper `scripts/run_local_vehicle_density.py` (the wrapper simply forwards arguments). Key arguments:

- `--image-root PATH` Required. Root directory organised as `<camera-id>/*.jpg`.
- `--output-csv PATH` Required. Destination CSV path (parent dirs created automatically).
- `--model NAME|PATH` Ultralytics YOLO segmentation model (e.g. `yolov8n-seg.pt`, `yolo11x-seg.pt`, local `.pt`). Must be a segmentation variant (`-seg`).
- `--classes NAMES...` Vehicle classes to count. Defaults: `car bus truck motorcycle`.
- `--device {auto,cpu,cuda,mps,0,1,...}` Inference device. `auto` prefers CUDA, then MPS, else CPU.
- `--conf-threshold FLOAT` Detection confidence threshold (default 0.25). Lower for higher recall.
- `--iou-threshold FLOAT` NMS IoU threshold (ultralytics `iou`). Larger (e.g. 0.7–0.85) reduces mutual suppression in crowded scenes.
- `--batch-size INT` Images per batch (default 16). Reduce if memory-constrained.
- `--yolo-imgsz INT` YOLO input size, e.g. 1024/1280/1536/1792/1920. Larger improves small/close targets, costs time/memory.
- `--yolo-augment` Try Ultralytics built-in TTA (some models ignore it).
- `--retina-masks` Enable higher-resolution instance masks (Ultralytics `retina_masks`). Helpful for separating close vehicles.
- `--yolo-max-det INT` Max detections per image (e.g. 300–1000 for jams).
- `--save-viz-dir PATH` Save visualisations with road (green) and vehicle masks overlaid.
 - `--unify-vehicle-counts` Write a single `count_vehicles` column (sum across selected classes) instead of per-class columns.

Additional controls and notes

- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` Logging verbosity (default `INFO`).
- Some models may ignore `--yolo-augment`; use `--flip-tta` instead for manual horizontal-flip TTA.
- Crowded scenes: raise `--yolo-max-det` (e.g. 600–1000) and `--iou-threshold` (0.7–0.85), and consider `--retina-masks`.

S3 integration (optional, compute script only)

- `--s3-bucket NAME`, `--s3-prefix STR` Download images before analysis and optionally upload the CSV.
- `--csv-s3-key KEY` Custom destination key for the CSV; defaults to `<prefix>/<basename(output)>`.
- `--aws-profile NAME`, `--aws-region NAME` boto3 session configuration.

Downloader CLI reference (fetch_lta_camera_images.py)

- `--camera-csv PATH` Camera list CSV (default `reference/camera_info.csv`).
- `--output-dir PATH` Destination directory (default `data/lta_images`).
- `--interval-minutes FLOAT` Polling interval in minutes (default 5.0).
- `--duration-days FLOAT` Total duration in days (default 7.0).
- `--active-start HH:MM`, `--active-end HH:MM` Daily active window (Asia/Singapore).
- `--api-key STR` LTA DataMall API key (defaults to env `LTA_API_KEY`).
- `--s3-bucket NAME`, `--s3-prefix STR`, `--aws-profile NAME`, `--aws-region NAME` Upload options via boto3.

Road area (segmentation and ROI):

- `--road-model HF_ID|PATH` Road semantic segmentation model (Hugging Face ID). Default `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`. Models with safetensors are preferred for security.
- `--road-class NAME` Label to extract as road from the model config (default `road`).
- `--road-threshold FLOAT` Probability threshold. Set `<0` to use argmax (often more contiguous roads).
- `--roi-config PATH` Optional JSON/YAML with per-camera polygons. Structure: `{ "2701": { "include": [[[x,y],...]], "exclude": [[[x,y],...]] } }`.
- `--roi-labelme-dir DIR` Directory of LabelMe JSONs; each filename (without suffix) is treated as `camera_id`. shape_type=`polygon` is parsed. Labels `road|roi|include` go to include; `exclude|water|ignore|mask_out` go to exclude; others default to include. If `imageWidth/Height` present, polygons are scaled to the current image size.

Advanced recall options (crowded scenes):

- `--flip-tta` Manual horizontal-flip TTA. Runs a second pass on a flipped image and keeps the result with more vehicles.
- `--refine-with-sam` Two-stage refinement: use SAM to derive instance masks from YOLO boxes, improving separation of adjacent vehicles.
- `--sam-checkpoint PATH` Required when using SAM (e.g. `./sam_vit_h_4b8939.pth`).
- `--sam-model-type {vit_h,vit_l,vit_b}` SAM backbone matching the checkpoint (default `vit_h`). Install `segment-anything` and `opencv-python`.

S3 options (only when pulling/pushing to S3 via compute script):

- `--s3-bucket NAME` Download images from an S3 bucket before analysis; optionally upload the CSV.
- `--s3-prefix STR` Prefix within the bucket.
- `--csv-s3-key KEY` Destination key for the output CSV (defaults to `<prefix>/<basename(output)>`).
- `--aws-profile NAME`, `--aws-region NAME` AWS session configuration.

Timestamp extraction details:

- Filenames produced by `fetch_lta_camera_images.py` are of the form `<UTC>_<camera>_<HHMM>_...`. The analysis parses the first token’s date (YYYYMMDD) and the third token’s time (HHMM) and formats `YYYY-MM-DD HH:MM`.
- If the pattern is not present, it falls back to the first token and normalises `%Y%m%dT%H%M%SZ` to `YYYY-MM-DD HH:MM:SS`.

Examples:

- Local CPU, baseline:

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density.csv --model yolov8n-seg.pt --device cpu --max-images 20 --save-viz-dir outputs/viz`

- Higher recall + ROI (crowded):

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density_v11x.csv --model yolo11x-seg.pt --device cuda --conf-threshold 0.18 --iou-threshold 0.8 --yolo-imgsz 1536 --yolo-max-det 800 --retina-masks --flip-tta --road-threshold -1 --roi-labelme-dir reference/pictures_ROI --max-images 20 --save-viz-dir outputs/viz_v11x`

- Two-stage refinement with SAM:

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density_sam.csv --model yolo11x-seg.pt --device cuda --conf-threshold 0.18 --yolo-imgsz 1536 --iou-threshold 0.8 --yolo-max-det 800 --retina-masks --flip-tta --road-threshold -1 --roi-labelme-dir reference/pictures_ROI --max-images 20 --save-viz-dir outputs/viz_sam --refine-with-sam --sam-checkpoint ./sam_vit_h_4b8939.pth --sam-model-type vit_h`

### Running on GitHub Actions

The workflow defined in `.github/workflows/fetch_lta_images.yml` runs the
downloader on GitHub-hosted runners so you do not need to keep a local machine
online. It is scheduled four times per day at 21:00, 02:00, 07:00, and 12:00 UTC
(05:00, 10:00, 15:00, and 20:00 Singapore time). Each run polls for six hours by
default, respecting the script's 05:00–24:00 active window and GitHub Actions'
six-hour job limit.

1. In the repository settings, add the following secrets under **Actions →
   Secrets and variables → Secrets**:
   - `LTA_API_KEY` (required): your LTA DataMall API key.
   - `LTA_S3_BUCKET` (optional): destination bucket name for uploads.
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (optional): credentials with
     permission to write to the bucket.
   - `AWS_REGION` (optional): AWS region for the bucket; defaults to
     `ap-southeast-1` if omitted.
2. (Optional) Define repository-level variables under **Actions → Secrets and
   variables → Variables** to tweak behaviour without editing the workflow. The
   available variables are:

   | Variable name | Purpose | Default |
   | ------------- | ------- | ------- |
   | `LTA_CAMERA_CSV_PATH` | Camera CSV path passed to `--camera-csv`. | `reference/camera_info.csv` |
   | `LTA_OUTPUT_DIR` | Local directory for downloaded files. | `data/lta_images` |
   | `LTA_POLL_INTERVAL_MINUTES` | Value for `--interval-minutes`. | `5` |
   | `LTA_RUN_DURATION_DAYS` | Value for `--duration-days`. | `0.25` (≈6 hours) |
   | `LTA_ACTIVE_START` | Value for `--active-start`. | `05:00` |
   | `LTA_ACTIVE_END` | Value for `--active-end`. | `24:00` |
   | `LTA_S3_PREFIX` | Optional prefix for uploaded S3 keys. | _(empty)_ |
   | `LTA_AWS_REGION` | Overrides the AWS region if `AWS_REGION` secret is not set. | `ap-southeast-1` |
   | `LTA_UPLOAD_ARTIFACT` | Set to `true` to keep a zipped copy as a workflow artifact. | _(disabled)_ |
   | `LTA_ARTIFACT_RETENTION_DAYS` | Days to retain the artifact if enabled. | `7` |

3. Enable the workflow (if disabled) and monitor runs under the **Actions** tab.
   You can also trigger it manually through **Run workflow**, optionally
   overriding the interval, duration, or active window inputs for ad-hoc runs.

Because GitHub-hosted runners are ephemeral, ensure S3 uploads or artifacts are
enabled if you need to keep the images after each job completes.

### Dataset
[Target cameras images](https://drive.google.com/drive/folders/1esR4cBL6VO0we1n5ZnQd0-0fYFSW-XpI?usp=sharing)  
Among them，**2701、2702、2704、2706** correspond to **Causeway**；**4703、4707、4712、4713** correspond to **Second Link**.

---

## 使用说明

脚本 `scripts/fetch_lta_camera_images.py` 会按照固定时间间隔调用新加坡陆交局（LTA）
Traffic Images v2 API，下载 `reference/camera_info.csv` 中列出的摄像头图片，并在本地保存。
默认设置为每 5 分钟抓取一次并持续 7 天，同时只在新加坡时间每天 05:00 到 24:00 之间抓取。
文件将保存在 `data/lta_images/<CameraID>/` 目录下，文件名包含 UTC 时间戳，方便按时间排序。

### 前置条件

1. 向 [LTA DataMall](https://datamall.lta.gov.sg/) 申请账号并获取 API Key。
2. 将 API Key 通过 `LTA_API_KEY` 环境变量或 `--api-key` 参数传入脚本。

### 运行方式

```bash
export LTA_API_KEY="<你的 API Key>"
python scripts/fetch_lta_camera_images.py \
  --camera-csv reference/camera_info.csv \
  --output-dir data/lta_images \
  --interval-minutes 5 \
  --duration-days 7 \
  --active-start 05:00 \
  --active-end 24:00
```

可以根据需要调整 `--interval-minutes`（抓取间隔，单位：分钟）和 `--duration-days`
（运行时长，单位：天）参数，以便缩短或延长抓取周期。通过 `--active-start` 与
`--active-end` 指定每日的抓取时间窗口（HH:MM 格式，使用新加坡时区）。若提供
`--s3-bucket`（以及可选的 `--s3-prefix`、`--aws-profile`、`--aws-region`），脚本会在本地
保存文件的同时将其上传到指定的 AWS S3 桶中。


### 车辆密度 CSV 字段说明

脚本 `scripts/compute_vehicle_density.py`（或其封装 `scripts/run_local_vehicle_density.py`）
会将每张图片的检测结果汇总到一个 CSV 文件中，包含以下列：

- `camera_id`：摄像头编号，来自图片所在目录名，对应 `CameraID`。
- `timestamp`：图片时间戳，取自文件名。默认用文件名第 1 段的日期（YYYYMMDD）和第 3 段的时间（HHMM）组合，输出为分钟级：`YYYY-MM-DD HH:MM`（如 `2025-10-15 21:39`）。若不符合该模式，则回退为按 `%Y%m%dT%H%M%SZ` 解析并格式化为 `YYYY-MM-DD HH:MM:SS`。
- `total_vehicles`：该图片检测到的车辆总数，等于所选类别计数之和。
- `vehicles_per_mpx`：车辆密度指标，等于车辆像素面积（由 YOLO 分割掩码统计）与道路像素面积（由语义分割模型统计）之间的比例。
- `count_<class>`：对每个指定类别（默认包含 `car`、`bus`、`truck`、`motorcycle`）分别记录的检测数量。

生成流程为：脚本遍历 `<image-root>/<camera-id>/` 结构下的图片，通过 Ultralytics YOLO 模型检测，
并把每张图片的统计结果写入上述列。

### 图像分析命令行参数参考（中文）

分析入口包括 `scripts/compute_vehicle_density.py` 与其封装 `scripts/run_local_vehicle_density.py`（封装器仅转发参数）。常用参数：

- `--image-root PATH` 必填。图片根目录（`<camera-id>/*.jpg`）。
- `--output-csv PATH` 必填。CSV 输出路径（自动创建父目录）。
- `--model NAME|PATH` YOLO 分割模型（如 `yolov8n-seg.pt`、`yolo11x-seg.pt`、或本地 `.pt`；必须为 `-seg`）。
- `--classes NAMES...` 统计的类别（默认 `car bus truck motorcycle`）。
- `--device {auto,cpu,cuda,mps,0,1,...}` 推理设备；`auto` 优先 CUDA，其次 MPS，否则 CPU。
- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` 日志等级（默认 `INFO`）。

YOLO 推理控制：

- `--conf-threshold` 置信度阈值（默认 0.25）。调低召回更高，误检也可能更多。
- `--iou-threshold` NMS IoU（拥堵时 0.7–0.85 减少互抑）。
- `--batch-size` 批大小（默认 16）。
- `--yolo-imgsz` 输入尺寸 1024/1280/1536/1792/1920（越大越清晰，耗时/显存↑）。
- `--yolo-augment` 尝试内置 TTA（部分模型忽略）。
- `--retina-masks` 更高分辨率实例掩膜，边界更清晰。
- `--yolo-max-det` 每图最大检测数（拥堵建议 600–1000）。

可视化与输出：

- `--save-viz-dir PATH` 保存叠加了道路/车辆掩膜的图。
- `--unify-vehicle-counts` 将各类别计数合并为单列 `count_vehicles`。

道路区域（语义分割与 ROI）：

- `--road-model HF_ID|PATH` 道路语义分割模型（默认 `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`）。
- `--road-class NAME` “道路”标签名（默认 `road`）。
- `--road-threshold` 概率阈值；设 `<0` 时使用 argmax（更连贯）。
- `--roi-config PATH` 集中式 JSON/YAML，配置每个相机的 include/exclude 多边形。
- `--roi-labelme-dir DIR` LabelMe JSON 目录；文件名（去后缀）作为 `camera_id`。解析 `polygon` 形状；`road|roi|include` 计入 include，`exclude|water|ignore|mask_out` 计入 exclude；若含 `imageWidth/Height` 会按当前图像尺寸等比缩放多边形。

高级召回（拥堵）：

- `--flip-tta` 手动水平翻转 TTA（取“车辆数更多”的一侧）。
- `--refine-with-sam` 开启 YOLO+SAM 二阶段；需 `--sam-checkpoint`, `--sam-model-type`，并安装 `segment-anything` 与 `opencv-python`。
- `--sam-checkpoint PATH` SAM 权重路径（如 `./sam_vit_h_4b8939.pth`）。
- `--sam-model-type {vit_h,vit_l,vit_b}` SAM 骨干（默认 `vit_h`）。

S3 集成（仅 compute 脚本适用）：

- `--s3-bucket`, `--s3-prefix`, `--csv-s3-key`, `--aws-profile`, `--aws-region` 下载图片并回传 CSV 的 S3 选项。

下载器参数（fetch_lta_camera_images.py）：

- `--camera-csv` 摄像头 CSV；`--output-dir` 输出目录；`--interval-minutes` 抓取间隔；`--duration-days` 运行时长；`--active-start/--active-end` 每日时间窗；`--api-key` API Key；S3 上传相关：`--s3-bucket/--s3-prefix/--aws-profile/--aws-region`。

### 图像分析命令行参数参考

分析入口包括 `scripts/compute_vehicle_density.py` 与其封装 `scripts/run_local_vehicle_density.py`（封装器仅做参数转发）。常用参数如下：

- `--image-root PATH` 必填。图片根目录，组织结构为 `<camera-id>/*.jpg`。
- `--output-csv PATH` 必填。结果 CSV 输出路径（父目录会自动创建）。
- `--model NAME|PATH` Ultralytics YOLO 分割模型（如 `yolov8n-seg.pt`、`yolo11x-seg.pt`、或本地 `.pt`）。必须为分割权重（`-seg`）。
- `--classes NAMES...` 统计的车辆类别。默认：`car bus truck motorcycle`。
- `--device {auto,cpu,cuda,mps,0,1,...}` 推理设备。`auto` 优先 CUDA，其次 MPS，否则 CPU。
- `--conf-threshold FLOAT` 置信度阈值（默认 0.25）。调低可提高召回。
- `--iou-threshold FLOAT` NMS IoU 阈值（Ultralytics 的 `iou`）。拥堵时可设大些（如 0.7–0.85）以减少互相抑制。
- `--batch-size INT` 批大小（默认 16）。内存紧张时下调。
- `--yolo-imgsz INT` YOLO 输入尺寸，如 1024/1280/1536/1792/1920。越大对小目标更友好，但更耗时/显存。
- `--yolo-augment` 尝试内置 TTA（部分模型不支持，会忽略）。
- `--retina-masks` 启用更高分辨率的实例掩膜（`retina_masks`），有助于分离相邻车辆。
- `--yolo-max-det INT` 每图最大检测数（拥堵时可 300–1000）。
- `--save-viz-dir PATH` 可视化输出目录（叠加道路/车辆掩膜）。
 - `--unify-vehicle-counts` 将 CSV 的类别明细合并为一列 `count_vehicles`（为所选类别之和）。

道路区域（语义分割与 ROI）：

- `--road-model HF_ID|PATH` 道路语义分割模型（Hugging Face ID）。默认 `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`。优先加载 safetensors 权重以满足安全要求。
- `--road-class NAME` 语义分割模型中表示“道路”的标签名（默认 `road`）。
- `--road-threshold FLOAT` 概率阈值。设为 `<0` 时使用 argmax（通常道路更连贯）。
- `--roi-config PATH`（可选）集中式 JSON/YAML，每个相机配置包含/排除多边形。结构示例：`{"2701": {"include": [[[x,y],...]], "exclude": [[[x,y],...]]}}`。
- `--roi-labelme-dir DIR`（推荐）LabelMe JSON 目录；每个文件名（去后缀）作为 `camera_id`。解析 `shape_type=polygon`；标签 `road|roi|include` 归为包含，`exclude|water|ignore|mask_out` 归为排除；其他标签默认按包含处理。若 JSON 含 `imageWidth/Height`，会按当前图片尺寸自动等比缩放多边形。

拥堵场景召回增强：

- `--flip-tta` 手动水平翻转 TTA：对翻转图再推理一次，取“车辆数更多”的一侧。
- `--refine-with-sam` 二阶段细化：用 YOLO 检测框提示 SAM 生成更精细实例掩膜，分离贴近车辆。
- `--sam-checkpoint PATH` 开启 SAM 时必填（如 `./sam_vit_h_4b8939.pth`）。
- `--sam-model-type {vit_h,vit_l,vit_b}` 与权重匹配的骨干（默认 `vit_h`）。需先安装 `segment-anything` 与 `opencv-python`。

S3 相关（当需要从 S3 下载或将结果回传 S3 时）：

- `--s3-bucket NAME` 从该桶下载图片并在结束后可上传 CSV。
- `--s3-prefix STR` 桶内前缀。
- `--csv-s3-key KEY` CSV 上传目标键（默认 `<prefix>/<输出文件名>`）。
- `--aws-profile NAME`, `--aws-region NAME` AWS 会话配置。

时间戳解析说明：

- 由抓取脚本保存的文件名通常形如 `<UTC>_<camera>_<HHMM>_...`。分析脚本取第 1 段的日期（YYYYMMDD）与第 3 段的时间（HHMM），格式化为 `YYYY-MM-DD HH:MM`。若不符合该模式，则回退为按 `%Y%m%dT%H%M%SZ` 解析并格式化为 `YYYY-MM-DD HH:MM:SS`。

示例：

- 本地 CPU，基础运行：

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density.csv --model yolov8n-seg.pt --device cpu --max-images 20 --save-viz-dir outputs/viz`

- 拥堵场景高召回 + ROI：

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density_v11x.csv --model yolo11x-seg.pt --device cuda --conf-threshold 0.18 --iou-threshold 0.8 --yolo-imgsz 1536 --yolo-max-det 800 --retina-masks --flip-tta --road-threshold -1 --roi-labelme-dir reference/pictures_ROI --max-images 20 --save-viz-dir outputs/viz_v11x`

- SAM 二阶段细化：

  `python scripts/run_local_vehicle_density.py --image-root reference/pictures --output-csv outputs/vehicle_density_sam.csv --model yolo11x-seg.pt --device cuda --conf-threshold 0.18 --yolo-imgsz 1536 --iou-threshold 0.8 --yolo-max-det 800 --retina-masks --flip-tta --road-threshold -1 --roi-labelme-dir reference/pictures_ROI --max-images 20 --save-viz-dir outputs/viz_sam --refine-with-sam --sam-checkpoint ./sam_vit_h_4b8939.pth --sam-model-type vit_h`

### 在 GitHub Actions 上运行

仓库中的 `.github/workflows/fetch_lta_images.yml` 工作流可以让脚本在 GitHub 服务器上
定时执行，无需在本地长时间运行。它会在每天的 UTC 时间 21:00、02:00、07:00、12:00
（即新加坡时间 05:00、10:00、15:00、20:00）触发，每次默认运行约 6 小时，以满足脚本
05:00–24:00 的活跃时间限制，并符合 GitHub Actions 单次任务最长 6 小时的限制。

1. 进入仓库 **Settings → Actions → Secrets and variables → Secrets** 页面，新增以下
   密钥：
   - `LTA_API_KEY`（必填）：LTA DataMall 的 API Key。
   - `LTA_S3_BUCKET`（可选）：若需上传到 S3，请填写目标桶名。
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`（可选）：具备写入权限的 AWS 账户
     凭证。
   - `AWS_REGION`（可选）：S3 所在区域，若未设置则默认使用 `ap-southeast-1`。
2. （可选）在 **Settings → Actions → Secrets and variables → Variables** 中设置变量，
   以便无需修改工作流就能调整行为：

   | 变量名 | 作用 | 默认值 |
   | ------ | ---- | ------ |
   | `LTA_CAMERA_CSV_PATH` | 传给 `--camera-csv` 的摄像头 CSV 路径。 | `reference/camera_info.csv` |
   | `LTA_OUTPUT_DIR` | 下载文件保存的本地目录。 | `data/lta_images` |
   | `LTA_POLL_INTERVAL_MINUTES` | `--interval-minutes` 参数。 | `5` |
   | `LTA_RUN_DURATION_DAYS` | `--duration-days` 参数。 | `0.25`（约 6 小时） |
   | `LTA_ACTIVE_START` | `--active-start` 参数。 | `05:00` |
   | `LTA_ACTIVE_END` | `--active-end` 参数。 | `24:00` |
   | `LTA_S3_PREFIX` | 上传至 S3 时的对象前缀。 | _(留空)_ |
   | `LTA_AWS_REGION` | 若未配置 `AWS_REGION` 密钥时使用的区域。 | `ap-southeast-1` |
   | `LTA_UPLOAD_ARTIFACT` | 设置为 `true` 时会上传工作流构件备份。 | _(未启用)_ |
   | `LTA_ARTIFACT_RETENTION_DAYS` | 构件保留天数（启用时）。 | `7` |

3. 确认工作流已启用，可在 **Actions** 页面查看运行情况或手动触发，并在触发时按需
   修改运行参数。

由于 GitHub Runner 是临时实例，若需要保留图片，请开启 S3 上传或启用工作流构件。

### 数据集获取地址
[目标道路相关摄像头影像](https://drive.google.com/drive/folders/1esR4cBL6VO0we1n5ZnQd0-0fYFSW-XpI?usp=sharing)  
其中，**2701、2702、2704、2706** 对应道路为 **Causeway**；**4703、4707、4712、4713** 对应道路为 **Second Link**
