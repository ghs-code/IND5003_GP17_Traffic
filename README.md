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

## Traffic State Clustering and Visualization (Unsupervised Analysis)

This module performs K-means clustering on the processed vehicle density data to automatically identify congestion levels and visualize their temporal patterns. It corresponds to Question 1 of the project, focusing on understanding the daily and weekly traffic rhythm of Singapore’s Causeway and Second Link

### 1. vehicle_clustering.py — Data Clustering Pipeline
This script consolidates eight vehicle-density CSV files (veh_outputs_batch_0 – veh_outputs_batch_7) into one dataset and applies unsupervised learning to label congestion states.

Main workflow

1) Data ingestion & cleaning – merge CSVs, parse timestamps, convert all numeric columns, and remove missing values.
2) Temporal feature engineering – extract hour / weekday information and encode them cyclically using sine / cosine pairs (hour_sin, hour_cos, weekday_sin, weekday_cos) so that 23 h and 0 h are treated as adjacent
3) Feature scaling – standardise features to zero mean and unit variance
4) MiniBatch K-Means clustering – test k = 3 and k = 4, choose the best using the silhouette score.
5) Cluster interpretation – sort clusters by mean vehicle count / occupancy and assign semantic labels:Very Light → Light → Moderate → Severe
6) Per-camera statistics – compute the proportion of each congestion level per camera to compare Causeway vs Second Link.
7) Outputs – results are written to:
      result_ml/clustered_traffic.csv (full dataset + cluster labels)
      result_ml/cluster_stats.csv (average values per cluster)
      result_ml/per_camera_distribution.csv (state distribution per camera)
This process transforms raw detections into interpretable traffic-state labels that can later serve as ground truth for prediction models.

### 2. Visualization.py — Temporal Pattern Visualization

This script visualises the clustering results from result_ml/clustered_traffic.csv in four complementary plots to reveal temporal regularities.

Visual outputs:
1) Hourly Distribution of Traffic States – stacked bar plot showing each state’s frequency across 24 hours (morning and evening peaks appear clearly).
2) Weekday Distribution – compares congestion frequency across the seven weekdays, highlighting heavier traffic on workdays and smoother flow on weekends.
3) Heatmap of Severe Congestion Ratio (Hour × Weekday) – identifies spatio-temporal “hot zones” of severe congestion, notably weekday morning and evening periods.
4) Average Vehicle Count per Hour by Cluster – line chart illustrating average vehicle counts per cluster, confirming a strong correlation between vehicle density and cluster severity.

Together, these two scripts provide a full unsupervised traffic-pattern analysis pipeline, forming the analytical foundation for the subsequent LSTM congestion-prediction stage.

## LSTM Traffic Congestion Prediction Pipeline  
*(Machine Learning extension modules)*

This repository now includes a full workflow for **traffic congestion clustering and prediction** based on the LTA camera dataset.  
The process converts per-camera vehicle density data into temporal features, trains LSTM classifiers, and evaluates prediction accuracy.

### 1. Data Preparation — `prepare_training_data.py`
This script processes the clustered traffic CSV into training-ready datasets.

**Input:**  
- `scripts/result_ml/clustered_traffic.csv` 

**Main Steps:**  
1. Load and clean the CSV file.  
2. Merge multiple cameras per road using **majority voting** per timestamp.  
3. Add time features (`hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`).  
4. Construct **sliding windows** (`T_IN=24`, `T_OUT=6`) for supervised learning.  
5. Randomly split into 80% training / 20% validation.  
6. Save as `.npz` datasets and `.json` metadata under `scripts/artifacts/phase_cluster/`.

**Output:**  
- `phase_cluster_causeway_dataset.npz`  
- `phase_cluster_second_link_dataset.npz`  
Each file contains: `X_train`, `y_train`, `X_val`, `y_val`.

---

### 2. Model Training — `train_lstm_cluster.py`
Trains **Long Short-Term Memory (LSTM)** classification models to predict future congestion cluster labels.

**Input:**  
- `.npz` datasets from Step 1.

**Model Architecture:**  
- LSTM(64) → Dense(32) → Dense(`t_out * n_classes`) → Reshape → Softmax.  
- Loss: Sparse Categorical Crossentropy  
- Metrics: Accuracy per step and overall.  

**Training Features:**  
- Early stopping (patience=10)  
- Checkpoint saving (`*.h5`)  
- Validation accuracy evaluation  
- Training logs saved as `.json` for later analysis  

**Output:**  
- `causeway_lstm_best.h5`, `second_link_lstm_best.h5`  
- `causeway_train_log.json`, `second_link_train_log.json`

---

### 3. Result Analysis — `Model_analyze.ipynb`
Analyzes and visualizes model training performance.

**Input:**  
- Training logs (`*.json`) generated by Step 2.

**Key Outputs:**  
- loss/accuracy curves
 

This notebook helps identify overfitting, convergence trends, and model reliability.

---

## 🔁 Workflow Summary

```text
clustered_traffic.csv
        │
        ▼
prepare_training_data.py
        │──> phase_cluster_causeway_dataset.npz
        │──> phase_cluster_second_link_dataset.npz
        ▼
train_lstm_cluster.py
        │──> causeway_lstm_best.h5 / second_link_lstm_best.h5
        │──> *_train_log.json
        ▼
Model_analyze.ipynb
        └──> Visualization & Performance Report

```

---
