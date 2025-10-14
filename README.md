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

## 使用说明（中文）

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
