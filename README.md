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
