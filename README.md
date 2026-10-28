
# DriveBy

A Raspberry Pi 5 dashcam that records drives with synced GPS, plus an on-device object-detection mode with a live Russian-language leaderboard overlay.

## Scripts

### `record.py` — record-only mode (current)

Lean capture for the car, with no detection on the Pi.

- 1080p30 H.264 in 10-minute chunks (~90 MB/min at 12 Mbps)

- Per-chunk `.json` metadata (start time in local/UTC/monotonic clocks, resolution, bitrate)

- Per-chunk `.gps.jsonl` sidecar read straight from gpsd's JSON socket: lat/lon, altitude, speed, heading, fix mode, error estimate

- GPS timestamps share the same monotonic clock as the video, so a later processing step can line up position with frames

- Auto-reconnects if gpsd drops; closes the current chunk cleanly on Ctrl+C / SIGTERM

Output goes to `~/Videos/DriveRaw/`.

### `DriveBy.py` — on-device detection mode

The original all-in-one version that runs detection live on the Pi.

- YOLOv8n (or a custom model if one is found) runs every 3rd frame at 640x480

- Deduplicates objects across frames using color histograms + bounding-box overlap, so the same car isn't counted twice

- Draws boxes and a live leaderboard (count + confidence bars) with Russian labels for all 80 COCO classes

- Writes MP4 chunks and a per-chunk count summary to `~/Videos/DriveCounts/`

### `gps_test.py`

Quick check that the GPS is producing data.

## Hardware

- Raspberry Pi 5

- Raspberry Pi Camera Module

- USB GPS receiver, served through `gpsd`

- HyperPixel touchscreen (for the live display in `DriveBy.py`)

## Setup

```

sudo apt install python3-picamera2 gpsd gpsd-clients fonts-dejavu-core

```

`DriveBy.py` also needs `ultralytics` and `opencv-python`. It loads them from `~/packages`.

## Usage

```

python3 record.py     # record-only

python3 DriveBy.py    # live detection + overlay (press q to quit)

```

## Architecture

Recording now happens on the Pi with `record.py`. Detection, tracking, and rendering run afterward on a separate GPU machine, which isn't part of this repo yet.

## Privacy

Recordings and GPS logs contain location data. They're saved outside this folder and excluded by `.gitignore`.

