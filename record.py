#!/usr/bin/env python3
"""DriveBy record-only. Captures H.264 chunks + timing and GPS sidecars. No detection."""

import json, os, signal, socket, threading, time
from datetime import datetime, timezone
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# ---------- CONFIG ----------
RESOLUTION      = (1920, 1080)
FRAMERATE       = 30
BITRATE         = 12_000_000       # ~90 MB/min
CHUNK_MINUTES   = 10
SAVE_DIR        = os.path.expanduser("~/Videos/DriveRaw")
GPSD_HOST       = "127.0.0.1"
GPSD_PORT       = 2947
# ----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
running = True

# Current chunk's GPS sidecar; swapped by the main loop at chunk boundaries.
gps_lock = threading.Lock()
gps_fh = None


def stop(signum, frame):
    global running
    running = False
    print("\n[INFO] Stop requested, closing current chunk...", flush=True)


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


# ---------- GPS LOGGER ----------
def gps_logger():
    """Read TPV reports straight from gpsd's JSON socket and append them to
    the current chunk's .gps.jsonl. Reconnects if gpsd drops or goes quiet."""
    while running:
        s = None
        try:
            s = socket.create_connection((GPSD_HOST, GPSD_PORT), timeout=5)
            s.sendall(b'?WATCH={"enable":true,"json":true};\n')
            stream = s.makefile("r", encoding="utf-8", errors="replace")
            print("[GPS] Connected to gpsd", flush=True)
            last_utc = None

            for line in stream:
                if not running:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("class") != "TPV" or msg.get("mode", 0) < 2:
                    continue   # skip non-position reports and no-fix
                if msg.get("time") == last_utc:
                    continue   # gpsd sends several TPVs per GPS epoch; keep one
                last_utc = msg.get("time")

                rec = {
                    "t_mono": time.monotonic(),          # same clock as started_monotonic
                    "utc": msg.get("time"),
                    "lat": msg.get("lat"),
                    "lon": msg.get("lon"),
                    "alt_m": msg.get("altMSL", msg.get("alt")),
                    "speed_mps": msg.get("speed"),
                    "track_deg": msg.get("track"),
                    "mode": msg.get("mode"),             # 2 = 2D fix, 3 = 3D fix
                    "eph_m": msg.get("eph"),             # horizontal error estimate
                }
                with gps_lock:
                    if gps_fh is not None:
                        gps_fh.write(json.dumps(rec) + "\n")
                        gps_fh.flush()
        except OSError as e:
            if running:
                print(f"[GPS] {e}; retrying in 3s", flush=True)
                time.sleep(3)
        finally:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


threading.Thread(target=gps_logger, daemon=True).start()
# --------------------------------

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": RESOLUTION},
    controls={"FrameRate": FRAMERATE},
)
picam2.configure(config)
encoder = H264Encoder(bitrate=BITRATE)

picam2.start()
time.sleep(2)   # let AE/AWB settle before the first chunk
print(f"[INFO] Camera up. {RESOLUTION[0]}x{RESOLUTION[1]} @ {FRAMERATE}fps -> {SAVE_DIR}", flush=True)

chunk_no = 0
try:
    while running:
        chunk_no += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(SAVE_DIR, f"drive_{stamp}")
        h264 = base + ".h264"
        gps_path = base + ".gps.jsonl"

        meta = {
            "chunk": chunk_no,
            "file": os.path.basename(h264),
            "gps_file": os.path.basename(gps_path),
            "started_local": datetime.now().isoformat(timespec="seconds"),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "started_monotonic": time.monotonic(),
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "framerate": FRAMERATE,
            "bitrate": BITRATE,
        }
        with open(base + ".json", "w") as f:
            json.dump(meta, f, indent=2)

        with gps_lock:
            gps_fh = open(gps_path, "w")

        encoder.output = FileOutput(h264)
        picam2.start_encoder(encoder)
        print(f"[INFO] Chunk {chunk_no} recording: {os.path.basename(h264)}", flush=True)

        t_end = time.monotonic() + CHUNK_MINUTES * 60
        while running and time.monotonic() < t_end:
            time.sleep(0.5)

        picam2.stop_encoder()

        with gps_lock:
            gps_fh.close()
            gps_fh = None

        size_mb = os.path.getsize(h264) / 1e6
        with open(gps_path) as g:
            n_fix = sum(1 for _ in g)
        print(f"[INFO] Chunk {chunk_no} closed: {size_mb:.1f} MB, {n_fix} GPS fixes", flush=True)

finally:
    with gps_lock:
        if gps_fh is not None:
            gps_fh.close()
            gps_fh = None
    try:
        picam2.stop()
    except Exception as e:
        print(f"[WARN] camera stop: {e}")
    print("[INFO] Остановка записи.", flush=True)
