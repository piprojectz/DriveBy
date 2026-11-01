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
GPS_PORT        = "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7_-_GPS_GNSS_Receiver-if00"
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
    """Read NMEA straight from the USB GPS and append fixes to the current
    chunk's .gps.jsonl. Reconnects if the dongle drops or goes quiet."""
    import serial

    def dm(v, h):
        if not v: return None
        d = 2 if h in "NS" else 3
        x = float(v[:d]) + float(v[d:]) / 60
        return round(-x if h in "SW" else x, 7)

    gga = {}
    while running:
        try:
            with serial.Serial(GPS_PORT, 9600, timeout=2) as s:
                print("[GPS] Connected to dongle", flush=True)
                quiet = 0
                while running:
                    raw = s.readline()
                    if not raw:
                        quiet += 1
                        if quiet >= 5:
                            raise OSError("no GPS data for 10s")
                        continue
                    quiet = 0
                    line = raw.decode("ascii", "ignore").strip()
                    if not line.startswith("$") or "*" not in line:
                        continue
                    body, cs = line[1:].split("*", 1)
                    c = 0
                    for ch in body: c ^= ord(ch)
                    if f"{c:02X}" != cs[:2].upper():
                        continue
                    p = body.split(",")
                    t = p[0][2:]
                    if t == "GGA" and len(p) > 9:
                        gga = {"sats": p[7], "hdop": p[8], "alt": p[9]}
                    elif t == "RMC" and len(p) > 9 and p[2] == "A":
                        d = p[9]
                        utc = f"20{d[4:6]}-{d[2:4]}-{d[0:2]}T{p[1][0:2]}:{p[1][2:4]}:{float(p[1][4:]):06.3f}Z"
                        rec = {
                            "t_mono": time.monotonic(),
                            "utc": utc,
                            "lat": dm(p[3], p[4]),
                            "lon": dm(p[5], p[6]),
                            "alt_m": float(gga["alt"]) if gga.get("alt") else None,
                            "speed_mps": round(float(p[7]) * 0.514444, 3) if p[7] else None,
                            "track_deg": float(p[8]) if p[8] else None,
                            "mode": 3 if gga.get("alt") else 2,
                            "sats": int(gga["sats"]) if gga.get("sats") else None,
                            "hdop": float(gga["hdop"]) if gga.get("hdop") else None,
                        }
                        with gps_lock:
                            if gps_fh is not None:
                                gps_fh.write(json.dumps(rec) + "\n")
                                gps_fh.flush()
        except OSError as e:
            if running:
                print(f"[GPS] {e}; retrying in 3s", flush=True)
                time.sleep(3)


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
