#!/usr/bin/env python3
"""
DriveBy render — video + detection sidecar (+ optional GPS sidecar) -> annotated MP4.

All filtering happens HERE, not at detect time, so you can change your mind
without re-running the GPU pass.

Single chunk:
    python render.py VIDEO [--conf 0.4] [--min-frames 30] [--classes vehicles]

Whole drive, leaderboard carries across chunks:
    python render.py --drive D:\\DriveRaw

GPS overlays turn on automatically if VIDEO.gps.jsonl exists beside the video:
    bottom right   heading-up minimap, drifting car marker, speed
    above it       static route overview with a position dot

    --no-map            turn the minimap off
    --no-overview       heading-up map only
    --map-zoom 400      metres from car to map edge (bigger = zoomed out)
    --map-car-y 0.35    car position top-to-bottom; lower value = more
                        trailing road visible behind you
    --drift 1.0         drift exaggeration, 0 = none
    --drift-lag 0.06    how fast the car catches up, smaller = floatier
    --units mph         mph or kmh

Note: a vehicle in frame across a chunk boundary is counted twice — track IDs
do not survive between files. At 10-minute chunks this is a handful per drive.
"""

import argparse
import json
import math
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- appearance
BOX_COLOR      = (184, 162, 23)
BAR_COLOR      = (111, 66, 193)
LABEL_COLOR    = (232, 62, 140)
LB_TEXT_COLOR  = (255, 255, 255)
LB_WIDTH       = 260
LB_ROW_H       = 30
LB_MARGIN      = 12

MAP_BG         = (18, 18, 22)
MAP_EDGE       = (90, 90, 100)
TRACK_PAST     = (111, 66, 193)
TRACK_AHEAD    = (120, 120, 135)
CAR_BODY       = (232, 62, 140)
CAR_TRIM       = (255, 255, 255)
OVER_BG        = (18, 18, 22)
OVER_TRACK     = (90, 90, 105)
OVER_DONE      = (111, 66, 193)
OVER_DOT       = (232, 62, 140)
SPEED_COLOR    = (255, 255, 255)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

RU = {
    0: "человек", 1: "велосипед", 2: "машина", 3: "мотоцикл", 4: "самолет",
    5: "автобус", 6: "поезд", 7: "грузовик", 8: "лодка", 9: "светофор",
    10: "гидрант", 11: "знак стоп", 12: "паркомат", 13: "скамейка",
    14: "птица", 15: "кот", 16: "собака", 17: "лошадь", 18: "овца",
    19: "корова", 20: "слон", 21: "медведь", 22: "зебра", 23: "жираф",
    24: "рюкзак", 25: "зонт", 26: "сумка", 28: "чемодан", 32: "мяч",
    56: "стул", 57: "диван", 58: "растение", 60: "стол", 62: "телевизор",
    63: "ноутбук", 67: "телефон", 73: "книга", 74: "часы",
}

CLASS_GROUPS = {
    "vehicles": [1, 2, 3, 5, 6, 7],
    "road":     [0, 1, 2, 3, 5, 6, 7, 9, 11],
    "animals":  [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


# ------------------------------------------------------------------ helpers

def find_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def find_jsonl(video: Path, explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"jsonl not found: {p}")
        return p
    for suffix in (".bytetrack.jsonl", ".botsort.jsonl", ".jsonl"):
        p = video.with_name(video.stem + suffix)
        if p.exists():
            return p
    return None


def load(jsonl: Path):
    header = {}
    by_frame = defaultdict(list)
    track_len = defaultdict(int)
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
                continue
            by_frame[rec["f"]].append(rec)
            track_len[(rec["c"], rec["id"])] += 1
    return header, by_frame, track_len


def parse_classes(arg, names):
    if not arg:
        return None
    if arg in CLASS_GROUPS:
        return set(CLASS_GROUPS[arg])
    out = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            out.add(int(tok))
        else:
            hit = [int(k) for k, v in names.items() if v.lower() == tok.lower()]
            if not hit:
                sys.exit(f"unknown class: {tok}")
            out.update(hit)
    return out


# ---------------------------------------------------------------------- GPS

class Track:
    """GPS fixes with per-frame interpolation and local metre projection."""

    def __init__(self, path: Path):
        self.header = {}
        self.t, self.lat, self.lon, self.spd, self.hdg = [], [], [], [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("type") == "header":
                    self.header = r
                    continue
                self.t.append(r["t"])
                self.lat.append(r["lat"])
                self.lon.append(r["lon"])
                self.spd.append(r.get("speed", 0.0))
                self.hdg.append(r.get("heading", 0.0))

        if len(self.t) < 2:
            raise ValueError("need at least 2 GPS fixes")

        self.lat_a = np.array(self.lat)
        self.lon_a = np.array(self.lon)
        self.lat0 = float(np.mean(self.lat_a))
        # metres per degree at this latitude
        self.m_lat = 111_320.0
        self.m_lon = 111_320.0 * math.cos(math.radians(self.lat0))
        # projected once, reused every frame
        self.x = (self.lon_a - self.lon_a[0]) * self.m_lon
        self.y = (self.lat_a - self.lat_a[0]) * self.m_lat

    @property
    def duration(self):
        return self.t[-1]

    def at(self, sec):
        """Interpolated (x, y, speed, heading, index) at time `sec`."""
        i = bisect_left(self.t, sec)
        if i <= 0:
            return self.x[0], self.y[0], self.spd[0], self.hdg[0], 0
        if i >= len(self.t):
            n = len(self.t) - 1
            return self.x[n], self.y[n], self.spd[n], self.hdg[n], n
        t0, t1 = self.t[i - 1], self.t[i]
        f = 0.0 if t1 == t0 else (sec - t0) / (t1 - t0)
        x = self.x[i - 1] + (self.x[i] - self.x[i - 1]) * f
        y = self.y[i - 1] + (self.y[i] - self.y[i - 1]) * f
        s = self.spd[i - 1] + (self.spd[i] - self.spd[i - 1]) * f
        # circular interpolation for heading
        h0, h1 = self.hdg[i - 1], self.hdg[i]
        d = ((h1 - h0 + 180) % 360) - 180
        h = (h0 + d * f) % 360
        return x, y, s, h, i


def _rot(px, py, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return px * c - py * s, px * s + py * c


def draw_minimap(img, track, sec, box, args, visual_hdg, font):
    """Heading-up minimap, rendered into its own tile then pasted so that
    every line clips to the box. `box` is (x0, y0, size)."""
    x0, y0, size = box
    size = int(size)
    tile = Image.new("RGB", (size, size), MAP_BG)
    draw = ImageDraw.Draw(tile)
    cx = size / 2
    cy = size * args.map_car_y

    px, py, speed, hdg, idx = track.at(sec)
    scale = (size / 2) / max(args.map_zoom, 1)

    # points within a generous radius of the car
    dx = track.x - px
    dy = track.y - py
    keep = (np.abs(dx) < args.map_zoom * 2.2) & (np.abs(dy) < args.map_zoom * 2.2)
    ids = np.nonzero(keep)[0]

    if len(ids) > 1:
        a = math.radians(hdg)          # rotate so heading points up
        ca, sa = math.cos(a), math.sin(a)
        rx = dx[ids] * ca - dy[ids] * sa
        ry = dx[ids] * sa + dy[ids] * ca
        sx = cx + rx * scale
        sy = cy + ry * scale           # +y screen is down, which is behind us

        past, ahead = [], []
        for k, i in enumerate(ids):
            pt = (float(sx[k]), float(sy[k]))
            (past if i <= idx else ahead).append(pt)

        if len(ahead) > 1:
            draw.line(ahead, fill=TRACK_AHEAD, width=4, joint="curve")
        if len(past) > 1:
            draw.line(past, fill=TRACK_PAST, width=5, joint="curve")

    # ---- car marker, rotated by the drift offset
    drift = ((visual_hdg - hdg + 180) % 360) - 180
    drift = max(-45.0, min(45.0, drift * args.drift))

    L = size * 0.085          # half length
    W = size * 0.042          # half width
    body = [(-W, -L), (W, -L), (W * 1.15, L * 0.35), (W * 0.8, L),
            (-W * 0.8, L), (-W * 1.15, L * 0.35)]
    pts = []
    for bx, by in body:
        rxp, ryp = _rot(bx, by, drift)
        pts.append((cx + rxp, cy + ryp))
    draw.polygon(pts, fill=CAR_BODY, outline=CAR_TRIM)

    # nose marker
    nx, ny = _rot(0, -L * 1.35, drift)
    draw.ellipse([cx + nx - 3, cy + ny - 3, cx + nx + 3, cy + ny + 3],
                 fill=CAR_TRIM)

    # ---- speed
    val = speed if args.units == "mph" else speed * 1.60934
    txt = f"{val:.0f} {args.units}"
    draw.text((10, size - font.size - 10), txt,
              font=font, fill=SPEED_COLOR, stroke_width=2,
              stroke_fill=(0, 0, 0))

    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=10,
                           outline=MAP_EDGE, width=2)
    img.paste(tile, (int(x0), int(y0)))
    return drift


def draw_overview(img, ov, sec, track, box, font):
    """Static route with a moving dot, clipped to its own tile."""
    x0, y0, w, h = box
    w, h = int(w), int(h)
    tile = Image.new("RGB", (w, h), OVER_BG)
    draw = ImageDraw.Draw(tile)
    pts, idx_of = ov
    if len(pts) > 1:
        draw.line(pts, fill=OVER_TRACK, width=3, joint="curve")
        _, _, _, _, i = track.at(sec)
        k = idx_of(i)
        if k > 1:
            draw.line(pts[:k], fill=OVER_DONE, width=3, joint="curve")
        dx, dy = pts[min(k, len(pts) - 1)]
        draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=OVER_DOT,
                     outline=(255, 255, 255), width=2)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=10,
                           outline=MAP_EDGE, width=2)
    img.paste(tile, (int(x0), int(y0)))


def build_overview(track, w, h, pad=12):
    """Project the whole track into the overview box once."""
    xs, ys = track.x, track.y
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    spanx = max(maxx - minx, 1.0)
    spany = max(maxy - miny, 1.0)
    s = min((w - 2 * pad) / spanx, (h - 2 * pad) / spany)
    offx = (w - spanx * s) / 2
    offy = (h - spany * s) / 2

    # subsample so the polyline stays cheap
    step = max(1, len(xs) // 400)
    idxs = list(range(0, len(xs), step))
    pts = [(offx + (xs[i] - minx) * s,
            offy + (maxy - ys[i]) * s) for i in idxs]

    def idx_of(i):
        return min(int(i / step), len(pts) - 1)

    return pts, idx_of


# -------------------------------------------------------------------- render

def render_one(video: Path, args, carry: dict, jsonl_override=None):
    jsonl = find_jsonl(video, jsonl_override)
    if jsonl is None:
        print(f"[skip] {video.name} — no sidecar, run detect.py first")
        return carry

    header, by_frame, track_len = load(jsonl)
    names = {int(k): v for k, v in header.get("classes", {}).items()}
    keep_classes = parse_classes(args.classes, names)

    def label_for(cls):
        if args.lang == "ru" and cls in RU:
            return RU[cls]
        return names.get(cls, str(cls))

    good = {
        key for key, n in track_len.items()
        if n >= args.min_frames
        and (keep_classes is None or key[0] in keep_classes)
    }

    chunk_counts = defaultdict(int)
    for cls, _tid in good:
        chunk_counts[cls] += 1

    # ---- GPS
    track = None
    gps_path = video.with_name(video.stem + ".gps.jsonl")
    if gps_path.exists() and not args.no_map:
        try:
            track = Track(gps_path)
            print(f"  gps: {len(track.t)} fixes, {track.duration:.0f}s "
                  f"({gps_path.name})")
        except Exception as e:
            print(f"  [warn] gps load failed: {e}")

    print(f"\n[{video.name}]")
    print(f"  sidecar {jsonl.name} | {header.get('model','?')} / "
          f"{header.get('tracker','?')}")
    print(f"  conf>={args.conf} min_frames>={args.min_frames} "
          f"classes={args.classes or 'all'}")
    print(f"  {len(good)} tracks kept of {len(track_len)}")
    for cls, n in sorted(chunk_counts.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {label_for(cls)}: {n}  (drive total {carry.get(cls, 0) + n})")

    summary_path = video.with_name(video.stem + ".counts.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"{video.name}\n")
        fh.write(f"model {header.get('model','?')} "
                 f"tracker {header.get('tracker','?')}\n")
        fh.write(f"conf>={args.conf} min_frames>={args.min_frames}\n\n")
        fh.write("this chunk:\n")
        for cls, n in sorted(chunk_counts.items(), key=lambda kv: -kv[1]):
            fh.write(f"  {label_for(cls)}: {n}\n")
        if carry:
            tot = defaultdict(int, carry)
            for cls, n in chunk_counts.items():
                tot[cls] += n
            fh.write("\ndrive total through this chunk:\n")
            for cls, n in sorted(tot.items(), key=lambda kv: -kv[1]):
                fh.write(f"  {label_for(cls)}: {n}\n")

    if args.summary_only:
        out = defaultdict(int, carry)
        for cls, n in chunk_counts.items():
            out[cls] += n
        return dict(out)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [ERROR] could not open {video.name}")
        return carry
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or header.get("framerate", 30)

    out_path = (Path(args.out) if args.out
                else video.with_name(video.stem + ".annotated.mp4"))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    label_font = find_font(max(16, w // 70))
    lb_font = find_font(max(15, w // 80))
    speed_font = find_font(max(18, w // 55))

    # map geometry
    map_size = int(w * 0.16)
    map_x = w - map_size - LB_MARGIN
    map_y = h - map_size - LB_MARGIN
    ov_h = int(map_size * 0.55)
    ov_y = map_y - ov_h - 8
    overview = None
    if track is not None and not args.no_overview:
        overview = build_overview(track, map_size, ov_h)

    seen = set()
    running = defaultdict(int, carry)
    frame_idx = 0
    visual_hdg = None

    print(f"  rendering -> {out_path.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = [d for d in by_frame.get(frame_idx, [])
                if d["conf"] >= args.conf and (d["c"], d["id"]) in good]
        for d in dets:
            key = (d["c"], d["id"])
            if key not in seen:
                seen.add(key)
                running[d["c"]] += 1

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        if not args.no_boxes:
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
                txt = f"{label_for(d['c'])} #{d['id']} {d['conf']:.2f}"
                ty = max(0, y1 - label_font.size - 4)
                draw.text((x1, ty), txt, font=label_font, fill=LABEL_COLOR,
                          stroke_width=2, stroke_fill=(0, 0, 0))

        if not args.no_leaderboard and running:
            x0 = w - LB_WIDTH - LB_MARGIN
            top = max(running.values())
            rows = sorted(running.items(), key=lambda kv: -kv[1])[:args.lb_rows]
            for i, (cls, n) in enumerate(rows):
                y = LB_MARGIN + i * LB_ROW_H
                bar_w = LB_WIDTH - 10
                draw.rectangle([x0, y, x0 + bar_w, y + LB_ROW_H - 6],
                               outline=(0, 0, 0), width=2)
                fill_w = int((n / max(top, 1)) * bar_w)
                if fill_w > 2:
                    draw.rectangle([x0, y, x0 + fill_w, y + LB_ROW_H - 6],
                                   fill=BAR_COLOR)
                draw.text((x0 + 6, y + 2), f"{label_for(cls)}: {n}",
                          font=lb_font, fill=LB_TEXT_COLOR,
                          stroke_width=2, stroke_fill=(0, 0, 0))

        if track is not None:
            sec = frame_idx / fps
            _, _, _, true_hdg, _ = track.at(sec)
            if visual_hdg is None:
                visual_hdg = true_hdg
            else:
                d = ((true_hdg - visual_hdg + 180) % 360) - 180
                visual_hdg = (visual_hdg + d * args.drift_lag) % 360

            if overview is not None:
                draw_overview(img, overview, sec, track,
                              (map_x, ov_y, map_size, ov_h), lb_font)
            draw_minimap(img, track, sec, (map_x, map_y, map_size),
                         args, visual_hdg, speed_font)
            draw = ImageDraw.Draw(img)

        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        frame_idx += 1
        if frame_idx % 600 == 0:
            print(f"    {frame_idx} frames", flush=True)

    cap.release()
    writer.release()
    print(f"  done: {frame_idx} frames -> {out_path.name}")
    return dict(running)


def main():
    ap = argparse.ArgumentParser(description="DriveBy render pass")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--drive", default=None,
                    help="folder of chunks; renders all in order with a "
                         "leaderboard that carries across them")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--classes", default=None,
                    help="ids/names, or a group: " + ", ".join(CLASS_GROUPS))
    ap.add_argument("--lang", default="ru", choices=["ru", "en"])
    ap.add_argument("--lb-rows", type=int, default=8)
    ap.add_argument("--no-leaderboard", action="store_true")
    ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--carry-in", default=None)
    # --- map
    ap.add_argument("--no-map", action="store_true",
                    help="disable GPS overlays entirely")
    ap.add_argument("--no-overview", action="store_true",
                    help="heading-up map only, no route overview")
    ap.add_argument("--map-zoom", type=float, default=400,
                    help="metres from car to map edge (default 400)")
    ap.add_argument("--map-car-y", type=float, default=0.35,
                    help="car position in the box, 0=top 1=bottom; lower "
                         "shows more trailing road (default 0.35)")
    ap.add_argument("--drift", type=float, default=1.0,
                    help="drift exaggeration, 0 disables (default 1.0)")
    ap.add_argument("--drift-lag", type=float, default=0.06,
                    help="how fast the car catches up, smaller = floatier "
                         "(default 0.06)")
    ap.add_argument("--units", default="mph", choices=["mph", "kmh"])
    args = ap.parse_args()

    if not args.drive and not args.video:
        sys.exit("give a video file, or --drive FOLDER")

    if args.drive:
        folder = Path(args.drive)
        if not folder.is_dir():
            sys.exit(f"not a folder: {folder}")
        chunks = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in VIDEO_EXTS
                        and ".annotated" not in p.name
                        and find_jsonl(p) is not None)
        if not chunks:
            sys.exit("no chunks with sidecars found — run detect.py first")

        print(f"drive mode: {len(chunks)} chunk(s) in {folder}")
        carry = {}
        for c in chunks:
            args.out = None
            carry = render_one(c, args, carry)

        names = {}
        j = find_jsonl(chunks[-1])
        if j:
            hdr, _, _ = load(j)
            names = {int(k): v for k, v in hdr.get("classes", {}).items()}

        def label_for(cls):
            if args.lang == "ru" and cls in RU:
                return RU[cls]
            return names.get(cls, str(cls))

        total_path = folder / "drive_totals.txt"
        print("\n=== DRIVE TOTAL ===")
        with open(total_path, "w", encoding="utf-8") as fh:
            fh.write(f"{len(chunks)} chunks\n")
            fh.write(f"conf>={args.conf} min_frames>={args.min_frames}\n\n")
            for cls, n in sorted(carry.items(), key=lambda kv: -kv[1]):
                line = f"{label_for(cls)}: {n}"
                print(f"  {line}")
                fh.write(line + "\n")
        print(f"\nwrote {total_path.name}")
        return

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"not found: {video}")

    carry = {}
    if args.carry_in:
        p = Path(args.carry_in)
        if p.exists():
            carry = {int(k): v for k, v in json.loads(p.read_text()).items()}

    final = render_one(video, args, carry, args.jsonl)
    video.with_name(video.stem + ".carry.json").write_text(
        json.dumps({str(k): v for k, v in final.items()}, indent=2))


if __name__ == "__main__":
    main()
