#!/usr/bin/env python3
"""
DriveBy detect — Legion-side. Runs YOLO tracking over recorded footage,
writes a JSONL sidecar of every detection. Decides nothing; filters nothing.

Usage:
    python detect.py FILE_OR_FOLDER [--model yolo26l.pt] [--conf 0.25]
                                    [--imgsz 960] [--tracker botsort.yaml]

Examples:
    python detect.py drive_20260814_144530.mp4
    python detect.py D:\\DriveRaw --tracker botsort.yaml
    python detect.py D:\\DriveRaw --model yolo26x.pt --force

Output is named per-tracker, e.g. drive_xxx.bytetrack.jsonl / drive_xxx.botsort.jsonl,
so runs with different trackers sit side by side instead of overwriting.

Trackers:
    bytetrack.yaml  fast, motion only. Fragments when the camera pans.
    botsort.yaml    adds appearance ReID + camera motion compensation. Slower.
                    Better for a camera bolted to a moving car.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".webm"}


def jsonl_path(path: Path, tracker: str) -> Path:
    tag = tracker.split(".")[0]
    return path.with_name(f"{path.stem}.{tag}.jsonl")


def detect_one(model, path: Path, conf: float, imgsz: int, device: str,
               tracker: str) -> dict:
    out_path = jsonl_path(path, tracker)
    meta_path = path.with_suffix(".json")

    # Pull recording metadata from the Pi's sidecar if it's alongside.
    rec_meta = {}
    if meta_path.exists():
        try:
            rec_meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"  [warn] could not read {meta_path.name}: {e}")

    fps = rec_meta.get("framerate", 30)
    model_name = Path(model.ckpt_path or "unknown").name

    print(f"\n[{path.name}]")
    print(f"  model={model_name} tracker={tracker} conf={conf} "
          f"imgsz={imgsz} device={device}")

    t0 = time.time()
    n_frames = 0
    n_dets = 0
    seen_tracks = set()
    ok = False

    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            header = {
                "type": "header",
                "source": path.name,
                "model": model_name,
                "tracker": tracker,
                "conf_floor": conf,
                "imgsz": imgsz,
                "framerate": fps,
                "classes": {int(k): v for k, v in model.names.items()},
                "recording": rec_meta,
                "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")

            stream = model.track(
                source=str(path),
                stream=True,
                persist=True,
                tracker=tracker,
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            for frame_idx, result in enumerate(stream):
                n_frames += 1
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue

                ids = boxes.id.int().tolist()
                clss = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                xyxy = boxes.xyxy.tolist()

                for tid, cls, cf, box in zip(ids, clss, confs, xyxy):
                    seen_tracks.add((cls, tid))
                    n_dets += 1
                    fh.write(json.dumps({
                        "f": frame_idx,
                        "t": round(frame_idx / fps, 3),
                        "id": tid,
                        "c": cls,
                        "conf": round(cf, 4),
                        "box": [round(v, 1) for v in box],
                    }) + "\n")

                if n_frames % 300 == 0:
                    el = time.time() - t0
                    print(f"  {n_frames} frames | {n_frames/el:.1f} fps | "
                          f"{len(seen_tracks)} tracks", flush=True)
        ok = True

    finally:
        # Don't leave a half-written sidecar behind — it would look "done"
        # on the next run and get skipped.
        if not ok and out_path.exists():
            try:
                out_path.unlink()
                print(f"  [cleanup] removed incomplete {out_path.name}")
            except Exception:
                pass

    el = time.time() - t0
    print(f"  done: {n_frames} frames in {el:.1f}s "
          f"({n_frames/max(el, 0.001):.1f} fps)")
    print(f"  {n_dets} detections, {len(seen_tracks)} unique tracks "
          f"-> {out_path.name}")

    counts = {}
    for cls, _tid in seen_tracks:
        counts[cls] = counts.get(cls, 0) + 1
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {model.names[cls]}: {n}")

    return {"frames": n_frames, "dets": n_dets, "tracks": len(seen_tracks)}


def main():
    ap = argparse.ArgumentParser(description="DriveBy detection pass")
    ap.add_argument("path", help="video file or folder of videos")
    ap.add_argument("--model", default="yolo26l.pt",
                    help="weights (default yolo26l.pt)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence floor, keep LOW - filter at render "
                         "(default 0.25)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="inference size (default 960)")
    ap.add_argument("--device", default="0", help="cuda device index, or 'cpu'")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"],
                    help="botsort adds appearance ReID + camera motion "
                         "compensation; better for a moving camera")
    ap.add_argument("--force", action="store_true",
                    help="redo files that already have a sidecar")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        sys.exit(f"not found: {target}")

    if target.is_dir():
        files = sorted(p for p in target.iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)
    else:
        if target.suffix.lower() == ".h264":
            sys.exit("Ultralytics won't read raw .h264 — remux first:\n"
                     f"  ffmpeg -framerate 30 -i {target.name} -c copy "
                     f"{target.stem}.mp4")
        files = [target]

    if not args.force:
        done = [p for p in files if jsonl_path(p, args.tracker).exists()]
        files = [p for p in files if not jsonl_path(p, args.tracker).exists()]
        for p in done:
            print(f"[skip] {p.name} already has "
                  f"{jsonl_path(p, args.tracker).name} (--force to redo)")

    if not files:
        sys.exit("nothing to do")

    print(f"loading {args.model} ...")
    model = YOLO(args.model)

    grand = {"frames": 0, "dets": 0, "tracks": 0}
    t0 = time.time()
    for p in files:
        try:
            r = detect_one(model, p, args.conf, args.imgsz, args.device,
                           args.tracker)
            for k in grand:
                grand[k] += r[k]
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")

    print(f"\n=== {len(files)} file(s) in {time.time()-t0:.1f}s | "
          f"{grand['frames']} frames, {grand['dets']} detections, "
          f"{grand['tracks']} tracks ===")


if __name__ == "__main__":
    main()
