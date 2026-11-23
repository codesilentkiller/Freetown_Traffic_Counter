"""Detect, track and count road users in a video (CPU-friendly).

Pipeline, per frame:
  1. YOLO detects objects (boxes + class + confidence).
  2. ByteTrack links detections across frames, giving each object a stable ID
     (so one car seen in 100 frames is one car, not 100).
  3. LineZone counts an object when its centre point crosses the counting line.
     We tally crossings per class and per direction ("in" / "out").
  4. The frame is drawn on and written to the output video.

Outputs in <output.dir>/<video name>/ :
    <name>_annotated.mp4    boxes, labels, track IDs, counting line, running counts
    <name>_counts.csv       class, in, out, total
    <name>_detections.json  per-frame list of detections with track IDs

Examples:
    python scripts/count_traffic.py --video clip.mp4 --model models/best.pt
    python scripts/count_traffic.py --video clip.mp4 --orientation vertical --position 0.4 --flip
    python scripts/count_traffic.py --video clip.mp4 --stride 2 --max-frames 300   # quick test
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "model": "models/best.pt",
    "inference": {"imgsz": 416, "conf": 0.35, "iou": 0.5, "device": "cpu", "vid_stride": 1, "max_frames": None},
    "line": {"orientation": "horizontal", "position": 0.5, "flip": False},
    "output": {"dir": "outputs", "save_video": True, "save_csv": True, "save_json": True},
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides: dict | None = None) -> dict:
    """Defaults < config.yaml < overrides (command line / app widgets)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return _merge(cfg, overrides or {})


def build_line(orientation: str, position: float, width: int, height: int, flip: bool = False):
    """Return ((x1, y1), (x2, y2)) pixel endpoints of the counting line."""
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("line.orientation must be 'horizontal' or 'vertical'")
    if not 0.0 <= position <= 1.0:
        raise ValueError("line.position must be between 0 and 1")
    if orientation == "horizontal":
        y = int(round(position * height))
        start, end = (0, y), (width, y)
    else:
        x = int(round(position * width))
        start, end = (x, 0), (x, height)
    return (end, start) if flip else (start, end)


# --------------------------------------------------------------------------- #
# Video encoding
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str | None:
    """System FFmpeg if installed, otherwise the copy bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode src to browser-playable H.264 (yuv420p, faststart). True on success."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0 and dst.exists()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def process_video(video_path, model_path, cfg: dict, out_dir=None, progress_cb=None) -> dict:
    """Run the full pipeline on one video. Used by the CLI and by the Streamlit app.

    progress_cb(done, total) is called every few frames so a UI can show progress
    (the video is streamed frame by frame, so memory use stays flat for long files).
    Returns a dict with output paths, per-class counts and timing.
    """
    # Heavy imports live here so `--help` and config handling work without them.
    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    video_path, model_path = Path(video_path), Path(model_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    inf, line_cfg, out_cfg = cfg["inference"], cfg["line"], cfg["output"]
    stride = max(1, int(inf["vid_stride"]))
    max_frames = inf.get("max_frames")
    out_dir = Path(out_dir) if out_dir else Path(out_cfg["dir"]) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    model = YOLO(str(model_path))
    names = {int(k): v for k, v in model.names.items()}
    nc = len(names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_est = -(-n_total // stride) if n_total else 0
    if max_frames:
        total_est = min(total_est, int(max_frames)) if total_est else int(max_frames)
    eff_fps = fps / stride                       # fps of the frames we actually process

    start, end = build_line(line_cfg["orientation"], float(line_cfg["position"]), width, height, bool(line_cfg["flip"]))
    line_zone = sv.LineZone(start=sv.Point(*start), end=sv.Point(*end), triggering_anchors=[sv.Position.CENTER])
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(eff_fps))))
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.6)

    counts_in, counts_out = np.zeros(nc, dtype=int), np.zeros(nc, dtype=int)

    raw_path = out_dir / f"{stem}_raw.mp4"
    final_path = out_dir / f"{stem}_annotated.mp4"
    writer = None
    if out_cfg.get("save_video", True):
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (width, height))
    json_path = out_dir / f"{stem}_detections.json"
    jf = open(json_path, "w", encoding="utf-8") if out_cfg.get("save_json", True) else None
    if jf:
        jf.write("[\n")

    def draw_counts(frame):
        """Small running-total panel in the top-left corner (classes with crossings so far)."""
        y = 24
        for cid in range(nc):
            if counts_in[cid] or counts_out[cid]:
                text = f"{names[cid]}: in {counts_in[cid]} / out {counts_out[cid]}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 22

    t0, frame_idx, processed, first_json = time.time(), -1, 0, True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % stride:
                continue
            if max_frames and processed >= int(max_frames):
                break

            result = model.predict(frame, imgsz=int(inf["imgsz"]), conf=float(inf["conf"]), iou=float(inf["iou"]),
                                   device=inf.get("device", "cpu"), verbose=False)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))

            if len(detections):
                crossed_in, crossed_out = line_zone.trigger(detections)
                for cid in detections.class_id[crossed_in]:
                    counts_in[int(cid)] += 1
                for cid in detections.class_id[crossed_out]:
                    counts_out[int(cid)] += 1

            if writer is not None:
                labels = [f"#{tid} {names[int(c)]} {conf:.2f}"
                          for tid, c, conf in zip(detections.tracker_id, detections.class_id,
                                                  detections.confidence if detections.confidence is not None
                                                  else [0.0] * len(detections))]
                annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                annotated = line_annotator.annotate(annotated, line_counter=line_zone)
                draw_counts(annotated)
                writer.write(annotated)

            if jf is not None:
                rec = {"frame": frame_idx, "time_s": round(frame_idx / fps, 3), "detections": [
                    {"track_id": int(tid), "class_id": int(c), "class_name": names[int(c)],
                     "confidence": round(float(cf), 4), "bbox": [round(float(v), 1) for v in box]}
                    for box, c, tid, cf in zip(detections.xyxy, detections.class_id, detections.tracker_id,
                                               detections.confidence if detections.confidence is not None
                                               else [0.0] * len(detections))]}
                jf.write(("" if first_json else ",\n") + json.dumps(rec))
                first_json = False

            processed += 1
            if progress_cb and (processed % 5 == 0):
                progress_cb(processed, total_est)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if jf:
            jf.write("\n]\n")
            jf.close()
    if progress_cb:
        progress_cb(processed, max(total_est, processed))

    video_out, warning = None, None
    if writer is not None:
        if to_h264(raw_path, final_path):
            raw_path.unlink(missing_ok=True)
            video_out = final_path
        else:                                    # no FFmpeg: keep mp4v (many browsers cannot play it)
            raw_path.replace(final_path)
            video_out = final_path
            warning = "FFmpeg not found: video saved as MPEG-4 Part 2, which some browsers cannot play. " \
                      "Install FFmpeg or `pip install imageio-ffmpeg` for H.264."

    rows = [{"class": names[c], "in": int(counts_in[c]), "out": int(counts_out[c]),
             "total": int(counts_in[c] + counts_out[c])} for c in range(nc)]
    if nc > 20:                                  # e.g. a COCO model: only show classes that were counted
        rows = [r for r in rows if r["total"]]
    csv_path = None
    if out_cfg.get("save_csv", True):
        csv_path = out_dir / f"{stem}_counts.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "in", "out", "total"])
            w.writeheader()
            w.writerows(rows)

    elapsed = time.time() - t0
    return {"counts": rows, "video": video_out, "csv": csv_path, "json": json_path if jf else None,
            "frames_processed": processed, "seconds": elapsed, "processing_fps": processed / elapsed if elapsed else 0.0,
            "line": {"start": start, "end": end}, "warning": warning, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=None, help="Overrides config 'model'")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default=None, help="Overrides <output.dir>/<video name>")
    ap.add_argument("--orientation", choices=["horizontal", "vertical"], default=None)
    ap.add_argument("--position", type=float, default=None, help="Line position, 0-1")
    ap.add_argument("--flip", action="store_true", help="Swap in/out directions")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None, help="Process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="Skip writing the annotated video (faster)")
    args = ap.parse_args()

    ov: dict = {"inference": {}, "line": {}, "output": {}}
    if args.model:
        ov["model"] = args.model
    for key, val, section in [("conf", args.conf, "inference"), ("imgsz", args.imgsz, "inference"),
                              ("vid_stride", args.stride, "inference"), ("max_frames", args.max_frames, "inference"),
                              ("orientation", args.orientation, "line"), ("position", args.position, "line")]:
        if val is not None:
            ov[section][key] = val
    if args.flip:
        ov["line"]["flip"] = True
    if args.no_video:
        ov["output"]["save_video"] = False
    cfg = load_config(args.config, ov)

    def show(done, total):
        pct = f" ({done / total:.0%})" if total else ""
        print(f"\rprocessed {done} frames{pct}", end="", flush=True)

    try:
        res = process_video(args.video, cfg["model"], cfg, args.out_dir, show)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        sys.exit(f"\nERROR: {e}")
    print(f"\nDone in {res['seconds']:.1f}s ({res['processing_fps']:.1f} frames/s on {cfg['inference']['device']}).")
    print(f"\n{'class':<12}{'in':>6}{'out':>6}{'total':>7}")
    for r in res["counts"]:
        print(f"{r['class']:<12}{r['in']:>6}{r['out']:>6}{r['total']:>7}")
    for k in ("video", "csv", "json"):
        if res[k]:
            print(f"{k:>5}: {res[k]}")
    if res["warning"]:
        print("WARNING:", res["warning"])


if __name__ == "__main__":
    main()
