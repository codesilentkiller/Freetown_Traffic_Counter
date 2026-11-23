"""Extract every Nth frame from traffic videos.

Frames are saved as JPEGs named  <location>__<period>__<video>__f<frame>.jpg  so
later steps can tell where and when each frame was filmed (the dataset is split
by location, so this matters).

Example:
    python scripts/extract_frames.py --video data/raw_videos/wilkinson.mp4 \
        --location wilkinson-road --period morning --every 15
"""
import argparse
import sys
from pathlib import Path

import cv2

from common import VIDEO_EXTS, make_frame_name, slugify


def collect_videos(paths):
    videos = []
    for p in map(Path, paths):
        if p.is_dir():
            videos += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            videos.append(p)
        else:
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
    return videos


def extract(video: Path, out_dir: Path, location: str, period: str, every: int, quality: int, max_frames=None) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video}", file=sys.stderr)
        return 0
    saved, idx = 0, 0
    while True:
        # grab() advances one frame without the cost of decoding it; only the
        # frames we keep are decoded with retrieve().
        if not cap.grab():
            break
        if idx % every == 0:
            ok, frame = cap.retrieve()
            if ok:
                name = make_frame_name(location, period, video.stem, idx) + ".jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved += 1
                if max_frames and saved >= max_frames:
                    break
        idx += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, help="Video file(s) or folder(s) of videos")
    ap.add_argument("--location", required=True, help="Location name, e.g. wilkinson-road")
    ap.add_argument("--period", required=True, help="Time period, e.g. morning, midday, afternoon")
    ap.add_argument("--every", type=int, default=15, help="Keep one frame every N frames (default 15)")
    ap.add_argument("--out", default="data/frames", help="Output root (a sub-folder per location is made)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N frames per video")
    args = ap.parse_args()

    out_dir = Path(args.out) / slugify(args.location)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(args.video)
    if not videos:
        sys.exit("No videos found.")
    total = 0
    for v in videos:
        n = extract(v, out_dir, args.location, args.period, args.every, args.quality, args.max_frames)
        print(f"{v.name}: {n} frames -> {out_dir}")
        total += n
    print(f"Done: {total} frames.")


if __name__ == "__main__":
    main()
