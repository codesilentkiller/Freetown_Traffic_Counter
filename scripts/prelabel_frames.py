"""Generate YOLO prelabels for Freetown frames using stock COCO weights.

The output is intentionally in the FreetownRoad-6 class layout. Keke has no
COCO source class and receives no labels. Review every generated box manually.
"""
import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import CLASS_NAMES

COCO_TO_FREETOWN = {
    2: 0,   # car
    7: 1,   # truck
    5: 2,   # bus -> poda-poda approximation
    3: 4,   # motorcycle -> okada approximation
    0: 5,   # person -> pedestrian
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--out", default="data/freetownroad6_raw")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    frame_dir = Path(args.frames)
    out_dir = Path(args.out)
    image_dir = out_dir / "images"
    label_dir = out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(p for p in frame_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise SystemExit(f"No frames found under {frame_dir}")

    model = YOLO(args.model)
    coco_names = {int(k): v for k, v in model.names.items()}
    copied = generated = 0

    for frame in frames:
        dst = image_dir / frame.name
        shutil.copy2(frame, dst)
        result = model.predict(str(frame), conf=args.conf, verbose=False)[0]
        lines = []
        if result.boxes is not None:
            h, w = result.orig_shape
            for cls_id, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
                coco_id = int(cls_id)
                if coco_id not in COCO_TO_FREETOWN:
                    continue
                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                ft_id = COCO_TO_FREETOWN[coco_id]
                lines.append(f"{ft_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (label_dir / f"{frame.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        generated += len(lines)
        copied += 1

    print(f"Prelabeled {copied} frames with {generated} boxes")
    print("Keke: no stock-COCO predictions; add those boxes manually")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")


if __name__ == "__main__":
    main()
