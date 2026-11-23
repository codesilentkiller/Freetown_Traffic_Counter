"""Evaluate the UNMODIFIED COCO-pretrained YOLOv8n on the same test set.

The COCO model knows 80 generic classes, not ours, so its predictions are mapped
to the nearest FreetownRoad-6 class and everything else is discarded:

    COCO car        -> car
    COCO truck      -> truck
    COCO bus        -> poda-poda      (closest generic class; an approximation)
    COCO motorcycle -> okada          (closest generic class; an approximation)
    COCO person     -> pedestrian
    (nothing)       -> keke           COCO has no tricycle class, so keke scores 0

That zero is not a bug and not a handicap: it is exactly what an off-the-shelf
model can do for Freetown traffic. Scoring uses the same code as evaluate.py.

By default the baseline runs at 640 px (its native size). Use --imgsz 416 to
match the fine-tuned model's resolution if you prefer that comparison.

Example:
    python scripts/evaluate_baseline.py --data data/freetownroad6/data.yaml
"""
import argparse

import pandas as pd
from ultralytics import YOLO

from common import (COCO_TO_FT, evaluate_detector, list_images, resolve_split_dirs,
                    save_results, yolo_prediction_stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n.pt", help="COCO weights (downloaded by Ultralytics on first use)")
    ap.add_argument("--data", default="data/freetownroad6/data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--tag", default="baseline_coco_yolov8n")
    args = ap.parse_args()

    image_dir, label_dir, names = resolve_split_dirs(args.data, args.split)
    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images in {image_dir}")
    model = YOLO(args.model)
    print(f"Evaluating COCO baseline {args.model} on {len(images)} {args.split} images ...")
    stream = yolo_prediction_stream(model, images, args.imgsz, args.conf, args.iou, args.device,
                                    args.batch, class_map=COCO_TO_FT)
    rows, cm, n = evaluate_detector(stream, label_dir, names)
    csv_path, cm_csv, png = save_results(rows, cm, names, args.out_dir, f"{args.tag}_{args.split}")

    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nSaved: {csv_path}\n       {cm_csv}\n       {png}")


if __name__ == "__main__":
    main()
