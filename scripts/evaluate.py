"""Evaluate a trained model on the held-out test set.

Reports per-class precision, recall, mAP@0.5, mAP@0.5:0.95, overall mAP, and a
confusion matrix, saved as CSV and PNG. The metric code lives in common.py and is
shared with evaluate_baseline.py, so both models are scored identically.

Runs on CPU by default.

Example:
    python scripts/evaluate.py --model models/best.pt --data data/freetownroad6/data.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from common import (CLASS_NAMES, evaluate_detector, list_images, resolve_split_dirs,
                    save_results, yolo_prediction_stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Trained .pt file")
    ap.add_argument("--data", default="data/freetownroad6/data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=416, help="Use the size the model was trained at")
    ap.add_argument("--conf", type=float, default=0.001, help="Very low on purpose: mAP needs the full ranking")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS overlap threshold")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--tag", default=None, help="Output file prefix (default: model file name)")
    args = ap.parse_args()

    image_dir, label_dir, names = resolve_split_dirs(args.data, args.split)
    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images in {image_dir}")
    model = YOLO(args.model)
    model_names = [model.names[k] for k in sorted(model.names)]
    if model_names != names:
        print(f"WARNING: model classes {model_names} differ from dataset classes {names}. "
              "Scores will be meaningless unless they match (use evaluate_baseline.py for COCO models).")

    print(f"Evaluating {args.model} on {len(images)} {args.split} images ...")
    stream = yolo_prediction_stream(model, images, args.imgsz, args.conf, args.iou, args.device, args.batch)
    rows, cm, n = evaluate_detector(stream, label_dir, names)
    tag = args.tag or f"{Path(args.model).stem}_{args.split}"
    csv_path, cm_csv, png = save_results(rows, cm, names, args.out_dir, tag)

    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nSaved: {csv_path}\n       {cm_csv}\n       {png}")


if __name__ == "__main__":
    main()
