"""Train (or fine-tune) a YOLO model on FreetownRoad-6. Meant for Google Colab's free GPU.

What "training" means here: the model sees the labeled training images many times
(one pass = one epoch), nudging its weights each time. After every epoch it is
scored on the validation set. If the validation score stops improving for
`--patience` epochs, training stops early (it has stopped learning, and more
epochs would only memorise the training images).

Outputs (under --project/--name):
    weights/best.pt      best checkpoint by validation score  <- use this one
    weights/last.pt      most recent checkpoint
    weights/epochN.pt    a checkpoint EVERY epoch (save_period=1)
    results.csv          metrics for every epoch (also copied to outputs/metrics/)

Examples:
    # baseline-style fine-tune from COCO weights
    python scripts/train.py --data data/freetownroad6/data.yaml --model yolov8n.pt --name ft6_coco
    # fine-tune from a checkpoint pre-trained on the Akujobi keke subset
    python scripts/train.py --data data/freetownroad6/data.yaml --model runs/akujobi/weights/best.pt --name ft6_akujobi
    # resume an interrupted run (Colab disconnects happen)
    python scripts/train.py --resume outputs/runs/ft6_coco/weights/last.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/freetownroad6/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Base weights: yolov8n.pt, yolo11n.pt or your own .pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    ap.add_argument("--device", default="", help="'' = automatic (GPU if present), or 0 / cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="outputs/runs")
    ap.add_argument("--name", default="freetownroad6")
    ap.add_argument("--exist-ok", action="store_true", help="Reuse the run folder if it exists")
    ap.add_argument("--resume", default=None, help="Path to a last.pt to resume")
    ap.add_argument("--metrics-dir", default="outputs/metrics", help="Where the CSV log is copied")
    ap.add_argument("--export-to", default=None, help="Folder to copy best.pt into, e.g. models/")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(Path(args.data).resolve()),
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=args.epochs,
            patience=args.patience,
            save_period=1,                         # checkpoint every epoch
            project=str(Path(args.project).resolve()),   # absolute: avoids Ultralytics re-routing relative paths
            name=args.name,
            exist_ok=args.exist_ok,
            seed=args.seed,
            workers=args.workers,
            device=args.device or None,
            plots=True,
        )

    save_dir = Path(model.trainer.save_dir)
    log = save_dir / "results.csv"                # Ultralytics writes one row per epoch
    if log.exists():
        dst = Path(args.metrics_dir) / f"{save_dir.name}_training_log.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dst)
        print(f"Training log: {dst}")
    best = save_dir / "weights" / "best.pt"
    print(f"Best weights: {best}")
    if args.export_to and best.exists():
        out = Path(args.export_to)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out / f"{save_dir.name}.pt")
        print(f"Copied to {out / (save_dir.name + '.pt')}")


if __name__ == "__main__":
    main()
