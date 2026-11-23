"""Shared helpers used by every script in this project.

If you are new to computer vision, here are the three ideas this file relies on:

* YOLO label files: one text file per image, one line per object:
      <class_id> <x_center> <y_center> <width> <height>
  All four box numbers are fractions of the image size (0 to 1).
* IoU (Intersection over Union): how much two boxes overlap, from 0 (none) to 1
  (identical). A detection counts as correct when its IoU with a real box is
  above a threshold (0.5 is the common one).
* mAP (mean Average Precision): a single score summarising how well a detector
  finds objects (precision) without missing them (recall). mAP@0.5 uses an IoU
  threshold of 0.5; mAP@0.5:0.95 averages over thresholds 0.5, 0.55 ... 0.95 and
  is stricter.

Run every script from the project root (the folder containing README.md).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

# Canonical class order. IDs in every label file and data.yaml follow this list.
CLASS_NAMES = ["car", "truck", "poda-poda", "keke", "okada", "pedestrian"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Frame filenames look like: <location>__<period>__<video>__f000123.jpg
# The location is the text before the first "__"; splitting by location relies on it.
SEP = "__"

# COCO class id -> FreetownRoad-6 class id, used ONLY for the baseline model.
# COCO: 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck. COCO has no keke class,
# so keke can never be detected by the baseline (its score will be 0, which is
# the honest result). COCO "bicycle" (1) has no counterpart and is ignored.
COCO_TO_FT = {2: 0, 7: 1, 5: 2, 3: 4, 0: 5}


# --------------------------------------------------------------------------- #
# Files, names, YAML
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    """Lower-case text with only letters, digits and single hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def make_frame_name(location: str, period: str, video_stem: str, frame_idx: int) -> str:
    """Build a frame filename stem that encodes location and time period."""
    return SEP.join([slugify(location), slugify(period), slugify(video_stem), f"f{frame_idx:06d}"])


def parse_location(filename: str) -> str:
    """Return the location encoded at the start of a frame filename."""
    return Path(filename).stem.split(SEP)[0]


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_images(folder) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def find_label_dir(image_dir: Path) -> Path:
    """Swap the last 'images' path component for 'labels' (the YOLO convention)."""
    parts = list(Path(image_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"'images' not found in path: {image_dir}")


def resolve_split_dirs(data_yaml, split: str) -> tuple[Path, Path, list[str]]:
    """From a data.yaml, return (image_dir, label_dir, class_names) for a split.

    If data.yaml has no 'path:' entry, folders are relative to the yaml file
    itself, which keeps the dataset movable between your laptop and Colab.
    """
    data_yaml = Path(data_yaml).resolve()
    cfg = load_yaml(data_yaml)
    if split not in cfg:
        raise KeyError(f"'{split}' is not defined in {data_yaml}")
    base = Path(cfg["path"]) if cfg.get("path") else Path(".")
    if not base.is_absolute():
        base = data_yaml.parent / base
    image_dir = (base / cfg[split]).resolve()
    names = cfg.get("names", CLASS_NAMES)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return image_dir, find_label_dir(image_dir), list(names)


def read_yolo_labels(path) -> np.ndarray:
    """Read a YOLO label file into an (n, 5) float array. Missing file -> (0, 5)."""
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 5), dtype=float)
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    return np.array(rows, dtype=float).reshape(-1, 5)


# --------------------------------------------------------------------------- #
# Evaluation maths (NumPy only, so the baseline and the trained model are scored
# by exactly the same code)
# --------------------------------------------------------------------------- #
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def xywhn_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalised (cx, cy, w, h) boxes to pixel (x1, y1, x2, y2)."""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    cx, cy, w, h = boxes[:, 0] * width, boxes[:, 1] * height, boxes[:, 2] * width, boxes[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n,4) and b (m,4), both x1y1x2y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


def _one_to_one(iou: np.ndarray, threshold: float):
    """Greedy one-to-one matching of ground truths (rows) and predictions (cols)."""
    g, p = np.nonzero(iou >= threshold)
    if len(g) == 0:
        return g, p
    order = np.argsort(-iou[g, p], kind="stable")
    g, p = g[order], p[order]
    _, first = np.unique(p, return_index=True)       # each prediction used once
    g, p = g[first], p[first]
    order = np.argsort(-iou[g, p], kind="stable")
    g, p = g[order], p[order]
    _, first = np.unique(g, return_index=True)       # each ground truth used once
    return g[first], p[first]


def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls) -> np.ndarray:
    """For each prediction, was it a true positive at each of the 10 IoU thresholds?

    A prediction is a true positive when it overlaps an unclaimed real box of the
    SAME class by at least the threshold. Returns a bool array (n_pred, 10).
    """
    tp = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    iou = box_iou_np(gt_boxes, pred_boxes) * (gt_cls[:, None] == pred_cls[None, :])
    for j, thr in enumerate(IOU_THRESHOLDS):
        g, p = _one_to_one(iou, thr)
        tp[p, j] = True
    return tp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve (101-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def per_class_metrics(tp, conf, pred_cls, gt_counts, names) -> list[dict]:
    """Per-class precision, recall, mAP@0.5 and mAP@0.5:0.95, plus an 'all' row.

    Precision and recall are read at the confidence threshold that maximises F1
    for that class (the same convention Ultralytics uses). Classes with no real
    boxes in the test set get NaN, and are left out of the 'all' averages.
    """
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    rows = []
    for c, name in enumerate(names):
        n_gt = int(gt_counts[c])
        if n_gt == 0:
            rows.append(dict(**{"class": name}, instances=0, precision=np.nan, recall=np.nan,
                             mAP50=np.nan, mAP50_95=np.nan))
            continue
        mask = pred_cls == c
        if mask.sum() == 0:                       # class exists but was never predicted
            rows.append(dict(**{"class": name}, instances=n_gt, precision=0.0, recall=0.0,
                             mAP50=0.0, mAP50_95=0.0))
            continue
        tpc = np.cumsum(tp[mask], axis=0)
        fpc = np.cumsum(~tp[mask], axis=0)
        rec = tpc / n_gt
        prec = tpc / (tpc + fpc)
        aps = [compute_ap(rec[:, j], prec[:, j]) for j in range(tp.shape[1])]
        f1 = 2 * prec[:, 0] * rec[:, 0] / (prec[:, 0] + rec[:, 0] + 1e-16)
        k = int(np.argmax(f1))
        rows.append(dict(**{"class": name}, instances=n_gt, precision=float(prec[k, 0]),
                         recall=float(rec[k, 0]), mAP50=aps[0], mAP50_95=float(np.mean(aps))))
    valid = [r for r in rows if r["instances"] > 0]
    rows.append({"class": "all", "instances": int(sum(r["instances"] for r in rows)),
                 "precision": float(np.mean([r["precision"] for r in valid])) if valid else np.nan,
                 "recall": float(np.mean([r["recall"] for r in valid])) if valid else np.nan,
                 "mAP50": float(np.mean([r["mAP50"] for r in valid])) if valid else np.nan,
                 "mAP50_95": float(np.mean([r["mAP50_95"] for r in valid])) if valid else np.nan})
    return rows


def update_confusion(cm, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls, conf_thr=0.25, iou_thr=0.45):
    """Add one image to the confusion matrix (rows = true class, cols = predicted).

    The last row/column is 'background': a real object nobody found (missed) or a
    prediction with no real object behind it (false alarm).
    """
    nc = cm.shape[0] - 1
    keep = pred_conf >= conf_thr
    pred_boxes, pred_cls = pred_boxes[keep], pred_cls[keep]
    iou = box_iou_np(gt_boxes, pred_boxes)
    g, p = _one_to_one(iou, iou_thr) if iou.size else (np.array([], int), np.array([], int))
    for gi, pi in zip(g, p):
        cm[gt_cls[gi], pred_cls[pi]] += 1
    for gi in set(range(len(gt_cls))) - set(g.tolist()):
        cm[gt_cls[gi], nc] += 1
    for pi in set(range(len(pred_cls))) - set(p.tolist()):
        cm[nc, pred_cls[pi]] += 1


def evaluate_detector(stream, label_dir, names):
    """Score a detector over a test set.

    `stream` yields (image_path, boxes_xyxy, confidences, class_ids, (height, width))
    for every test image, with class ids already in FreetownRoad-6 numbering.
    Returns (list of per-class metric dicts, confusion matrix, number of images).
    """
    nc = len(names)
    tps, confs, pclss = [], [], []
    gt_counts = np.zeros(nc, dtype=int)
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    n_images = 0
    for path, boxes, conf, cls, (h, w) in stream:
        n_images += 1
        labels = read_yolo_labels(Path(label_dir) / (Path(path).stem + ".txt"))
        gt_cls = labels[:, 0].astype(int)
        gt_boxes = xywhn_to_xyxy(labels[:, 1:5], w, h)
        for c in gt_cls:
            if 0 <= c < nc:
                gt_counts[c] += 1
        tps.append(match_predictions(boxes, cls, gt_boxes, gt_cls))
        confs.append(conf)
        pclss.append(cls)
        update_confusion(cm, boxes, conf, cls, gt_boxes, gt_cls)
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    pcls = np.concatenate(pclss).astype(int) if pclss else np.zeros(0, int)
    return per_class_metrics(tp, conf, pcls, gt_counts, names), cm, n_images


def save_results(rows, cm, names, out_dir, tag):
    """Write <tag>_per_class.csv, <tag>_confusion_matrix.csv and .png. Returns paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(names) + ["background"]
    csv_path = out_dir / f"{tag}_per_class.csv"
    cm_csv = out_dir / f"{tag}_confusion_matrix.csv"
    png_path = out_dir / f"{tag}_confusion_matrix.png"
    pd.DataFrame(rows).round(4).to_csv(csv_path, index=False)
    pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels]).to_csv(cm_csv)

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {tag}\n(cell = count, colour = share of true class)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return csv_path, cm_csv, png_path


def yolo_prediction_stream(model, image_paths, imgsz, conf, iou, device="cpu", batch=16, class_map=None):
    """Run a YOLO model over images, yielding what `evaluate_detector` expects.

    class_map: optional {model_class_id: freetown_class_id}. Predictions of any
    other class are dropped (used for the COCO baseline).
    """
    for i in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[i:i + batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        for path, r in zip(chunk, results):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            if class_map is not None:
                keep = np.array([c in class_map for c in cl], dtype=bool)
                xyxy, cf, cl = xyxy[keep], cf[keep], cl[keep]
                cl = np.array([class_map[c] for c in cl], dtype=int)
            yield path, xyxy, cf, cl, r.orig_shape
