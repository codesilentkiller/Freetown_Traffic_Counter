"""Check a YOLO dataset for label problems before training.

Expected layout (flat):  <root>/images/*.jpg   and   <root>/labels/*.txt

Problems checked:
  ERROR    image has no label file
  ERROR    class id outside 0..N-1
  ERROR    box with zero (or negative) width/height
  ERROR    malformed line, or box coordinates outside 0..1
  WARNING  label file exists but is empty (image with no objects)
  WARNING  label file with no matching image

Exit code is 1 if any ERROR is found (use --strict to make warnings fail too).

Example:
    python scripts/validate_labels.py --root data/freetownroad6_raw
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from common import CLASS_NAMES, IMAGE_EXTS


def validate(images_dir: Path, labels_dir: Path, nc: int):
    issues = []          # (severity, file, line_no, message)
    class_counts = Counter()
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    image_stems = {p.stem for p in images}

    for img in images:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            issues.append(("ERROR", img.name, 0, "missing label file"))
            continue
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            issues.append(("WARNING", lbl.name, 0, "empty label file (no objects)"))
            continue
        for n, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(("ERROR", lbl.name, n, f"expected 5 values, got {len(parts)}"))
                continue
            try:
                cls_f, cx, cy, w, h = map(float, parts)
            except ValueError:
                issues.append(("ERROR", lbl.name, n, "non-numeric value"))
                continue
            if cls_f != int(cls_f) or not (0 <= int(cls_f) < nc):
                issues.append(("ERROR", lbl.name, n, f"class id {parts[0]} outside 0..{nc - 1}"))
                continue
            class_counts[int(cls_f)] += 1
            if w <= 0 or h <= 0:
                issues.append(("ERROR", lbl.name, n, "zero-area box"))
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)) or cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                    or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                issues.append(("ERROR", lbl.name, n, "box outside image (coordinates not within 0..1)"))

    for lbl in sorted(labels_dir.glob("*.txt")):
        if lbl.stem not in image_stems:
            issues.append(("WARNING", lbl.name, 0, "label file has no matching image"))
    return images, issues, class_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="Folder containing images/ and labels/")
    ap.add_argument("--images", help="Images folder (if not <root>/images)")
    ap.add_argument("--labels", help="Labels folder (if not <root>/labels)")
    ap.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))
    ap.add_argument("--report", default=None, help="Optional CSV file listing every issue")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    if args.root:
        images_dir, labels_dir = Path(args.root) / "images", Path(args.root) / "labels"
    elif args.images and args.labels:
        images_dir, labels_dir = Path(args.images), Path(args.labels)
    else:
        sys.exit("Give --root, or both --images and --labels.")
    if args.images:
        images_dir = Path(args.images)
    if args.labels:
        labels_dir = Path(args.labels)
    for d in (images_dir, labels_dir):
        if not d.is_dir():
            sys.exit(f"Folder not found: {d}")

    images, issues, counts = validate(images_dir, labels_dir, args.num_classes)
    errors = [i for i in issues if i[0] == "ERROR"]
    warnings = [i for i in issues if i[0] == "WARNING"]

    for sev, f, n, msg in issues[:50]:
        print(f"{sev:7s} {f}" + (f":{n}" if n else "") + f"  {msg}")
    if len(issues) > 50:
        print(f"... and {len(issues) - 50} more (use --report to save all)")

    print(f"\nImages: {len(images)}   Errors: {len(errors)}   Warnings: {len(warnings)}")
    print("Boxes per class:")
    for c in range(args.num_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        print(f"  {c} {name:<11s} {counts.get(c, 0)}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["severity", "file", "line", "message"])
            w.writerows(issues)
        print(f"Report written to {args.report}")

    if errors or (args.strict and warnings):
        sys.exit(1)
    print("Dataset looks valid.")


if __name__ == "__main__":
    main()
