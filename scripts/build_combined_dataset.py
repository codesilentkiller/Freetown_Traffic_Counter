from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASS_NAMES = ["car", "truck", "poda-poda", "keke", "okada", "pedestrian"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(label: Path) -> Path | None:
    parts = list(label.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "labels":
            image_dir = Path(*parts[:i]) / "images"
            for ext in IMAGE_EXTS:
                candidate = image_dir / (label.stem + ext)
                if candidate.exists():
                    return candidate
                candidate = image_dir / (label.stem + ext.upper())
                if candidate.exists():
                    return candidate
    return None


def copy_pair(src_img: Path, src_label: Path, dst_img: Path, dst_label: Path, mapping=None) -> tuple:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    counts = Counter()
    for raw in src_label.read_text().splitlines():
        parts = raw.split()
        if len(parts) != 5:
            continue
        try:
            source_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        target_id = mapping.get(source_id) if mapping else source_id
        if target_id is None:
            continue
        if not all(0 <= value <= 1 for value in values):
            continue
        cx, cy, w, h = values
        if w <= 0 or h <= 0 or cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            continue
        lines.append(f"{target_id} " + " ".join(parts[1:]))
        counts[CLASS_NAMES[target_id]] += 1
    shutil.copy2(src_img, dst_img)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))
    return bool(lines), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge remapped Akujobi and FreetownRoad-6 datasets.")
    parser.add_argument("--akujobi", default="data/akujobi/datasets")
    parser.add_argument("--freetown", default="data/freetownroad6_raw")
    parser.add_argument("--out", default="data/combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ft-train", type=float, default=0.60)
    parser.add_argument("--ft-val", type=float, default=0.20)
    parser.add_argument("--aku-train-limit", type=int, default=6000)
    parser.add_argument("--aku-val-limit", type=int, default=800)
    args = parser.parse_args()

    src = Path(args.akujobi)
    ft = Path(args.freetown)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    counts = Counter()
    images = {"train": 0, "val": 0, "test": 0}

    # Akujobi IDs: 0 bicycle, 1 bus, 2 car, 3 keke, 4 motorcycle, 5 truck.
    aku_mapping = {1: 2, 2: 0, 3: 3, 4: 4, 5: 1}
    rng = random.Random(args.seed)
    for source_split, target_split in (("train", "train"), ("valid", "val")):
        labels = src / source_split / "labels"
        eligible = sorted(labels.rglob("*.txt"))
        rng.shuffle(eligible)
        limit = args.aku_train_limit if target_split == "train" else args.aku_val_limit
        seen = 0
        for label in eligible:
            image = find_image(label)
            if image is None:
                continue
            stem = f"aku_{label.stem}_{target_split}"
            dst_img = out / "images" / target_split / (stem + image.suffix.lower())
            dst_label = out / "labels" / target_split / (stem + ".txt")
            kept, class_counts = copy_pair(image, label, dst_img, dst_label, aku_mapping)
            if kept:
                images[target_split] += 1
                counts.update(class_counts)
                seen += 1
            if seen >= limit:
                break

    ft_images = sorted(p for p in (ft / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    ft_images = [p for p in ft_images if (ft / "labels" / (p.stem + ".txt")).exists()]
    rng = random.Random(args.seed)
    rng.shuffle(ft_images)
    n_train = int(len(ft_images) * args.ft_train)
    n_val = int(len(ft_images) * args.ft_val)
    ft_splits = {p: "train" for p in ft_images[:n_train]}
    ft_splits.update({p: "val" for p in ft_images[n_train:n_train + n_val]})
    ft_splits.update({p: "test" for p in ft_images[n_train + n_val:]})

    for image, target_split in ft_splits.items():
        source_label = ft / "labels" / (image.stem + ".txt")
        stem = f"ft6_{image.stem}_{target_split}"
        dst_img = out / "images" / target_split / (stem + image.suffix.lower())
        dst_label = out / "labels" / target_split / (stem + ".txt")
        kept, class_counts = copy_pair(image, source_label, dst_img, dst_label)
        if kept:
            images[target_split] += 1
            counts.update(class_counts)

    data = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": len(CLASS_NAMES), "names": {i: name for i, name in enumerate(CLASS_NAMES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    print("Combined dataset written:", out)
    print("Images:", " ".join(f"{split}={count}" for split, count in images.items()))
    print("Boxes:")
    for name in CLASS_NAMES:
        print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
