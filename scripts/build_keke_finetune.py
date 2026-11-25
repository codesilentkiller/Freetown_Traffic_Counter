from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

CLASS_NAMES = ["car", "truck", "poda-poda", "keke", "okada", "pedestrian"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a small keke-focused fine-tuning dataset.")
    ap.add_argument("--combined", default="data/combined")
    ap.add_argument("--out", default="data/keke_finetune")
    ap.add_argument("--train-limit", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.walk_args() if False else ap.parse_args()

    src = Path(args.combined)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    rng = random.Random(args.seed)
    train_images = []
    val_images = []
    for split, destination in (("train", "train"), ("val", "val")):
        image_dir = src / "images" / split
        label_dir = src / "labels" / split
        candidates = []
        for label in label_dir.glob("*.txt"):
            lines = label.read_text().splitlines()
            if any(line.split() and int(float(line.split()[0])) == 3 for line in lines if len(line.split()) == 5):
                image = None
                for ext in IMAGE_EXTS:
                    candidate = image_dir / (label.stem + ext)
                    if candidate.exists():
                        image = candidate
                        break
                if image:
                    candidates.append((image, label))
        rng.shuffle(candidates)
        if destination == "train":
            candidates = candidates[:args.train_limit]
            train_images = candidates
        else:
            val_images = candidates

    counts = {"train": 0, "val": 0}
    for split, pairs in (("train", train_images), ("val", val_images)):
        for image, label in pairs:
            dst_image = out / "images" / split / image.name
            dst_label = out / "labels" / split / (image.stem + ".txt")
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dst_image)
            shutil.copy2(label, dst_label)
            counts[split] += 1

    local_test_dir = src / "images" / "test"
    local_test_labels = src / "labels" / "test"
    for local_image in sorted(local_test_dir.glob("keke*.jpeg")):
        local_label = local_test_labels / (local_image.stem + ".txt")
        if not local_label.exists():
            continue
        dst_image = out / "images" / "train" / ("local_" + local_image.name)
        dst_label = out / "labels" / "train" / ("local_" + local_image.stem + ".txt")
        shutil.copy2(local_image, dst_image)
        shutil.copy2(local_label, dst_label)
        counts["train"] += 1

    pseudo_dir = Path("data/rickshaw_pseudo")
    pseudo_images = sorted(pseudo_dir.glob("images/*.jpg"))
    rng.shuffle(pseudo_images)
    pseudo_val = pseudo_images[:10]
    pseudo_train = pseudo_images[10:]
    for split, pairs in (("train", pseudo_train), ("val", pseudo_val)):
        for source in pairs:
            label = pseudo_dir / "labels" / (source.stem + ".txt")
            if not label.exists():
                continue
            dst_image = out / "images" / split / ("openverse_" + source.name)
            dst_label = out / "labels" / split / ("openverse_" + source.stem + ".txt")
            shutil.copy2(source, dst_image)
            shutil.copy2(label, dst_label)
            counts[split] += 1

    data = {"train": "images/train", "val": "images/val", "nc": 6,
            "names": {i: name for i, name in enumerate(CLASS_NAMES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print("Keke fine-tune dataset:", counts)
    print("Wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
