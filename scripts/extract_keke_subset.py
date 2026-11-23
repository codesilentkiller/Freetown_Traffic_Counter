"""Pull the keke-containing images out of the Akujobi dataset (optional pre-training).

Download the dataset yourself from Figshare (see README) and unzip it into
data/akujobi/. This script never downloads anything.

It looks for YOLO label files anywhere under --src (any 'labels' folder, with
the matching 'images' folder beside it), keeps images with at least one keke
box, and writes a new train/val dataset with its own data.yaml.

Class names are read from the dataset's own yaml if present; otherwise pass
--names (in class-id order), e.g.
    --names bicycle car motorcycle bus truck keke

--remap-to-freetown converts labels to FreetownRoad-6 ids so the pre-trained
model has the same six outputs: car->car, truck->truck, bus->poda-poda,
motorcycle->okada, keke->keke, bicycle dropped. This is an approximation (a bus
is not always a poda-poda, a motorcycle is not always an okada); fine-tuning on
FreetownRoad-6 corrects it.

Example:
    python scripts/extract_keke_subset.py --src data/akujobi --out data/akujobi_keke --remap-to-freetown
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

from common import CLASS_NAMES, IMAGE_EXTS, load_yaml

AKUJOBI_TO_FT = {"car": "car", "truck": "truck", "bus": "poda-poda", "motorcycle": "okada", "keke": "keke"}


def find_names(src: Path, cli_names):
    if cli_names:
        return list(cli_names)
    for yml in sorted(src.rglob("*.yaml")) + sorted(src.rglob("*.yml")):
        names = load_yaml(yml).get("names")
        if names:
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    sys.exit("Could not find class names. Pass --names bicycle car motorcycle bus truck keke (in id order).")


def image_for(label: Path):
    """Find the image matching a label file by swapping 'labels' -> 'images' in its path."""
    parts = list(label.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "labels":
            parts[i] = "images"
            base = Path(*parts).with_suffix("")
            for ext in IMAGE_EXTS | {e.upper() for e in IMAGE_EXTS}:
                cand = base.with_suffix(ext)
                if cand.exists():
                    return cand
            return None
    return None


def split_of(label: Path) -> str:
    for p in label.parts:
        if p.lower() in ("train", "val", "valid", "test"):
            return "val" if p.lower() == "valid" else p.lower()
    return "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/akujobi", help="Unzipped Akujobi dataset folder")
    ap.add_argument("--out", default="data/akujobi_keke")
    ap.add_argument("--names", nargs="*", default=None, help="Class names in id order (if no yaml found)")
    ap.add_argument("--keke-name", default="keke")
    ap.add_argument("--remap-to-freetown", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Used only if the dataset has no val split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir() or not any(src.iterdir()):
        sys.exit(f"{src} is empty or missing. Download the Akujobi dataset from Figshare first (see README).")
    names = find_names(src, args.names)
    lower = [n.lower() for n in names]
    if args.keke_name.lower() not in lower:
        sys.exit(f"'{args.keke_name}' not among class names: {names}")
    keke_id = lower.index(args.keke_name.lower())

    id_map = None
    if args.remap_to_freetown:
        id_map = {}
        for i, n in enumerate(lower):
            if n in AKUJOBI_TO_FT:
                id_map[i] = CLASS_NAMES.index(AKUJOBI_TO_FT[n])

    kept = []   # (image, label_lines, split)
    for lbl in sorted(src.rglob("*.txt")):
        if "labels" not in lbl.parts:
            continue
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not any(len(p) == 5 and p[0].isdigit() and int(p[0]) == keke_id for p in lines):
            continue
        img = image_for(lbl)
        if img is None:
            continue
        if id_map is not None:
            lines = [[str(id_map[int(p[0])])] + p[1:] for p in lines if len(p) == 5 and int(p[0]) in id_map]
        kept.append((img, lines, split_of(lbl)))

    if not kept:
        sys.exit("No keke images found. Check --src, --names and --keke-name.")

    if not any(s == "val" for _, _, s in kept):
        rng = random.Random(args.seed)
        idx = list(range(len(kept)))
        rng.shuffle(idx)
        for i in idx[: max(1, int(len(kept) * args.val_fraction))]:
            kept[i] = (kept[i][0], kept[i][1], "val")

    for img, lines, sp in kept:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / sp / img.name)
        (out / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(" ".join(p) for p in lines) + "\n")

    out_names = CLASS_NAMES if id_map is not None else names
    data = {"train": "images/train", "val": "images/val",
            "nc": len(out_names), "names": {i: n for i, n in enumerate(out_names)}}
    if any(s == "test" for _, _, s in kept):
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Kept {len(kept)} keke-containing images -> {out}")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
