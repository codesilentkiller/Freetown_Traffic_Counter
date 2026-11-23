"""Split a labeled dataset into train / val / test BY LOCATION and write data.yaml.

Why not split randomly? Frames from the same camera spot look almost identical.
A random split leaks near-duplicates into the test set and inflates scores. Here
every location lands in exactly one split, so the test set is made of places the
model never saw in training.

Input (flat):   <root>/images/*.jpg  and  <root>/labels/*.txt
                filenames must start with "<location>__" (extract_frames.py does this)
Output:         <out>/images/{train,val,test}, <out>/labels/{train,val,test},
                <out>/data.yaml, <out>/split_manifest.csv

With few locations an exact 70/20/10 image split is impossible, so the script
tries every assignment of locations to splits and picks the one closest to the
target. Use --val-locations / --test-locations to choose yourself.

Example:
    python scripts/split_by_location.py --root data/freetownroad6_raw --out data/freetownroad6
"""
import argparse
import csv
import itertools
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from common import CLASS_NAMES, IMAGE_EXTS, parse_location

SPLITS = ["train", "val", "test"]


def choose_assignment(counts: dict, targets, seed: int) -> dict:
    """Assign each location to a split, minimising distance from the target fractions."""
    locs = sorted(counts)
    total = sum(counts.values())
    rng = random.Random(seed)
    if len(locs) <= 10:                               # try every assignment (3^10 = 59k)
        combos = list(itertools.product(range(3), repeat=len(locs)))
        rng.shuffle(combos)                           # seed decides ties
        best, best_score = None, None
        for combo in combos:
            if len(set(combo)) < 3:
                continue                              # every split needs a location
            sums = [0, 0, 0]
            for loc, s in zip(locs, combo):
                sums[s] += counts[loc]
            score = sum(abs(sums[i] / total - targets[i]) for i in range(3))
            if best_score is None or score < best_score:
                best, best_score = combo, score
        return dict(zip(locs, [SPLITS[s] for s in best]))
    rng.shuffle(locs)                                 # many locations: simple greedy fill
    out, filled = {}, [0, 0, 0]
    for loc in locs:
        i = max(range(3), key=lambda k: targets[k] - filled[k] / total)
        out[loc] = SPLITS[i]
        filled[i] += counts[loc]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder with images/ and labels/")
    ap.add_argument("--out", required=True, help="Output dataset folder")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-locations", nargs="*", default=None)
    ap.add_argument("--test-locations", nargs="*", default=None)
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copying files")
    ap.add_argument("--overwrite", action="store_true", help="Delete --out first if it exists")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    images = sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {root / 'images'}")

    by_loc = defaultdict(list)
    skipped = 0
    for img in images:
        if (root / "labels" / (img.stem + ".txt")).exists():
            by_loc[parse_location(img.name)].append(img)
        else:
            skipped += 1
    if skipped:
        print(f"WARNING: {skipped} images skipped (no label file). Run validate_labels.py.")
    counts = {loc: len(v) for loc, v in by_loc.items()}
    if len(counts) < 3:
        sys.exit(f"Need at least 3 locations to split by location, found {len(counts)}: {sorted(counts)}")

    s = sum(args.ratios)
    targets = [r / s for r in args.ratios]
    if args.val_locations is not None or args.test_locations is not None:
        val_l, test_l = set(args.val_locations or []), set(args.test_locations or [])
        unknown = (val_l | test_l) - set(counts)
        if unknown:
            sys.exit(f"Unknown locations: {sorted(unknown)}. Available: {sorted(counts)}")
        if val_l & test_l:
            sys.exit("A location cannot be in both val and test.")
        assignment = {loc: "test" if loc in test_l else "val" if loc in val_l else "train" for loc in counts}
    else:
        assignment = choose_assignment(counts, targets, args.seed)

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty. Use --overwrite to replace it.")
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    def put(src: Path, dst: Path):
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    for loc, files in by_loc.items():
        sp = assignment[loc]
        for img in files:
            put(img, out / "images" / sp / img.name)
            put(root / "labels" / (img.stem + ".txt"), out / "labels" / sp / (img.stem + ".txt"))

    # data.yaml has no "path:" entry on purpose: Ultralytics then treats the
    # folders as relative to the yaml file, so the dataset works on any machine.
    data = {"train": "images/train", "val": "images/val", "test": "images/test",
            "nc": len(CLASS_NAMES), "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location", "split", "images"])
        for loc in sorted(counts):
            w.writerow([loc, assignment[loc], counts[loc]])

    total = sum(counts.values())
    print(f"\n{'split':<6} {'images':>7} {'share':>7}  locations")
    for sp in SPLITS:
        locs = sorted(l for l in counts if assignment[l] == sp)
        n = sum(counts[l] for l in locs)
        print(f"{sp:<6} {n:>7} {n / total:>6.1%}  {', '.join(locs)}")
    print(f"\nWrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
