# Freetown Traffic Detection and Counting System

A free, open-source computer vision system that detects, tracks and counts road users in Freetown, Sierra Leone from video. Training runs on Google Colab's free GPU; inference runs on a CPU-only laptop (16 GB RAM, no GPU needed).

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `car` | Private cars, saloons, SUVs, taxis |
| 1 | `truck` | Trucks and lorries |
| 2 | `poda-poda` | Locally operated minibus (distinct from generic "bus") |
| 3 | `keke` | Three-wheeled tricycle taxi |
| 4 | `okada` | Commercial motorcycle taxi (distinct from generic "motorcycle") |
| 5 | `pedestrian` | People on foot |

This order is fixed in `scripts/common.py` and `data.yaml`. Keep it consistent in every label file.

## How it works

1. **Detect:** a YOLOv8n model finds road users in each frame.
2. **Track:** `supervision`'s `ByteTrack` gives each object a persistent ID across frames.
3. **Count:** `supervision`'s `LineZone` counts objects whose centre crosses a configurable line, per class and per direction.
4. **Report:** an annotated H.264 video, a per-class CSV and per-frame JSON are written to `outputs/`.
5. **Explore:** a Streamlit app wraps the pipeline, with a live preview of the counting line.

## Project structure

```
freetown-traffic/
├── config.yaml                  # Inference settings (model, line, thresholds, outputs)
├── requirements.txt
├── data/
│   ├── data.sample.yaml         # Example data.yaml
│   ├── raw_videos/              # Source videos (not committed)
│   ├── frames/                  # Extracted frames
│   ├── freetownroad6_raw/       # YOLO images/ + labels/ before splitting (you create this)
│   ├── freetownroad6/           # Split dataset + data.yaml (made by split_by_location.py)
│   └── akujobi/                 # Akujobi dataset, downloaded manually
├── scripts/
│   ├── common.py                # Class list, helpers, evaluation maths (shared)
│   ├── extract_frames.py        # Video -> frames
│   ├── validate_labels.py       # Check YOLO labels
│   ├── split_by_location.py     # Train/val/test split by location + data.yaml
│   ├── extract_keke_subset.py   # Akujobi keke subset (optional pre-training)
│   ├── train.py                 # Training (Colab)
│   ├── evaluate.py              # Trained model on the test set
│   ├── evaluate_baseline.py     # Unmodified COCO YOLOv8n on the same test set
│   └── count_traffic.py         # Video -> tracked, counted, annotated output
├── app/app.py                   # Streamlit web app
├── notebooks/train_colab.ipynb  # Colab training notebook
├── models/                      # Trained .pt files (listed in the app's dropdown)
├── outputs/                     # Results (not committed)
└── docs/EVALUATION.md           # Template for baseline vs. trained results
```

Run every command from the project root. No script hardcodes absolute paths.

## Setup

**Requirements:** Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

FFmpeg is bundled through `imageio-ffmpeg`, so no separate install is needed. Once everything works, run `pip freeze > requirements.lock.txt` to record exact versions.

## Datasets

- **FreetownRoad-6 (yours):** at least 1,500 labeled images from eight Freetown locations, YOLO format. The split is 70/20/10 **by location**, so test images come from places the model never saw in training.
- **Akujobi Keke-Aware Vehicle Counting Dataset (optional):** 20,487 images, six classes (bicycle, car, motorcycle, bus, truck, keke). **Download it manually from Figshare** (the link is in the reference repo, https://github.com/abioyeayo/Keke-Aware-Vehicle-Counting) and unzip it into `data/akujobi/`. Nothing downloads it automatically.
- **Baseline:** unmodified COCO `yolov8n.pt`. Ultralytics fetches these weights on first use (the only download the code triggers), or place the file in `models/` and pass its path.

## Workflow

### 1. Prepare the data

```bash
# a) Frames: one every 15 frames; filenames encode location and period
python scripts/extract_frames.py --video data/raw_videos/wilkinson_am.mp4 --location wilkinson-road --period morning

# b) Label the frames with any YOLO-format tool (e.g. CVAT or Label Studio, both open source), then
#    arrange them as data/freetownroad6_raw/images/*.jpg and data/freetownroad6_raw/labels/*.txt
python scripts/validate_labels.py --root data/freetownroad6_raw --report outputs/label_report.csv

# c) Split by location (writes data/freetownroad6/data.yaml)
python scripts/split_by_location.py --root data/freetownroad6_raw --out data/freetownroad6

# d) Optional: keke subset of Akujobi for pre-training
python scripts/extract_keke_subset.py --src data/akujobi --out data/akujobi_keke --remap-to-freetown
```

Keep the `<location>__...` filename prefix: the split script reads the location from it. With eight locations an exact 70/20/10 image split is not possible, so the script picks the assignment closest to it (a typical result is 5 locations train, 2 val, 1 test). Use `--val-locations` / `--test-locations` to choose yourself.

### 2. Train on Colab

1. Put the project folder in Google Drive.
2. Open `notebooks/train_colab.ipynb` in Colab, set the runtime to **T4 GPU**, and run the cells. It executes:

```bash
python scripts/train.py --data data/freetownroad6/data.yaml --model yolov8n.pt \
    --imgsz 416 --batch 8 --epochs 100 --patience 20 --name ft6_coco --export-to models
```

A checkpoint is saved every epoch, the per-epoch metrics go to `outputs/metrics/<name>_training_log.csv`, and `--resume <path>/last.pt` continues an interrupted run. To use Akujobi pre-training, first train on `data/akujobi_keke/data.yaml`, then pass that run's `best.pt` as `--model` for the FreetownRoad-6 run.

3. Copy `models/ft6_coco.pt` to the `models/` folder on your laptop.

### 3. Evaluate (reproduces the comparison)

```bash
python scripts/evaluate_baseline.py --data data/freetownroad6/data.yaml
python scripts/evaluate.py --model models/ft6_coco.pt --data data/freetownroad6/data.yaml
```

Each writes, to `outputs/eval/`, a per-class CSV (precision, recall, mAP@0.5, mAP@0.5:0.95, plus an `all` row), a confusion-matrix CSV and a confusion-matrix PNG. Both scripts use the same metric code, and neither contains any placeholder numbers. Paste the results into `docs/EVALUATION.md`.

How the baseline is mapped: COCO car → car, truck → truck, bus → poda-poda, motorcycle → okada, person → pedestrian. COCO has no tricycle class, so keke scores 0. That is the genuine result for an off-the-shelf model.

### 4. Count traffic in a video (CPU)

```bash
python scripts/count_traffic.py --video clip.mp4 --model models/ft6_coco.pt
```

Outputs land in `outputs/<video name>/`: `*_annotated.mp4`, `*_counts.csv`, `*_detections.json`.

The counting line is configured in `config.yaml` (or with flags, no code edits):

```bash
python scripts/count_traffic.py --video clip.mp4 --orientation vertical --position 0.4 --flip
```

- `orientation`: `horizontal` counts traffic moving up/down the frame, `vertical` left/right.
- `position`: 0 to 1, as a fraction of frame height (horizontal) or width (vertical).
- `flip`: swaps which crossing direction is called "in" and which "out". Run a short test with `--max-frames 300` and flip if the labels are the wrong way round for your camera.

Speed tips on a slow CPU: `--stride 2` (process every 2nd frame), `--no-video` (skip annotated video).

### 5. Web app

```bash
streamlit run app/app.py
```

Pick a model, upload a video, adjust the line while watching it on the first frame, then press **Process Video**. A progress bar tracks the run. Results show the annotated video, a counts table and bar chart, and CSV/JSON/video downloads. The upload limit is set in `.streamlit/config.toml`.

## Swapping models

Drop any YOLOv8 or YOLO11 `.pt` into `models/` and pick it in the app, or pass `--model` on the command line. No code changes are needed.

## Limitations

- Daytime footage only. Night and rain are not supported.
- No license plate reading.
- Counts are estimates: occlusion, crowded scenes and ID switches in the tracker can cause missed or double counts.
- Accuracy depends on camera angle and distance; test on footage similar to FreetownRoad-6.

## Licensing

All components are free and open source. No paid APIs or cloud services are required. Ultralytics YOLO is AGPL-3.0, so check its terms before commercial redistribution.
