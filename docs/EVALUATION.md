# Evaluation: baseline vs. fine-tuned model

Fill this in after running the evaluation scripts. **Paste the real numbers from the CSV files; do not estimate.**

## Setup

| | |
|---|---|
| Test set | FreetownRoad-6 `test` split: ___ images, ___ boxes, locations: ___ (never seen in training) |
| Baseline | Unmodified COCO `yolov8n.pt`, run at ___ px, COCO classes mapped to FreetownRoad-6 (see below) |
| Trained model | `models/___.pt`, base weights: ___, run at ___ px |
| Date / hardware | ___ / CPU: ___ |

Commands used:

```bash
python scripts/evaluate_baseline.py --data data/freetownroad6/data.yaml
python scripts/evaluate.py --model models/___.pt --data data/freetownroad6/data.yaml
```

Class mapping used for the baseline: COCO car → car, truck → truck, bus → poda-poda, motorcycle → okada,
person → pedestrian. COCO has no tricycle class, so **keke is expected to score 0** for the baseline.

## Overall

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Baseline (COCO YOLOv8n) | | | | |
| Fine-tuned | | | | |

(Source: the `all` row of `outputs/eval/*_per_class.csv`.)

## Per class: mAP@0.5

| Class | Test instances | Baseline | Fine-tuned |
|---|---|---|---|
| car | | | |
| truck | | | |
| poda-poda | | | |
| keke | | | |
| okada | | | |
| pedestrian | | | |

## Per class: precision / recall (fine-tuned)

| Class | Precision | Recall | mAP@0.5:0.95 |
|---|---|---|---|
| car | | | |
| truck | | | |
| poda-poda | | | |
| keke | | | |
| okada | | | |
| pedestrian | | | |

## Confusion matrices

Baseline: `outputs/eval/baseline_coco_yolov8n_test_confusion_matrix.png`
Fine-tuned: `outputs/eval/<model>_test_confusion_matrix.png`

Rows are the true class, columns the predicted class; "background" means a missed object (last column) or a
false alarm (last row). Built at confidence 0.25 and IoU 0.45.

Main confusions observed: ___

## Inference speed (CPU)

| Model | Frames/s | Input size |
|---|---|---|
| Fine-tuned | | 416 |

## Notes and limitations

- ___
