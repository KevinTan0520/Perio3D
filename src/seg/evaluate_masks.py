from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.seg.io_utils import SEG_CLASSES, read_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase C segmentation masks.")
    parser.add_argument("--pred-manifest", type=Path, default=Path("outputs/seg/pred_manifest.csv"))
    parser.add_argument("--annotation-queue", type=Path, default=Path("outputs/seg/annotation_queue.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/seg/metrics.json"))
    return parser.parse_args()


def load_binary_mask(path: Path, shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (shape[1], shape[0]):
            img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        return np.asarray(img) > 0


def read_pred_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))


def dice_iou(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred_bool = pred.astype(bool)
    truth_bool = truth.astype(bool)
    intersection = int(np.logical_and(pred_bool, truth_bool).sum())
    pred_sum = int(pred_bool.sum())
    truth_sum = int(truth_bool.sum())
    union = int(np.logical_or(pred_bool, truth_bool).sum())
    dice = 1.0 if pred_sum + truth_sum == 0 else (2.0 * intersection) / (pred_sum + truth_sum)
    iou = 1.0 if union == 0 else intersection / union
    return {"dice": round(float(dice), 6), "iou": round(float(iou), 6)}


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(np.mean(values)), 6)


def main() -> None:
    args = parse_args()
    pred_rows = [row for row in read_csv_rows(args.pred_manifest) if row.get("status") == "ok"]
    queue_rows = read_csv_rows(args.annotation_queue) if args.annotation_queue.exists() else []
    queue_by_image = {row["image_path"]: row for row in queue_rows}

    class_scores = {
        "tooth": {"dice": [], "iou": []},
        "gingiva": {"dice": [], "iou": []},
    }
    evaluated_images = 0
    missing_gold = 0
    coverage = {"tooth": [], "gingiva": [], "semantic": []}
    confidence_values: List[float] = []

    for row in pred_rows:
        pred_mask = read_pred_mask(Path(row["mask_path"]))
        coverage["tooth"].append(float(row["tooth_coverage"]))
        coverage["gingiva"].append(float(row["gingiva_coverage"]))
        coverage["semantic"].append(float(row["semantic_coverage"]))
        confidence_values.append(float(row["confidence"]))

        queue_row = queue_by_image.get(row["image_path"])
        if not queue_row:
            missing_gold += 1
            continue

        truth_paths = {
            "tooth": Path(queue_row["tooth_mask_path"]),
            "gingiva": Path(queue_row["gingiva_mask_path"]),
        }
        image_evaluated = False
        for class_name, truth_path in truth_paths.items():
            truth = load_binary_mask(truth_path, pred_mask.shape)
            if truth is None:
                continue
            pred = pred_mask == SEG_CLASSES[class_name]
            scores = dice_iou(pred, truth)
            class_scores[class_name]["dice"].append(scores["dice"])
            class_scores[class_name]["iou"].append(scores["iou"])
            image_evaluated = True

        if image_evaluated:
            evaluated_images += 1
        else:
            missing_gold += 1

    classes = {}
    for class_name, scores in class_scores.items():
        classes[class_name] = {
            "dice": mean_or_none(scores["dice"]),
            "iou": mean_or_none(scores["iou"]),
            "num_gold_masks": len(scores["dice"]),
        }

    metrics = {
        "num_predictions": len(pred_rows),
        "num_evaluated_images": evaluated_images,
        "num_missing_gold_images": missing_gold,
        "classes": classes,
        "coverage": {
            "mean_tooth": mean_or_none(coverage["tooth"]),
            "mean_gingiva": mean_or_none(coverage["gingiva"]),
            "mean_semantic": mean_or_none(coverage["semantic"]),
        },
        "confidence": {
            "mean": mean_or_none(confidence_values),
            "min": round(float(np.min(confidence_values)), 6) if confidence_values else None,
            "max": round(float(np.max(confidence_values)), 6) if confidence_values else None,
        },
        "notes": (
            "Dice/IoU are null until binary gold masks are exported to the paths in "
            "outputs/seg/annotation_queue.csv."
        )
        if evaluated_images == 0
        else "",
    }
    write_json(args.metrics_out, metrics)
    print(f"[seg-eval] metrics written to: {args.metrics_out}")


if __name__ == "__main__":
    main()

