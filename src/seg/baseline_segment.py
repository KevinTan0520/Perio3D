from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

from src.seg.io_utils import SEG_CLASSES, read_csv_rows, relative_mask_path, write_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tooth/gingiva baseline masks for phase C.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/seg/masks"))
    parser.add_argument("--manifest-out", type=Path, default=Path("outputs/seg/pred_manifest.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/seg/pred_summary.json"))
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for smoke tests. 0 means all images.")
    parser.add_argument("--resize-long-edge", type=int, default=1600)
    return parser.parse_args()


def load_split_rows(split_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_path = split_dir / f"{split}.csv"
        if not split_path.exists():
            continue
        for row in read_csv_rows(split_path):
            row["split"] = split
            rows.append(row)
    return rows


def read_image_rgb(path: Path, resize_long_edge: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        original_size = (width, height)
        scale = 1.0
        long_edge = max(width, height)
        if resize_long_edge > 0 and long_edge > resize_long_edge:
            scale = resize_long_edge / float(long_edge)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(img), scale, original_size


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    opened = mask_img.filter(ImageFilter.MinFilter(kernel_size)).filter(ImageFilter.MaxFilter(kernel_size))
    closed = opened.filter(ImageFilter.MaxFilter(kernel_size)).filter(ImageFilter.MinFilter(kernel_size))
    return np.asarray(closed) > 0


def segment_image(image_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    hsv = np.asarray(Image.fromarray(image_rgb, mode="RGB").convert("HSV"))
    h = hsv[:, :, 0].astype(np.float32) / 255.0 * 360.0
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    tissue = (s > 20) & (v > 35) & (luminance > 35)
    tooth = tissue & (s < 95) & (v > 115) & (luminance > 135)
    red_hue = (h < 28) | (h > 330)
    gingiva = tissue & red_hue & (s > 35) & (v > 55) & (r > g * 0.92) & ~tooth

    tooth = clean_mask(tooth, kernel_size=5)
    gingiva = clean_mask(gingiva, kernel_size=9)
    gingiva = gingiva & ~tooth

    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    mask[gingiva] = SEG_CLASSES["gingiva"]
    mask[tooth] = SEG_CLASSES["tooth"]

    total_pixels = float(mask.size)
    tooth_coverage = float(np.count_nonzero(tooth) / total_pixels)
    gingiva_coverage = float(np.count_nonzero(gingiva) / total_pixels)
    semantic_coverage = float(np.count_nonzero(mask) / total_pixels)
    confidence = coverage_confidence(tooth_coverage, gingiva_coverage)
    return mask, {
        "tooth_coverage": tooth_coverage,
        "gingiva_coverage": gingiva_coverage,
        "semantic_coverage": semantic_coverage,
        "confidence": confidence,
    }


def coverage_confidence(tooth_coverage: float, gingiva_coverage: float) -> float:
    tooth_score = 1.0 - min(1.0, abs(tooth_coverage - 0.18) / 0.18)
    gingiva_score = 1.0 - min(1.0, abs(gingiva_coverage - 0.22) / 0.22)
    coverage_score = min(1.0, (tooth_coverage + gingiva_coverage) / 0.28)
    return round(max(0.0, 0.4 * tooth_score + 0.4 * gingiva_score + 0.2 * coverage_score), 4)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def restore_original_size(mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    if mask.shape[1] == original_size[0] and mask.shape[0] == original_size[1]:
        return mask
    resized = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    return np.asarray(resized)


def append_manifest_row(path: Path, row: Dict[str, object], fieldnames: List[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = load_split_rows(args.split_dir)
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise ValueError(f"no split rows found in {args.split_dir}")

    fieldnames = [
        "split",
        "case_uid",
        "group_id",
        "case_id",
        "view_id",
        "image_path",
        "mask_path",
        "model_name",
        "confidence",
        "tooth_coverage",
        "gingiva_coverage",
        "semantic_coverage",
        "resize_scale",
        "status",
        "error",
    ]
    if args.manifest_out.exists():
        args.manifest_out.unlink()

    manifest_rows: List[Dict[str, object]] = []
    processed = 0
    failed = 0
    confidence_values: List[float] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        mask_path = relative_mask_path(args.mask_dir, row)
        out_row: Dict[str, object] = {
            "split": row["split"],
            "case_uid": row["case_uid"],
            "group_id": row.get("group_id", ""),
            "case_id": row.get("case_id", ""),
            "view_id": row.get("view_id", ""),
            "image_path": row["image_path"],
            "mask_path": mask_path.as_posix(),
            "model_name": "pillow_numpy_color_baseline_v1",
            "confidence": "",
            "tooth_coverage": "",
            "gingiva_coverage": "",
            "semantic_coverage": "",
            "resize_scale": "",
            "status": "failed",
            "error": "",
        }
        try:
            image_rgb, scale, original_size = read_image_rgb(image_path, args.resize_long_edge)
            mask, stats = segment_image(image_rgb)
            mask = restore_original_size(mask, original_size)
            save_mask(mask, mask_path)
            out_row.update(
                {
                    "confidence": stats["confidence"],
                    "tooth_coverage": round(stats["tooth_coverage"], 6),
                    "gingiva_coverage": round(stats["gingiva_coverage"], 6),
                    "semantic_coverage": round(stats["semantic_coverage"], 6),
                    "resize_scale": round(scale, 6),
                    "status": "ok",
                }
            )
            processed += 1
            confidence_values.append(float(stats["confidence"]))
        except Exception as exc:  # noqa: BLE001 - batch pipeline should record and continue.
            out_row["error"] = str(exc)
            failed += 1

        manifest_rows.append(out_row)
        append_manifest_row(args.manifest_out, out_row, fieldnames, write_header=index == 1)

    ok_rows = [row for row in manifest_rows if row["status"] == "ok"]
    summary = {
        "model_name": "pillow_numpy_color_baseline_v1",
        "num_requested": len(rows),
        "num_processed": processed,
        "num_failed": failed,
        "mask_dir": args.mask_dir.as_posix(),
        "manifest": args.manifest_out.as_posix(),
        "mean_confidence": round(float(np.mean(confidence_values)), 4) if confidence_values else None,
        "mean_tooth_coverage": round(float(np.mean([float(row["tooth_coverage"]) for row in ok_rows])), 6)
        if ok_rows
        else None,
        "mean_gingiva_coverage": round(float(np.mean([float(row["gingiva_coverage"]) for row in ok_rows])), 6)
        if ok_rows
        else None,
    }
    write_json(args.summary_out, summary)
    write_csv_rows(args.manifest_out, manifest_rows, fieldnames)

    print(f"[seg-baseline] masks written under: {args.mask_dir}")
    print(f"[seg-baseline] manifest written to: {args.manifest_out}")
    print(f"[seg-baseline] summary written to: {args.summary_out}")


if __name__ == "__main__":
    main()
