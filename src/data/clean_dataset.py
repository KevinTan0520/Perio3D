from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import shutil

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and normalize dataset image files.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/processed/clean_manifest.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/reports/clean_summary.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-output", action="store_true", help="Remove existing copied images before cleaning.")
    return parser.parse_args()


def is_mac_metadata(path: Path) -> bool:
    return path.name == ".DS_Store" or path.name.startswith("._")


def parse_case_ids(dataset_root: Path, file_path: Path) -> Tuple[str, str]:
    rel = file_path.relative_to(dataset_root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def infer_view_id(file_path: Path) -> str:
    stem = file_path.stem
    parts = stem.split("-")
    if parts and parts[-1].isdigit():
        return f"P-{int(parts[-1]):02d}"
    return "unknown"


def verify_image(path: Path) -> Tuple[bool, int, int, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return True, width, height, ""
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        return False, 0, 0, str(exc)


def write_manifest(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root
    manifest_out = args.manifest_out
    summary_out = args.summary_out

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if args.reset_output and not args.dry_run and output_root.exists():
        shutil.rmtree(output_root)

    seq_counter = defaultdict(int)
    rows: List[Dict[str, object]] = []
    summary = {
        "total_files": 0,
        "ignored_mac_metadata": 0,
        "ignored_non_image": 0,
        "copied_images": 0,
        "broken_images": 0,
    }

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        summary["total_files"] += 1

        if is_mac_metadata(file_path):
            summary["ignored_mac_metadata"] += 1
            continue

        ext = file_path.suffix.lower()
        if ext not in IMAGE_EXTS:
            summary["ignored_non_image"] += 1
            continue

        group_id, case_id = parse_case_ids(dataset_root, file_path)
        key = f"{group_id}/{case_id}"
        seq_counter[key] += 1
        seq = seq_counter[key]

        is_valid, width, height, error = verify_image(file_path)
        case_uid = case_id
        view_id = infer_view_id(file_path)
        new_name = f"{case_uid}_{view_id}_{seq:03d}{ext}"
        out_dir = output_root / group_id / case_id
        dst_path = out_dir / new_name

        status = "copied"
        if not is_valid:
            status = "broken"
            summary["broken_images"] += 1
        else:
            summary["copied_images"] += 1
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dst_path)

        rows.append(
            {
                "source_path": str(file_path.as_posix()),
                "dest_path": str(dst_path.as_posix()) if status == "copied" else "",
                "case_uid": case_uid,
                "group_id": group_id,
                "case_id": case_id,
                "view_id": view_id,
                "sequence_id": seq,
                "new_name": new_name,
                "width": width,
                "height": height,
                "status": status,
                "error": error,
            }
        )

    fieldnames = [
        "source_path",
        "dest_path",
        "case_uid",
        "group_id",
        "case_id",
        "view_id",
        "sequence_id",
        "new_name",
        "width",
        "height",
        "status",
        "error",
    ]
    write_manifest(manifest_out, rows, fieldnames)

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[clean] manifest written to: {manifest_out}")
    print(f"[clean] summary written to: {summary_out}")


if __name__ == "__main__":
    main()
