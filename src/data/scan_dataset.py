from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PPT_EXTS = {".ppt", ".pptx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan dataset structure and quality.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def is_mac_metadata(path: Path) -> bool:
    return path.name == ".DS_Store" or path.name.startswith("._")


def parse_case_ids(dataset_root: Path, file_path: Path) -> Tuple[str, str]:
    rel = file_path.relative_to(dataset_root)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def inspect_image(path: Path) -> Tuple[bool, int, int, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return True, width, height, ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, 0, 0, str(exc)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    summary = {
        "total_files": 0,
        "image_files": 0,
        "valid_images": 0,
        "broken_images": 0,
        "ppt_files": 0,
        "mac_metadata_files": 0,
        "other_files": 0,
    }

    resolution_counter: Counter[str] = Counter()
    case_stats = defaultdict(lambda: {"images": 0, "valid_images": 0, "broken_images": 0})

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        summary["total_files"] += 1

        if is_mac_metadata(file_path):
            summary["mac_metadata_files"] += 1
            continue

        suffix = file_path.suffix.lower()
        group_id, case_id = parse_case_ids(dataset_root, file_path)
        case_key = f"{group_id}/{case_id}"

        if suffix in IMAGE_EXTS:
            summary["image_files"] += 1
            case_stats[case_key]["images"] += 1
            is_valid, width, height, _ = inspect_image(file_path)
            if is_valid:
                summary["valid_images"] += 1
                case_stats[case_key]["valid_images"] += 1
                resolution_counter[f"{width}x{height}"] += 1
            else:
                summary["broken_images"] += 1
                case_stats[case_key]["broken_images"] += 1
        elif suffix in PPT_EXTS:
            summary["ppt_files"] += 1
        else:
            summary["other_files"] += 1

    case_rows = []
    for case_key in sorted(case_stats.keys()):
        group_id, case_id = case_key.split("/")
        row = {
            "group_id": group_id,
            "case_id": case_id,
            "images": case_stats[case_key]["images"],
            "valid_images": case_stats[case_key]["valid_images"],
            "broken_images": case_stats[case_key]["broken_images"],
        }
        case_rows.append(row)

    case_csv = output_dir / "case_image_summary.csv"
    write_csv(case_csv, case_rows, ["group_id", "case_id", "images", "valid_images", "broken_images"])

    top_resolutions = [{"resolution": k, "count": v} for k, v in resolution_counter.most_common(10)]
    scan_json = {
        "summary": summary,
        "num_cases": len(case_rows),
        "top_resolutions": top_resolutions,
    }

    scan_json_path = output_dir / "dataset_scan.json"
    with scan_json_path.open("w", encoding="utf-8") as f:
        json.dump(scan_json, f, indent=2)

    report_path = output_dir / "data_quality_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Data Quality Report\n\n")
        f.write("## Summary\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"- num_cases: {len(case_rows)}\n\n")

        f.write("## Top Image Resolutions\n")
        for item in top_resolutions:
            f.write(f"- {item['resolution']}: {item['count']}\n")
        f.write("\n")

        f.write("## Cases With Fewest Valid Images (Top 20)\n")
        sorted_cases = sorted(case_rows, key=lambda r: (r["valid_images"], r["images"]))
        for row in sorted_cases[:20]:
            f.write(
                f"- {row['group_id']}/{row['case_id']}: "
                f"images={row['images']}, valid={row['valid_images']}, broken={row['broken_images']}\n"
            )

    print(f"[scan] summary written to: {scan_json_path}")
    print(f"[scan] case table written to: {case_csv}")
    print(f"[scan] report written to: {report_path}")


if __name__ == "__main__":
    main()
