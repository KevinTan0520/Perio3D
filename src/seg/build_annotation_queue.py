from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.seg.io_utils import read_csv_rows, write_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced phase C annotation queue.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/seg/annotation_queue.csv"))
    parser.add_argument("--schema-out", type=Path, default=Path("outputs/seg/annotation_schema.json"))
    parser.add_argument("--samples-per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split_rows(split_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_path = split_dir / f"{split}.csv"
        for row in read_csv_rows(split_path):
            row["split"] = split
            rows.append(row)
    return rows


def choose_rows(rows: List[Dict[str, str]], samples_per_split: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    selected: List[Dict[str, str]] = []
    split_to_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        split_to_rows[row["split"]].append(row)

    for split, split_rows in sorted(split_to_rows.items()):
        strata: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in split_rows:
            key = row.get("gingival_index") or "unknown"
            strata[key].append(row)

        split_selected: List[Dict[str, str]] = []
        quota = max(1, samples_per_split // max(1, len(strata)))
        for stratum_rows in strata.values():
            stratum_rows = sorted(stratum_rows, key=lambda item: (item["case_uid"], item.get("view_id", "")))
            rng.shuffle(stratum_rows)
            split_selected.extend(stratum_rows[:quota])

        if len(split_selected) < samples_per_split:
            already = {row["image_path"] for row in split_selected}
            remaining = [row for row in split_rows if row["image_path"] not in already]
            rng.shuffle(remaining)
            split_selected.extend(remaining[: samples_per_split - len(split_selected)])

        selected.extend(split_selected[:samples_per_split])

    return sorted(selected, key=lambda row: (row["split"], row["case_uid"], row.get("view_id", "")))


def main() -> None:
    args = parse_args()
    rows = load_split_rows(args.split_dir)
    selected = choose_rows(rows, args.samples_per_split, args.seed)

    out_rows = []
    for index, row in enumerate(selected, start=1):
        mask_base = Path("data/annotations/seg") / row["case_uid"] / Path(row["new_name"]).stem
        out_rows.append(
            {
                "queue_id": f"SEG-{index:04d}",
                "split": row["split"],
                "case_uid": row["case_uid"],
                "group_id": row["group_id"],
                "case_id": row["case_id"],
                "view_id": row.get("view_id", ""),
                "image_path": row["image_path"],
                "new_name": row["new_name"],
                "gingival_index": row.get("gingival_index", ""),
                "quality_bucket": row.get("quality_bucket", ""),
                "tooth_mask_path": f"{mask_base}_tooth.png",
                "gingiva_mask_path": f"{mask_base}_gingiva.png",
                "review_status": "pending",
            }
        )

    fieldnames = [
        "queue_id",
        "split",
        "case_uid",
        "group_id",
        "case_id",
        "view_id",
        "image_path",
        "new_name",
        "gingival_index",
        "quality_bucket",
        "tooth_mask_path",
        "gingiva_mask_path",
        "review_status",
    ]
    write_csv_rows(args.output_csv, out_rows, fieldnames)

    schema = {
        "classes": [
            {"id": 0, "name": "background", "color": [0, 0, 0]},
            {"id": 1, "name": "tooth", "color": [255, 255, 255]},
            {"id": 2, "name": "gingiva", "color": [255, 96, 96]},
        ],
        "mask_encoding": "single-channel PNG, uint8 class ids",
        "binary_gold_masks": {
            "tooth": "data/annotations/seg/{case_uid}/{image_stem}_tooth.png",
            "gingiva": "data/annotations/seg/{case_uid}/{image_stem}_gingiva.png",
        },
        "source": "X-AnyLabeling/SAM2.1 exports can be converted to the paths listed in annotation_queue.csv.",
    }
    write_json(args.schema_out, schema)

    print(f"[seg-queue] annotation queue written to: {args.output_csv}")
    print(f"[seg-queue] schema written to: {args.schema_out}")


if __name__ == "__main__":
    main()

