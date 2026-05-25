from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SEG_CLASSES = {"background": 0, "tooth": 1, "gingiva": 2}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def mask_stem(row: Dict[str, str]) -> str:
    image_name = Path(row.get("new_name") or row["image_path"]).stem
    return f"{row['case_uid']}_{row.get('view_id', 'view')}_{image_name}"


def relative_mask_path(mask_root: Path, row: Dict[str, str]) -> Path:
    return mask_root / row["split"] / row["case_uid"] / f"{mask_stem(row)}.png"

