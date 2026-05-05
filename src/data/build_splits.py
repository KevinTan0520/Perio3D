from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build case-level train/val/test splits from clean manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/clean_manifest.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument("--stratify-column", default="gingival_index")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["case_uid"]: row for row in csv.DictReader(f) if row.get("case_uid")}


def case_uid_from_row(row: Dict[str, str]) -> str:
    if row.get("case_uid"):
        return row["case_uid"]
    return f"{row['group_id']}-{row['case_id']}"


def allocate_cases(case_uids: List[str], train_ratio: float, val_ratio: float) -> tuple[set[str], set[str], set[str]]:
    n_cases = len(case_uids)
    n_train = int(n_cases * train_ratio)
    n_val = int(n_cases * val_ratio)
    train_cases = set(case_uids[:n_train])
    val_cases = set(case_uids[n_train : n_train + n_val])
    test_cases = set(case_uids[n_train + n_val :])
    return train_cases, val_cases, test_cases


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio must sum to 1.0")

    rows = read_manifest(args.manifest)
    copied_rows = [r for r in rows if r.get("status") == "copied"]
    if not copied_rows:
        raise ValueError("no copied image rows found in manifest")

    case_to_rows: Dict[str, List[Dict[str, str]]] = {}
    for row in copied_rows:
        case_uid = case_uid_from_row(row)
        case_to_rows.setdefault(case_uid, []).append(row)

    metadata = read_metadata(args.metadata)
    case_uids = sorted(case_to_rows.keys())
    rng = random.Random(args.seed)
    stratify_used = False
    train_cases: set[str] = set()
    val_cases: set[str] = set()
    test_cases: set[str] = set()

    if metadata and args.stratify_column:
        strata: Dict[str, List[str]] = defaultdict(list)
        for case_uid in case_uids:
            value = metadata.get(case_uid, {}).get(args.stratify_column, "")
            strata[value if value != "" else "unknown"].append(case_uid)

        if len(strata) > 1:
            stratify_used = True
            for stratum_cases in strata.values():
                stratum_cases = sorted(stratum_cases)
                rng.shuffle(stratum_cases)
                stratum_train, stratum_val, stratum_test = allocate_cases(
                    stratum_cases, args.train_ratio, args.val_ratio
                )
                train_cases.update(stratum_train)
                val_cases.update(stratum_val)
                test_cases.update(stratum_test)

    if not stratify_used:
        rng.shuffle(case_uids)
        train_cases, val_cases, test_cases = allocate_cases(case_uids, args.train_ratio, args.val_ratio)

    n_cases = len(case_uids)

    def collect(case_set: set[str], split_name: str) -> List[Dict[str, str]]:
        out = []
        for case_uid in sorted(case_set):
            for row in case_to_rows[case_uid]:
                out.append(
                    {
                        "split": split_name,
                        "case_uid": case_uid,
                        "group_id": row["group_id"],
                        "case_id": row["case_id"],
                        "view_id": row.get("view_id", ""),
                        "image_path": row["dest_path"],
                        "new_name": row["new_name"],
                        "width": row["width"],
                        "height": row["height"],
                        "gingival_index": metadata.get(case_uid, {}).get("gingival_index", ""),
                        "overall_evaluation": metadata.get(case_uid, {}).get("overall_evaluation", ""),
                        "quality_bucket": metadata.get(case_uid, {}).get("quality_bucket", ""),
                    }
                )
        return out

    train_rows = collect(train_cases, "train")
    val_rows = collect(val_cases, "val")
    test_rows = collect(test_cases, "test")

    fieldnames = [
        "split",
        "case_uid",
        "group_id",
        "case_id",
        "view_id",
        "image_path",
        "new_name",
        "width",
        "height",
        "gingival_index",
        "overall_evaluation",
        "quality_bucket",
    ]
    write_rows(args.split_dir / "train.csv", train_rows, fieldnames)
    write_rows(args.split_dir / "val.csv", val_rows, fieldnames)
    write_rows(args.split_dir / "test.csv", test_rows, fieldnames)

    summary = {
        "num_cases": n_cases,
        "num_train_cases": len(train_cases),
        "num_val_cases": len(val_cases),
        "num_test_cases": len(test_cases),
        "num_train_images": len(train_rows),
        "num_val_images": len(val_rows),
        "num_test_images": len(test_rows),
        "seed": args.seed,
        "stratify_column": args.stratify_column,
        "stratify_used": stratify_used,
    }

    with (args.split_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[split] train: {args.split_dir / 'train.csv'}")
    print(f"[split] val: {args.split_dir / 'val.csv'}")
    print(f"[split] test: {args.split_dir / 'test.csv'}")
    print(f"[split] summary: {args.split_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
