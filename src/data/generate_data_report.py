from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a consolidated data governance report.")
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/clean_manifest.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--report-out", type=Path, default=Path("outputs/reports/data_quality_report.md"))
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_by(rows: Iterable[Dict[str, str]], column: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(column, "") or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def count_cases_by(rows: Iterable[Dict[str, str]], column: str) -> Dict[str, int]:
    case_values: Dict[str, str] = {}
    for row in rows:
        case_uid = row.get("case_uid", "")
        if not case_uid:
            continue
        case_values[case_uid] = row.get(column, "") or "unknown"
    return count_by([{column: value} for value in case_values.values()], column)


def append_kv_lines(lines: List[str], title: str, values: Dict[str, Any]) -> None:
    lines.append(f"## {title}")
    if not values:
        lines.append("- Not generated.")
        lines.append("")
        return
    for key, value in values.items():
        lines.append(f"- {key}: {value}")
    lines.append("")


def main() -> None:
    args = parse_args()
    scan = read_json(args.reports_dir / "dataset_scan.json")
    clean = read_json(args.reports_dir / "clean_summary.json")
    metadata_summary = read_json(args.reports_dir / "metadata_summary.json")
    split_summary = read_json(args.split_dir / "split_summary.json")
    manifest_rows = read_csv_rows(args.manifest)
    metadata_rows = read_csv_rows(args.metadata)
    train_rows = read_csv_rows(args.split_dir / "train.csv")
    val_rows = read_csv_rows(args.split_dir / "val.csv")
    test_rows = read_csv_rows(args.split_dir / "test.csv")

    copied_rows = [row for row in manifest_rows if row.get("status") == "copied"]
    broken_rows = [row for row in manifest_rows if row.get("status") == "broken"]
    few_image_cases = sorted(
        metadata_rows,
        key=lambda row: int(float(row["dataset_image_count"])) if row.get("dataset_image_count") else 0,
    )[:20]

    lines: List[str] = [
        "# Data Quality Report",
        "",
        "This report is generated from the phase B data governance pipeline.",
        "",
    ]
    append_kv_lines(lines, "Dataset Scan", scan.get("summary", {}))
    append_kv_lines(lines, "Cleaning Summary", clean)
    append_kv_lines(lines, "Metadata Summary", metadata_summary)
    append_kv_lines(lines, "Split Summary", split_summary)

    lines.append("## Manifest Checks")
    lines.append(f"- manifest_rows: {len(manifest_rows)}")
    lines.append(f"- copied_rows: {len(copied_rows)}")
    lines.append(f"- broken_rows: {len(broken_rows)}")
    lines.append(f"- copied_case_count: {len({row.get('case_uid') for row in copied_rows})}")
    lines.append("")

    lines.append("## Label And Quality Distributions")
    lines.append(f"- metadata_gingival_index: {count_by(metadata_rows, 'gingival_index')}")
    lines.append(f"- metadata_quality_bucket: {count_by(metadata_rows, 'quality_bucket')}")
    lines.append(f"- train_case_gingival_index: {count_cases_by(train_rows, 'gingival_index')}")
    lines.append(f"- val_case_gingival_index: {count_cases_by(val_rows, 'gingival_index')}")
    lines.append(f"- test_case_gingival_index: {count_cases_by(test_rows, 'gingival_index')}")
    lines.append(f"- train_image_gingival_index: {count_by(train_rows, 'gingival_index')}")
    lines.append(f"- val_image_gingival_index: {count_by(val_rows, 'gingival_index')}")
    lines.append(f"- test_image_gingival_index: {count_by(test_rows, 'gingival_index')}")
    lines.append("")

    lines.append("## Cases With Fewest Dataset Images")
    for row in few_image_cases:
        lines.append(
            f"- {row['case_uid']}: images={row.get('dataset_image_count', '')}, "
            f"GI={row.get('gingival_index', '')}, quality={row.get('quality_bucket', '')}"
        )
    lines.append("")

    if broken_rows:
        lines.append("## Broken Images")
        for row in broken_rows[:50]:
            lines.append(f"- {row.get('source_path', '')}: {row.get('error', '')}")
        if len(broken_rows) > 50:
            lines.append(f"- ... {len(broken_rows) - 50} more")
        lines.append("")

    lines.append("## Generated Assets")
    lines.append(f"- Manifest: `{args.manifest.as_posix()}`")
    lines.append(f"- Metadata: `{args.metadata.as_posix()}`")
    lines.append(f"- Splits: `{args.split_dir.as_posix()}`")
    lines.append("")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] report written to: {args.report_out}")


if __name__ == "__main__":
    main()
