$ErrorActionPreference = "Stop"

Write-Host "[1/3] Scan dataset"
python -m src.data.scan_dataset --dataset-root dataset --output-dir outputs/reports

Write-Host "[2/3] Clean dataset"
python -m src.data.clean_dataset --dataset-root dataset --output-root data/processed/images --manifest-out data/processed/clean_manifest.csv --summary-out outputs/reports/clean_summary.json

Write-Host "[3/3] Build splits"
python -m src.data.build_splits --manifest data/processed/clean_manifest.csv --split-dir data/splits --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42

Write-Host "Data pipeline completed."
