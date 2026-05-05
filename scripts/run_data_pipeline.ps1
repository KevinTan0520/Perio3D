$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed: python $($Arguments -join ' ')"
    }
}

Write-Host "[1/5] Scan dataset"
Invoke-PythonStep @("-m", "src.data.scan_dataset", "--dataset-root", "dataset", "--output-dir", "outputs/reports")

Write-Host "[2/5] Build case metadata"
Invoke-PythonStep @("-m", "src.data.build_metadata", "--quality-workbook", "dataset/SIOP quality evaluation.xlsx", "--dataset-root", "dataset", "--metadata-out", "data/processed/metadata.csv", "--summary-out", "outputs/reports/metadata_summary.json")

Write-Host "[3/5] Clean dataset"
Invoke-PythonStep @("-m", "src.data.clean_dataset", "--dataset-root", "dataset", "--output-root", "data/processed/images", "--manifest-out", "data/processed/clean_manifest.csv", "--summary-out", "outputs/reports/clean_summary.json", "--reset-output")

Write-Host "[4/5] Build splits"
Invoke-PythonStep @("-m", "src.data.build_splits", "--manifest", "data/processed/clean_manifest.csv", "--metadata", "data/processed/metadata.csv", "--split-dir", "data/splits", "--train-ratio", "0.7", "--val-ratio", "0.15", "--test-ratio", "0.15", "--seed", "42", "--stratify-column", "gingival_index")

Write-Host "[5/5] Generate consolidated report"
Invoke-PythonStep @("-m", "src.data.generate_data_report", "--reports-dir", "outputs/reports", "--manifest", "data/processed/clean_manifest.csv", "--metadata", "data/processed/metadata.csv", "--split-dir", "data/splits", "--report-out", "outputs/reports/data_quality_report.md")

Write-Host "Data pipeline completed."
