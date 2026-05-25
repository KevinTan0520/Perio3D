param(
    [int]$SamplesPerSplit = 30,
    [int]$MaxImages = 0,
    [int]$ResizeLongEdge = 1600
)

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

Write-Host "[1/3] Build annotation queue and schema"
Invoke-PythonStep @("-m", "src.seg.build_annotation_queue", "--split-dir", "data/splits", "--output-csv", "outputs/seg/annotation_queue.csv", "--schema-out", "outputs/seg/annotation_schema.json", "--samples-per-split", "$SamplesPerSplit", "--seed", "42")

Write-Host "[2/3] Generate baseline tooth/gingiva masks"
Invoke-PythonStep @("-m", "src.seg.baseline_segment", "--split-dir", "data/splits", "--mask-dir", "outputs/seg/masks", "--manifest-out", "outputs/seg/pred_manifest.csv", "--summary-out", "outputs/seg/pred_summary.json", "--max-images", "$MaxImages", "--resize-long-edge", "$ResizeLongEdge")

Write-Host "[3/3] Evaluate masks"
Invoke-PythonStep @("-m", "src.seg.evaluate_masks", "--pred-manifest", "outputs/seg/pred_manifest.csv", "--annotation-queue", "outputs/seg/annotation_queue.csv", "--metrics-out", "outputs/seg/metrics.json")

Write-Host "Segmentation pipeline completed."
