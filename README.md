# Oral Photo 3D Reconstruction and Periodontitis Diagnosis （Perio3D）

## Project Scope
This repository implements a Python pipeline for:
1. Data governance for intraoral photo groups.
2. Tooth and gingiva segmentation workflow integration.
3. Multi-view 3D reconstruction integration (VGGT).
4. Semantic constraint fusion from 2D masks to 3D geometry.
5. Case-level periodontitis risk/stage modeling.

## Folder Layout
- `src/data`: dataset scan, cleaning, split generation.
- `src/seg`: segmentation related modules (planned).
- `src/recon`: reconstruction interface modules (planned).
- `src/diag`: diagnosis models and fusion (planned).
- `src/eval`: metric and report modules (planned).
- `configs`: YAML configuration files.
- `scripts`: launch scripts.
- `outputs`: generated reports and model outputs.
- `data/processed`: cleaned data assets.
- `data/splits`: train/val/test image lists.

## Quick Start
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full phase B data pipeline:

```powershell
.\scripts\run_data_pipeline.ps1
```

Or run each step manually:

Scan the dataset:

```bash
python -m src.data.scan_dataset --dataset-root dataset --output-dir outputs/reports
```

Build case-level metadata from the SIOP quality workbook:

```bash
python -m src.data.build_metadata --quality-workbook "dataset/SIOP quality evaluation.xlsx" --dataset-root dataset --metadata-out data/processed/metadata.csv
```

Clean and normalize image files:

```bash
python -m src.data.clean_dataset --dataset-root dataset --output-root data/processed/images --manifest-out data/processed/clean_manifest.csv --reset-output
```

Build case-level train/val/test splits:

```bash
python -m src.data.build_splits --manifest data/processed/clean_manifest.csv --metadata data/processed/metadata.csv --split-dir data/splits --stratify-column gingival_index
```

Regenerate the consolidated data quality report:

```bash
python -m src.data.generate_data_report --reports-dir outputs/reports --manifest data/processed/clean_manifest.csv --metadata data/processed/metadata.csv --split-dir data/splits
```

Run the phase C segmentation pipeline:

```powershell
.\scripts\run_seg_pipeline.ps1
```

For a quick smoke test:

```powershell
.\scripts\run_seg_pipeline.ps1 -MaxImages 12
```

Phase C generates an annotation queue/schema for X-AnyLabeling/SAM2.1 exports, baseline tooth/gingiva masks, visible previews under `outputs/seg/vis`, a prediction manifest with confidence and coverage fields, and `outputs/seg/metrics.json`. Dice/IoU are populated when gold binary masks exist at the paths listed in `outputs/seg/annotation_queue.csv`.

## Notes
- Mac metadata files (`._*`, `.DS_Store`) are ignored by cleaning scripts.
- Split generation is case-level and can stratify by a metadata column such as `gingival_index`.
- `metadata.csv` contains case-level IQS scores, overall evaluation, gingival index, and a coarse quality bucket.
- The current implementation is the first runnable baseline and will evolve phase by phase.
- Phase C currently uses a Pillow/NumPy color-threshold baseline as a replaceable segmentation backend; SAM2.1 or a trained lightweight model can write the same mask manifest contract.
- Files under `outputs/seg/masks` are single-channel label masks, where pixel values are 0=background, 1=tooth, and 2=gingiva. They may look black in a normal image viewer; use `outputs/seg/vis` for human-readable color masks and overlays.
