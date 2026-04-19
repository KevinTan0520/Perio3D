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

3. Run data scan:

```bash
python -m src.data.scan_dataset --dataset-root dataset --output-dir outputs/reports
```

4. Run data cleaning:

```bash
python -m src.data.clean_dataset --dataset-root dataset --output-root data/processed/images --manifest-out data/processed/clean_manifest.csv
```

5. Build splits:

```bash
python -m src.data.build_splits --manifest data/processed/clean_manifest.csv --split-dir data/splits
```

## Notes
- Mac metadata files (`._*`, `.DS_Store`) are ignored by cleaning scripts.
- Split generation is case-level to reduce data leakage.
- The current implementation is the first runnable baseline and will evolve phase by phase.
