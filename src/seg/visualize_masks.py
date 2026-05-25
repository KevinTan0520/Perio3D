from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.seg.io_utils import read_csv_rows


PALETTE = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (255, 64, 96),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visible color previews for phase C label masks.")
    parser.add_argument("--pred-manifest", type=Path, default=Path("outputs/seg/pred_manifest.csv"))
    parser.add_argument("--vis-dir", type=Path, default=Path("outputs/seg/vis"))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--preview-long-edge", type=int, default=1600)
    return parser.parse_args()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, rgb in PALETTE.items():
        color[mask == label_id] = rgb
    return color


def blend_overlay(image: Image.Image, color_mask: np.ndarray, mask: np.ndarray, alpha: float) -> Image.Image:
    image = image.convert("RGB")
    if image.size != (mask.shape[1], mask.shape[0]):
        image = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    image_arr = np.asarray(image).astype(np.float32)
    color_arr = color_mask.astype(np.float32)
    semantic = mask > 0
    out = image_arr.copy()
    out[semantic] = (1.0 - alpha) * image_arr[semantic] + alpha * color_arr[semantic]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def resize_preview(image: Image.Image, mask: np.ndarray, long_edge: int) -> tuple[Image.Image, np.ndarray]:
    if long_edge <= 0:
        return image, mask
    width, height = image.size
    current_long_edge = max(width, height)
    if current_long_edge <= long_edge:
        return image, mask
    scale = long_edge / float(current_long_edge)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    mask_img = Image.fromarray(mask, mode="L").resize(new_size, Image.Resampling.NEAREST)
    return image, np.asarray(mask_img)


def output_paths(vis_dir: Path, row: Dict[str, str]) -> tuple[Path, Path]:
    stem = Path(row["mask_path"]).stem
    base = vis_dir / row["split"] / row["case_uid"]
    return base / f"{stem}_color.png", base / f"{stem}_overlay.jpg"


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = [row for row in read_csv_rows(args.pred_manifest) if row.get("status") == "ok"]
    if args.max_images > 0:
        rows = rows[: args.max_images]
    if not rows:
        raise ValueError(f"no successful predictions found in {args.pred_manifest}")

    written = 0
    for row in rows:
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"))
        image = Image.open(row["image_path"]).convert("RGB")
        image, mask = resize_preview(image, mask, args.preview_long_edge)
        color_mask = colorize_mask(mask)
        color_path, overlay_path = output_paths(args.vis_dir, row)
        color_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color_mask, mode="RGB").save(color_path)

        overlay = blend_overlay(image, color_mask, mask, args.alpha)
        overlay.save(overlay_path, quality=92)
        written += 1

    print(f"[seg-vis] wrote {written} color masks and overlays under: {args.vis_dir}")


if __name__ == "__main__":
    main()
