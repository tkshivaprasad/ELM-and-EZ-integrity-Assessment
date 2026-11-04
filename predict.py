"""Run the trained models on new OCT images.

Run:   python predict.py --input "D:\\path\\to\\new_images"
       python predict.py --input "D:\\path\\to\\new_images" --output results_folder

For every image it saves a mask (0/1/2), a colour overlay, and one row in
predictions.csv with the predicted ELM/EZ coverage and an uncertainty score.
Images with the highest uncertainty are the most useful ones to annotate next.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import read_gray, write_png, write_jpg, overlay, column_coverage
from model_utils import load_models, predict_probs, get_device

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with OCT images")
    ap.add_argument("--output", default=str(C.OUTPUT_DIR / "predictions"))
    ap.add_argument("--run", default=str(C.run_dir(False)), help="training run folder with fold*/best.pt")
    ap.add_argument("--no-tta", action="store_true", help="skip the flipped-image averaging (faster)")
    args = ap.parse_args()

    device = get_device()
    models = load_models(Path(args.run), device)
    print(f"Loaded {len(models)} models on {device}")

    files = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {args.input}")
    out = Path(args.output)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(files, 1):
        img = read_gray(f)
        probs = predict_probs(models, img, device, tta=not args.no_tta)
        mask = probs.argmax(0).astype(np.uint8)
        layer_prob = 1 - probs[0]
        region = layer_prob > 0.1
        p = np.clip(probs[:, region], 1e-7, 1)
        uncertainty = float((-(p * np.log(p)).sum(0)).mean()) if region.any() else 0.0
        write_png(out / "masks" / f"{f.stem}.png", mask)
        write_jpg(out / "overlays" / f"{f.stem}.jpg", overlay(img, mask))
        rows.append(dict(filename=f.name, elm_coverage_pct=round(column_coverage(mask, 1), 1),
                         ez_coverage_pct=round(column_coverage(mask, 2), 1),
                         uncertainty=round(uncertainty, 4)))
        print(f"[{i}/{len(files)}] {f.name}: ELM {rows[-1]['elm_coverage_pct']}%, EZ {rows[-1]['ez_coverage_pct']}%")
    df = pd.DataFrame(rows).sort_values("uncertainty", ascending=False)
    df.to_csv(out / "predictions.csv", index=False)
    print(f"\nSaved to {out}. predictions.csv is sorted with the most uncertain images first.")


if __name__ == "__main__":
    main()
