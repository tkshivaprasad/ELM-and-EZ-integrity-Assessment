"""Run nnU-Net (2D) on the same data and the same 5 folds as the U-Net baseline,
then bring its predictions back so evaluate.py / stats_compare.py can compare them.

Before using this, set the three nnU-Net folders for this PowerShell window
(use a path WITHOUT spaces, e.g. D:\\nnunet):
    $env:nnUNet_raw="D:\\nnunet\\raw"
    $env:nnUNet_preprocessed="D:\\nnunet\\preprocessed"
    $env:nnUNet_results="D:\\nnunet\\results"

Steps:
  1. python prepare_nnunet.py export        (writes images, labels, dataset.json)
  2. nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
  3. python prepare_nnunet.py splits        (makes nnU-Net use YOUR 5 folds)
  4. nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs   (then folds 1, 2, 3, 4)
  5. python prepare_nnunet.py import -tr nnUNetTrainer_250epochs
  6. python evaluate.py --name nnunet
     python stats_compare.py --a cv --b nnunet

The folds are read from outputs/cv/folds.csv (made by train_cv.py), so run the
baseline first. nnU-Net trains on full-resolution patches, so no resizing to 512.
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import load_table, read_gray, read_mask, write_png

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_OCTELMEZ"


def env_dir(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"The environment variable {name} is not set. In PowerShell run:\n"
            f'  $env:nnUNet_raw="D:\\nnunet\\raw"\n'
            f'  $env:nnUNet_preprocessed="D:\\nnunet\\preprocessed"\n'
            f'  $env:nnUNet_results="D:\\nnunet\\results"\n'
            "then run this command again in the same window.")
    if " " in v:
        print(f"Warning: {name} contains spaces ({v}); nnU-Net is safer with a path without spaces.")
    return Path(v)


def case_id(stem):
    """nnU-Net-safe case name (letters, digits, underscore, dash)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def load_folds():
    f = C.run_dir(False, "cv") / "folds.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run the baseline first (python train_cv.py) so both models use the same folds.")
    return pd.read_csv(f)


def export():
    raw = env_dir("nnUNet_raw") / DATASET_NAME
    df = load_table()
    folds = load_folds()
    df = df.merge(folds[["filename", "fold"]], on="filename", how="inner")
    if len(df) == 0:
        raise SystemExit("No images match outputs/cv/folds.csv. Re-run train_cv.py on the current data.")
    if raw.exists():
        shutil.rmtree(raw)
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    mapping, seen = {}, set()
    for r in df.to_dict("records"):
        cid = case_id(r["stem"])
        if cid in seen:
            raise SystemExit(f"Two images map to the same case name {cid}; rename one of them.")
        seen.add(cid)
        img, mask = read_gray(r["image_path"]), read_mask(r["mask_path"])
        if img.shape != mask.shape:
            raise SystemExit(f"Image and mask sizes differ for {r['filename']}.")
        if mask.max() > 2:
            raise SystemExit(f"Unexpected label values in the mask of {r['filename']}.")
        write_png(raw / "imagesTr" / f"{cid}_0000.png", img)
        write_png(raw / "labelsTr" / f"{cid}.png", mask.astype(np.uint8))
        mapping[cid] = {"filename": r["filename"], "stem": r["stem"], "fold": int(r["fold"])}
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "OCT"},
        "labels": {"background": 0, "ELM": 1, "EZ": 2},
        "numTraining": len(mapping),
        "file_ending": ".png",
    }, indent=2))
    (raw / "case_mapping.json").write_text(json.dumps(mapping, indent=2))
    n_folds = len({m["fold"] for m in mapping.values()})
    print(f"Exported {len(mapping)} cases ({n_folds} folds) to {raw}")
    print(f"Next: nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity")


def splits():
    raw = env_dir("nnUNet_raw") / DATASET_NAME
    pre = env_dir("nnUNet_preprocessed") / DATASET_NAME
    if not (raw / "case_mapping.json").exists():
        raise SystemExit("Run 'python prepare_nnunet.py export' first.")
    if not pre.exists():
        raise SystemExit(f"{pre} not found. Run nnUNetv2_plan_and_preprocess first.")
    mapping = json.loads((raw / "case_mapping.json").read_text())
    folds = sorted({m["fold"] for m in mapping.values()})
    out = []
    for k in folds:
        val = sorted(c for c, m in mapping.items() if m["fold"] == k)
        tr = sorted(c for c, m in mapping.items() if m["fold"] != k)
        out.append({"train": tr, "val": val})
    (pre / "splits_final.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} folds to {pre / 'splits_final.json'} (same folds as the U-Net baseline).")
    print(f"Next: nnUNetv2_train {DATASET_ID} 2d 0 -tr nnUNetTrainer_250epochs   (repeat for folds 1-{len(out) - 1})")


def import_preds(trainer, plans, name):
    raw = env_dir("nnUNet_raw") / DATASET_NAME
    res = env_dir("nnUNet_results") / DATASET_NAME
    mapping = json.loads((raw / "case_mapping.json").read_text())
    model_dir = res / f"{trainer}__{plans}__2d"
    if not model_dir.exists():
        found = [p.name for p in res.glob("*__2d")] if res.exists() else []
        raise SystemExit(f"{model_dir} not found. Trained models present: {found or 'none'}. Use -tr to choose.")
    out = C.run_dir(False, name)
    (out / "oof_masks").mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for cid, m in mapping.items():
        p = model_dir / f"fold_{m['fold']}" / "validation" / f"{cid}.png"
        if not p.exists():
            missing.append(f"fold {m['fold']}: {cid}")
            continue
        pred = read_mask(p)
        write_png(out / "oof_masks" / f"{m['stem']}.png", pred.astype(np.uint8))
        rows.append({"filename": m["filename"], "fold": m["fold"]})
    if missing:
        print(f"Missing {len(missing)} validation predictions (has every fold finished training?): {missing[:5]}")
    df = load_table()[["filename", "disease_class"]].merge(pd.DataFrame(rows), on="filename")
    df.to_csv(out / "folds.csv", index=False)
    print(f"Imported {len(df)} nnU-Net predictions into {out}")
    print(f"Next: python evaluate.py --name {name}   then   python stats_compare.py --a cv --b {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "splits", "import"])
    ap.add_argument("-tr", "--trainer", default="nnUNetTrainer_250epochs")
    ap.add_argument("-p", "--plans", default="nnUNetPlans")
    ap.add_argument("--name", default="nnunet", help="run name used by evaluate.py")
    a = ap.parse_args()
    if a.step == "export":
        export()
    elif a.step == "splits":
        splits()
    else:
        import_preds(a.trainer, a.plans, a.name)


if __name__ == "__main__":
    main()
