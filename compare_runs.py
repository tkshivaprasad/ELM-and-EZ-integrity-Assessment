"""Put the results of several runs side by side (for the paper's comparison table).

Run:   python compare_runs.py
       python compare_runs.py --runs cv cv_pretrained jhu_zeroshot
"""
import argparse
import json

import pandas as pd

import config as C

LABELS = {"cv": "ImageNet only (baseline)", "cv_pretrained": "JHU pretrained + fine-tuned",
          "jhu_zeroshot": "JHU only (no fine-tuning)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["cv", "jhu_zeroshot", "cv_pretrained"])
    args = ap.parse_args()
    rows = []
    for name in args.runs:
        f = C.eval_dir(False, name) / "summary.json"
        if not f.exists():
            print(f"  {name}: no results yet ({f} missing); run: python evaluate.py --name {name}")
            continue
        s = json.loads(f.read_text())
        for layer in ("elm", "ez"):
            d = s.get(layer, {})
            rows.append({"Run": LABELS.get(name, name), "Layer": layer.upper(),
                         "Dice": d.get("dice"), "Tolerant Dice": d.get("tol_dice"),
                         "Boundary error (px)": d.get("boundary_err"), "Offset (px)": d.get("offset"),
                         "Coverage r": d.get("pearson_r"), "Coverage MAE": d.get("mae"),
                         "Integrity AUC": d.get("auc_pred")})
    if not rows:
        print("Nothing to compare yet.")
        return
    df = pd.DataFrame(rows)
    out = C.OUTPUT_DIR / "comparison.csv"
    df.round(3).to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
