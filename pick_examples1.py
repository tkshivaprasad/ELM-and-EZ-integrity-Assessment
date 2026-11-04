"""Contact sheets to choose clinically representative examples for Figure 3.

Renders every image of a class, cropped to the outer retina and enlarged, with its
Dice, integrity grades and coverage printed beneath. Show the sheet to the
ophthalmologist, let her pick the panel that best represents the disease, then pass
the chosen file names to make_figures.py with --examples.

    python pick_examples.py                         # all classes
    python pick_examples.py --cls Drusen AMD        # only these
    python pick_examples.py --typical               # only images within +/-0.05
                                                    #   of the class median Dice
Sheets are written to outputs/example_candidates/ (change with --out). Nothing in an
existing outputs/figures/ folder is modified.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from utils import load_table, read_gray

# label shown on the sheet for each source class
DISPLAY = {"Normal": "Normal", "DME": "DME",
           "Drusen": "AMD, non-neovascular (drusen)",
           "AMD": "AMD, neovascular (CNV)"}


def crop_outer_retina(img, aspect=2.6):
    """Crop vertically around the bright outer-retinal complex."""
    prof = img.mean(axis=1).astype(float)
    k = max(3, img.shape[0] // 60)
    prof = np.convolve(prof, np.ones(k) / k, "same")
    rows = np.where(prof > prof.mean() + 0.5 * prof.std())[0]
    if rows.size == 0:
        return img
    centre = (rows.min() + rows.max()) / 2
    h = max(img.shape[1] / aspect, (rows.max() - rows.min()) * 1.8)
    top = int(np.clip(centre - h / 2, 0, max(0, img.shape[0] - h)))
    return img[top:int(min(img.shape[0], top + h))]


def stretch(img, lo=1.0, hi=99.5):
    """Percentile contrast stretch, so washed-out scans stay readable."""
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return img
    return np.clip((img.astype(float) - a) / (b - a), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cv", help="run whose metrics to label with")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cls", nargs="*", default=None, help="classes to render")
    ap.add_argument("--typical", action="store_true",
                    help="keep only images within 0.05 of the class median mean-Dice")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="example_candidates", metavar="FOLDER",
                    help="folder under outputs/ for the sheets (default: example_candidates, "
                         "a folder of its own so existing figures are not touched)")
    a = ap.parse_args()

    ev = C.eval_dir(a.quick, a.name)
    f = ev / "metrics_per_image.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run evaluate.py first.")
    res = pd.read_csv(f)
    res["dice_mean"] = res[["dice_elm", "dice_ez"]].mean(axis=1)
    tbl = load_table().set_index("filename")

    out = C.OUTPUT_DIR / a.out
    out.mkdir(parents=True, exist_ok=True)

    classes = a.cls or [c for c in DISPLAY if c in set(res.disease_class)]
    for cls in classes:
        g = res[res.disease_class == cls].copy()
        if g.empty:
            print(f"no images for class {cls}"); continue
        med = g.dice_mean.median()
        if a.typical:
            g = g[(g.dice_mean - med).abs() <= 0.05]
        g = g.sort_values("dice_mean", ascending=False).reset_index(drop=True)

        cols = a.cols
        rows = int(np.ceil(len(g) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.5 * rows), squeeze=False)
        for i, row in g.iterrows():
            ax = axes[i // cols][i % cols]
            img = read_gray(tbl.loc[row.filename, "image_path"])
            ax.imshow(stretch(crop_outer_retina(img)), cmap="gray", aspect="auto",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            flag = "  <-- current median pick" if abs(row.dice_mean - med) < 1e-9 else ""
            ax.set_title(
                f"{row.filename}{flag}\nDice {row.dice_elm:.2f}/{row.dice_ez:.2f}   "
                f"grades {row.elm_grade[:4]}/{row.ez_grade[:4]}   "
                f"cov {row.cov_traced_elm:.0f}%/{row.cov_traced_ez:.0f}%",
                fontsize=8, loc="left")
        for j in range(len(g), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"{DISPLAY.get(cls, cls)}   (n = {len(g)}, class median mean-Dice "
                     f"{med:.3f}; images cropped to the outer retina, contrast stretched)",
                     fontsize=11, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        p = out / f"candidates_{cls}.jpg"
        fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
        print(f"wrote {p}  ({len(g)} images)")

    print("\nShow these sheets to the ophthalmologist, then regenerate Figure 3 with, e.g.:")
    print('  python make_figures.py --out figures_revised '
          '--examples "Drusen=DRUSEN-XXXX-X.jpeg" "AMD=amd_XXXXXXX_X.jpg"')


if __name__ == "__main__":
    main()
