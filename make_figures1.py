"""Make publication-quality figures from the training and evaluation results.

Run after train_cv.py and evaluate.py:
    python make_figures.py
    python make_figures.py --image amd_train_2910.jpg   (choose the image for the integrity-profile figure)

Saves PNG (300 dpi) and PDF versions of each figure in outputs/figures/.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import config as C
from utils import load_table, read_gray, read_mask, overlay, column_coverage

CLASS_ORDER = ["Normal", "DME", "Drusen", "AMD"]
# labels printed on figures; the source class names stay as they are in grades.csv
DISPLAY = {"Normal": "Normal", "DME": "DME",
           "Drusen": "AMD\nnon-neovascular", "AMD": "AMD\nneovascular"}
LAYERS = [("elm", "ELM", "#d6339f"), ("ez", "EZ", "#1f9fc4")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "font.family": "sans-serif"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}")


def classes_in(df):
    present = list(df["disease_class"].unique())
    return [c for c in CLASS_ORDER if c in present] + [c for c in present if c not in CLASS_ORDER]


def box_with_points(ax, groups, positions, color, width=0.32):
    bp = ax.boxplot(groups, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.35, edgecolor=color)
    rng = np.random.default_rng(0)
    for pos, g in zip(positions, groups):
        ax.scatter(pos + rng.uniform(-width / 3, width / 3, len(g)), g, s=10, color=color, alpha=0.8, zorder=3)


def fig_training_curves(run, out):
    logs = sorted(run.glob("fold*/log.csv"))
    if not logs:
        print("  no training logs found, skipping training curves")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, p in enumerate(logs):
        lg = pd.read_csv(p)
        col = plt.cm.tab10(i)
        axes[0].plot(lg.epoch, lg.train_loss, color=col, lw=1, ls="--")
        axes[0].plot(lg.epoch, lg.val_loss, color=col, lw=1.5, label=f"Fold {i + 1}")
        axes[1].plot(lg.epoch, lg.val_dice_mean, color=col, lw=1.5, label=f"Fold {i + 1}")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss (dashed = training, solid = validation)")
    axes[1].set(xlabel="Epoch", ylabel="Mean Dice (ELM, EZ)", title="Validation Dice", ylim=(0, 1))
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig1_training_curves")


def fig_dice_by_class(res, out):
    cls = classes_in(res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (key, name, _) in zip(axes, LAYERS):
        x = np.arange(len(cls))
        box_with_points(ax, [res.loc[res.disease_class == c, f"dice_{key}"] for c in cls], x - 0.2, "#7a7a7a")
        box_with_points(ax, [res.loc[res.disease_class == c, f"tol_dice_{key}"] for c in cls], x + 0.2, "#2b8a5e")
        ax.set_xticks(x, [f"{c}\n(n={(res.disease_class == c).sum()})" for c in cls])
        ax.set(title=name, ylim=(-0.02, 1.02))
    axes[0].set_ylabel("Score")
    axes[1].legend(handles=[Patch(color="#7a7a7a", alpha=0.5, label="Dice"),
                            Patch(color="#2b8a5e", alpha=0.5, label=f"Dice, {C.TOLERANCE_PX} px tolerance")],
                   fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, out, "fig2_dice_by_class")


def fig_coverage_by_class(res, out):
    cls = classes_in(res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (key, name, color) in zip(axes, LAYERS):
        x = np.arange(len(cls))
        box_with_points(ax, [res.loc[res.disease_class == c, f"cov_traced_{key}"] for c in cls], x - 0.2, "#7a7a7a")
        box_with_points(ax, [res.loc[res.disease_class == c, f"cov_pred_{key}"] for c in cls], x + 0.2, color)
        ax.set_xticks(x, cls)
        ax.set(title=f"{name} coverage", ylim=(-2, 104))
        ax.legend(handles=[Patch(color="#7a7a7a", alpha=0.5, label="Manual tracing"),
                           Patch(color=color, alpha=0.5, label="Model")], fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("Layer present across image width (%)")
    fig.tight_layout()
    save(fig, out, "fig3_coverage_by_class")


def fig_confusion(res, summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    per_class = []
    for ax, (key, name, _) in zip(axes[:2], LAYERS):
        cut = summary.get(key, {}).get("cutoff")
        g = res[res[f"{key}_grade"].isin(["Intact", "Disrupted", "Absent"])].copy()
        if cut is None or g.empty:
            ax.set_visible(False)
            continue
        truth = (g[f"{key}_grade"] != "Intact").astype(int)
        pred = (g[f"cov_pred_{key}"] < cut).astype(int)
        cm = np.array([[((truth == t) & (pred == p)).sum() for p in (0, 1)] for t in (0, 1)])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Intact", "Not intact"])
        ax.set_yticks([0, 1], ["Intact", "Not intact"])
        ax.set(xlabel="Model", ylabel="Grader", title=f"{name} (cut-off {cut:.1f}% coverage)")
        ax.spines[:].set_visible(False)
        g["correct"] = truth.values == pred.values
        for c, gc in g.groupby("disease_class"):
            per_class.append(dict(layer=name, cls=c, correct=int(gc.correct.sum()), wrong=int((~gc.correct).sum())))
    ax = axes[2]
    if per_class:
        pc = pd.DataFrame(per_class)
        cls = [c for c in CLASS_ORDER if c in pc.cls.values]
        x = np.arange(len(cls))
        for k, (name, off) in enumerate([("ELM", -0.2), ("EZ", 0.2)]):
            sub = pc[pc.layer == name].set_index("cls").reindex(cls).fillna(0)
            ax.bar(x + off, sub.correct, 0.38, color=LAYERS[k][2], alpha=0.8, label=f"{name} correct")
            ax.bar(x + off, sub.wrong, 0.38, bottom=sub.correct, color="#e0a030", alpha=0.9,
                   label="Misclassified" if k == 0 else None)
        ax.set_xticks(x, cls)
        ax.set(ylabel="Images", title="Grading result by disease class")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig4_integrity_confusion")


def fig_integrity_profile(df, run, out, image_name):
    rec = df[df.filename == image_name]
    if rec.empty:
        print(f"  {image_name} not found, skipping integrity profile")
        return
    rec = rec.iloc[0]
    pred_path = run / "oof_masks" / f"{rec['stem']}.png"
    if not pred_path.exists():
        print(f"  no prediction for {image_name}, skipping integrity profile")
        return
    img, gt, pred = read_gray(rec["image_path"]), read_mask(rec["mask_path"]), read_mask(pred_path)
    H, W = img.shape
    img_h = 8 * H / W
    fig = plt.figure(figsize=(9, img_h + 2.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[img_h, 1.9], hspace=0.12, left=0.14, right=0.93)
    ax = fig.add_subplot(gs[0])
    ax.imshow(overlay(img, pred), aspect="auto", extent=(0, W, H, 0))
    ax.set_title(f"{rec['disease_class']}: model prediction (pink = ELM, blue = EZ)", fontsize=10)
    ax.set_yticks([]); ax.spines[:].set_visible(False); ax.tick_params(labelbottom=False, bottom=False)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    rows = [("ELM, manual", gt, 1), ("ELM, model", pred, 1), ("EZ, manual", gt, 2), ("EZ, model", pred, 2)]
    for i, (label, m, c) in enumerate(rows):
        present = (m == c).any(axis=0).astype(float)
        color = LAYERS[0][2] if c == 1 else LAYERS[1][2]
        strip = np.zeros((1, W, 4))
        strip[0, present > 0] = matplotlib.colors.to_rgba(color)
        strip[0, present == 0] = (0.85, 0.85, 0.85, 1)
        ax2.imshow(strip, aspect="auto", extent=(0, W, len(rows) - i - 1, len(rows) - i))
        ax2.text(W * 1.01, len(rows) - i - 0.5, f"{column_coverage(m, c):.0f}%", va="center", fontsize=9)
    for yb in range(1, len(rows)):
        ax2.axhline(yb, color="white", lw=3 if yb == 2 else 1.5)
    ax2.set_yticks([len(rows) - i - 0.5 for i in range(len(rows))], [r[0] for r in rows])
    ax2.set(xlim=(0, W), ylim=(0, len(rows)), xlabel="Horizontal position (pixels)")
    ax2.spines[:].set_visible(False)
    ax2.set_title("Where each layer is present (grey = gap)", fontsize=10, loc="left")
    save(fig, out, f"fig5_integrity_profile_{rec['stem']}")


def fig_example_grid(df, res, run, out, picks=None):
    """picks: {source_class: filename} to override the median-Dice choice."""
    picks = picks or {}
    res = res.copy()
    res["dice_mean"] = res[["dice_elm", "dice_ez"]].mean(axis=1)
    cls = classes_in(res)
    fig, axes = plt.subplots(len(cls), 3, figsize=(11, 2.3 * len(cls)))
    axes = np.atleast_2d(axes)
    lookup = df.set_index("filename")
    for r, c in enumerate(cls):
        g = res[res.disease_class == c].sort_values("dice_mean")
        if c in picks:
            sel = g[g.filename == picks[c]]
            if sel.empty:
                raise SystemExit(f"--examples: {picks[c]} is not an image of class {c}")
            row = sel.iloc[0]
        else:
            row = g.iloc[len(g) // 2]
        rec = lookup.loc[row.filename]
        img = read_gray(rec["image_path"])
        panels = [np.repeat(img[..., None], 3, 2), overlay(img, read_mask(rec["mask_path"])),
                  overlay(img, read_mask(run / "oof_masks" / f"{rec['stem']}.png"))]
        for k, p in enumerate(panels):
            axes[r, k].imshow(p, aspect="auto")
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
            axes[r, k].spines[:].set_visible(False)
        axes[r, 0].set_ylabel(DISPLAY.get(c, c), fontsize=10)
        axes[r, 2].set_title(f"Dice ELM {row.dice_elm:.2f}, EZ {row.dice_ez:.2f}", fontsize=8, loc="right")
    for k, t in enumerate(["OCT B-scan", "Manual annotation", "Model prediction"]):
        axes[0, k].set_title(t + ("\n" if k == 2 else ""), fontsize=11)
    fig.legend(handles=[Patch(color=LAYERS[0][2], label="ELM"), Patch(color=LAYERS[1][2], label="EZ")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig6_examples_per_class")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="use the quick-test results")
    ap.add_argument("--image", default=None, help="file name for the integrity-profile figure")
    ap.add_argument("--name", default="cv", help="run to plot, e.g. cv or cv_pretrained")
    ap.add_argument("--examples", nargs="*", default=None, metavar="CLASS=FILE",
                    help="override the example shown for a class, e.g. Drusen=DRUSEN-1-1.jpeg")
    ap.add_argument("--out", default=None, metavar="FOLDER",
                    help="write into outputs/FOLDER instead of the default folder, so an "
                         "earlier set of figures is left untouched")
    args = ap.parse_args()
    run = C.run_dir(args.quick, args.name)
    ev = C.eval_dir(args.quick, args.name)
    if args.out:
        out = C.OUTPUT_DIR / args.out
    else:
        out = C.OUTPUT_DIR / ("figures_quick" if args.quick else
                              ("figures" if args.name == "cv" else f"figures_{args.name}"))
    out.mkdir(parents=True, exist_ok=True)
    if not (ev / "metrics_per_image.csv").exists():
        raise FileNotFoundError(f"{ev / 'metrics_per_image.csv'} not found. Run evaluate.py first.")
    res = pd.read_csv(ev / "metrics_per_image.csv")
    summary = json.loads((ev / "summary.json").read_text()) if (ev / "summary.json").exists() else {}
    df = load_table()

    print(f"Making figures in {out}")
    fig_training_curves(run, out)
    fig_dice_by_class(res, out)
    fig_coverage_by_class(res, out)
    fig_confusion(res, summary, out)
    name = args.image
    if name is None:  # default: the median AMD image, or the median diseased image
        dis = res[res.disease_class != "Normal"].copy()
        pick = dis[dis.disease_class == "AMD"] if (dis.disease_class == "AMD").any() else dis
        if not pick.empty:
            pick = pick.assign(d=pick[["dice_elm", "dice_ez"]].mean(axis=1)).sort_values("d")
            name = pick.iloc[len(pick) // 2].filename
    if name:
        fig_integrity_profile(df, run, out, name)
    picks = {}
    for item in (args.examples or []):
        if "=" not in item:
            raise SystemExit(f'--examples expects CLASS=FILE, got "{item}"')
        k, v = item.split("=", 1)
        picks[k.strip()] = v.strip()
    fig_example_grid(df, res, run, out, picks)
    print("Done.")


if __name__ == "__main__":
    main()
