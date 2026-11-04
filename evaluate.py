"""Evaluate the cross-validation predictions made by train_cv.py.

Run:   python evaluate.py          (or: python evaluate.py --quick)

Outputs (in outputs/evaluation):
  metrics_per_image.csv   Dice, tolerant Dice, boundary error and coverage per image
  summary.txt             the main numbers for the paper
  coverage_agreement.png  predicted vs traced coverage (ELM and EZ)
  bland_altman.png        agreement plot for coverage
  roc_integrity.png       how well predicted coverage separates Intact from Disrupted/Absent
  examples/               image | your tracing | model prediction, for a few images
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from utils import (load_table, read_gray, read_mask, dice, tolerant_f1, boundary_error, boundary_offset,
                   column_coverage, overlay, write_jpg)

LAYERS = {1: "elm", 2: "ez"}


def mean_sd(s):
    s = pd.Series(s).dropna()
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else (f"{s.mean():.3f}" if len(s) else "n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--name", default="cv", help="run to evaluate, e.g. cv, cv_pretrained, jhu_zeroshot")
    args = ap.parse_args()
    run = C.run_dir(args.quick, args.name)
    out = C.eval_dir(args.quick, args.name)
    (out / "examples").mkdir(parents=True, exist_ok=True)

    df = load_table()
    folds = pd.read_csv(run / "folds.csv")
    df = df.merge(folds[["filename", "fold"]], on="filename", how="inner")

    rows = []
    for r in df.to_dict("records"):
        pred_path = run / "oof_masks" / f"{r['stem']}.png"
        if not pred_path.exists():
            continue
        gt, pred = read_mask(r["mask_path"]), read_mask(pred_path)
        m = dict(filename=r["filename"], disease_class=r["disease_class"], fold=r["fold"],
                 elm_grade=r.get("elm_grade"), ez_grade=r.get("ez_grade"))
        for c, name in LAYERS.items():
            m[f"dice_{name}"] = dice(pred, gt, c)
            m[f"tol_dice_{name}"] = tolerant_f1(pred, gt, c)
            m[f"boundary_err_px_{name}"] = boundary_error(pred, gt, c)
            m[f"offset_px_{name}"] = boundary_offset(pred, gt, c)
            m[f"cov_traced_{name}"] = column_coverage(gt, c)
            m[f"cov_pred_{name}"] = column_coverage(pred, c)
        rows.append(m)
    if not rows:
        raise RuntimeError(f"No predictions found in {run / 'oof_masks'}. Run train_cv.py first.")
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_per_image.csv", index=False)

    lines, summary = [], {}
    lines.append(f"Images evaluated: {len(res)}  (cross-validation, each image predicted by a model that did not train on it)\n")
    lines.append("SEGMENTATION (mean ± SD over images)")
    for name in LAYERS.values():
        lines.append(f"  {name.upper():3s}  Dice {mean_sd(res[f'dice_{name}'])} | "
                     f"Dice with {C.TOLERANCE_PX}px tolerance {mean_sd(res[f'tol_dice_{name}'])} | "
                     f"boundary error {mean_sd(res[f'boundary_err_px_{name}'])} px | "
                     f"signed offset {mean_sd(res[f'offset_px_{name}'])} px (+ = model lower)")
    lines.append("\nDice by disease class")
    by = res.groupby("disease_class")[[f"dice_{n}" for n in LAYERS.values()]].agg(["mean", "std", "count"]).round(3)
    lines.append(by.to_string())
    lines.append("\nDice by fold (mean)")
    lines.append(res.groupby("fold")[[f"dice_{n}" for n in LAYERS.values()]].mean().round(3).to_string())

    # coverage agreement
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    lines.append("\nCOVERAGE AGREEMENT (percentage of image width where the layer is present)")
    for ax, ax2, name in zip(axes, axes2, LAYERS.values()):
        t, p = res[f"cov_traced_{name}"], res[f"cov_pred_{name}"]
        r_val, r_p = stats.pearsonr(t, p) if t.std() > 0 and p.std() > 0 else (np.nan, np.nan)
        rho, _ = stats.spearmanr(t, p) if t.std() > 0 and p.std() > 0 else (np.nan, np.nan)
        diff, avg = p - t, (p + t) / 2
        bias, loa = diff.mean(), 1.96 * diff.std(ddof=1)
        mae = diff.abs().mean()
        lines.append(f"  {name.upper():3s}  Pearson r {r_val:.3f} (p={r_p:.2g}) | Spearman {rho:.3f} | "
                     f"MAE {mae:.1f} pts | bias {bias:+.1f} | 95% limits {bias - loa:.1f} to {bias + loa:.1f}")
        summary[name] = dict(pearson_r=r_val, spearman=rho, mae=mae, bias=bias, loa_low=bias - loa, loa_high=bias + loa,
                             dice=res[f"dice_{name}"].mean(), tol_dice=res[f"tol_dice_{name}"].mean(),
                             boundary_err=res[f"boundary_err_px_{name}"].mean(),
                             offset=res[f"offset_px_{name}"].mean())
        for cls, g in res.groupby("disease_class"):
            ax.scatter(g[f"cov_traced_{name}"], g[f"cov_pred_{name}"], label=cls, alpha=0.8, s=28)
        ax.plot([0, 100], [0, 100], "k--", lw=1)
        ax.set(xlim=(0, 102), ylim=(0, 102), xlabel="Traced coverage (%)", ylabel="Predicted coverage (%)",
               title=f"{name.upper()}  (r = {r_val:.2f})")
        ax.legend(fontsize=8)
        ax2.scatter(avg, diff, s=24, alpha=0.8)
        for yv, st in [(bias, "-"), (bias + loa, "--"), (bias - loa, "--")]:
            ax2.axhline(yv, color="k", ls=st, lw=1)
        ax2.set(xlabel="Mean of traced and predicted coverage (%)", ylabel="Predicted − traced (%)",
                title=f"{name.upper()} Bland–Altman")
    fig.tight_layout(); fig.savefig(out / "coverage_agreement.png", dpi=200)
    fig2.tight_layout(); fig2.savefig(out / "bland_altman.png", dpi=200)
    plt.close("all")

    # integrity: does predicted coverage separate Intact from Disrupted/Absent?
    lines.append("\nINTEGRITY FROM PREDICTED COVERAGE (Intact vs Disrupted/Absent)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, LAYERS.values()):
        g = res[res[f"{name}_grade"].isin(["Intact", "Disrupted", "Absent"])]
        y = (g[f"{name}_grade"] != "Intact").astype(int)
        if y.nunique() < 2:
            lines.append(f"  {name.upper()}: needs both Intact and non-Intact images; skipped")
            ax.set_visible(False)
            continue
        score_pred = 100 - g[f"cov_pred_{name}"]
        score_traced = 100 - g[f"cov_traced_{name}"]
        auc_p, auc_t = roc_auc_score(y, score_pred), roc_auc_score(y, score_traced)
        fpr, tpr, thr = roc_curve(y, score_pred)
        j = np.argmax(tpr - fpr)
        cut = 100 - thr[j]
        lines.append(f"  {name.upper():3s}  AUC {auc_p:.3f} (from your traced lines: {auc_t:.3f}) | "
                     f"best cut-off: coverage below {cut:.1f}% = not intact | sensitivity {tpr[j]:.2f}, "
                     f"specificity {1 - fpr[j]:.2f} | n = {len(y)} ({y.sum()} not intact)")
        summary[name].update(auc_pred=auc_p, auc_traced=auc_t, cutoff=cut, n_not_intact=int(y.sum()), n=len(y))
        ax.plot(fpr, tpr, label=f"Predicted coverage (AUC {auc_p:.2f})")
        f2, t2, _ = roc_curve(y, score_traced)
        ax.plot(f2, t2, "--", label=f"Traced coverage (AUC {auc_t:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(xlabel="1 − specificity", ylabel="Sensitivity", title=f"{name.upper()} integrity")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc_integrity.png", dpi=200); plt.close(fig)
    lines.append("  Note: the cut-off is chosen on the same images, so treat it as optimistic.")

    # example figures: best, median and worst image per class
    res["dice_mean"] = res[["dice_elm", "dice_ez"]].mean(axis=1)
    picks = []
    for cls, g in res.groupby("disease_class"):
        g = g.sort_values("dice_mean")
        picks += [(g.iloc[0], "worst"), (g.iloc[len(g) // 2], "median"), (g.iloc[-1], "best")]
    lookup = df.set_index("filename")
    for row, tag in picks:
        rec = lookup.loc[row["filename"]]
        img = read_gray(rec["image_path"])
        gt = read_mask(rec["mask_path"])
        pred = read_mask(run / "oof_masks" / f"{rec['stem']}.png")
        panel = np.concatenate([np.repeat(img[..., None], 3, 2), overlay(img, gt), overlay(img, pred)], axis=0)
        write_jpg(out / "examples" / f"{row['disease_class']}_{tag}_{rec['stem']}.jpg", panel)

    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n\nExamples: top = image, middle = your tracing, bottom = model "
                                     "(pink = ELM, blue = EZ)\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(text)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
