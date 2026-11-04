"""Test whether the differences between two runs are statistically significant.

Run:   python stats_compare.py                          (baseline vs JHU pretrained + fine-tuned)
       python stats_compare.py --a cv --b jhu_zeroshot  (any two runs)

Both runs are evaluated on the same images, so the tests are PAIRED:
  - Wilcoxon signed-rank test per metric (Dice, tolerant Dice, boundary error,
    coverage error), for all images and for the diseased images only
  - paired bootstrap (2,000 resamples) for the difference in integrity AUC,
    with a 95% confidence interval
Results: outputs/stats_<a>_vs_<b>.csv and .txt
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import config as C

METRICS = [("dice", "Dice", "higher"), ("tol_dice", "Tolerant Dice", "higher"),
           ("boundary_err_px", "Boundary error (px)", "lower"), ("cov_abs_err", "Coverage error (points)", "lower")]
LABELS = {"cv": "Baseline", "cv_pretrained": "JHU + fine-tuned", "jhu_zeroshot": "JHU only"}


def load(name):
    f = C.eval_dir(False, name) / "metrics_per_image.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run: python evaluate.py --name {name}")
    d = pd.read_csv(f)
    for layer in ("elm", "ez"):
        d[f"cov_abs_err_{layer}"] = (d[f"cov_pred_{layer}"] - d[f"cov_traced_{layer}"]).abs()
    return d


def paired_test(x, y):
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.allclose(x.values, y.values):
        return len(x), np.nan
    return len(x), float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def auc_diff(da, db, layer, n_boot=2000, seed=0):
    g = da[[f"{layer}_grade", f"cov_pred_{layer}"]].join(db[[f"cov_pred_{layer}"]], rsuffix="_b")
    g = g[g[f"{layer}_grade"].isin(["Intact", "Disrupted", "Absent"])]
    y = (g[f"{layer}_grade"] != "Intact").astype(int).values
    if len(np.unique(y)) < 2:
        return None
    sa, sb = 100 - g[f"cov_pred_{layer}"].values, 100 - g[f"cov_pred_{layer}_b"].values
    auc_a, auc_b = roc_auc_score(y, sa), roc_auc_score(y, sb)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], sb[i]) - roc_auc_score(y[i], sa[i]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return auc_a, auc_b, auc_b - auc_a, lo, hi, p


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cv", help="first run (reference)")
    ap.add_argument("--b", default="cv_pretrained", help="second run")
    args = ap.parse_args()
    la, lb = LABELS.get(args.a, args.a), LABELS.get(args.b, args.b)
    A, B = load(args.a).set_index("filename"), load(args.b).set_index("filename")
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    diseased = A.disease_class != "Normal"

    rows, lines = [], [f"PAIRED COMPARISON: {la} (A) vs {lb} (B), {len(common)} images "
                       f"({int(diseased.sum())} diseased)", ""]
    lines.append(f"{'Layer':5s} {'Metric':24s} {'Subset':9s} {'A mean':>8s} {'B mean':>8s} {'B - A':>8s} "
                 f"{'better':>7s} {'p (Wilcoxon)':>13s}")
    for layer in ("elm", "ez"):
        for key, label, good in METRICS:
            for subset, mask in (("all", slice(None)), ("diseased", diseased)):
                a, b = A.loc[mask, f"{key}_{layer}"], B.loc[mask, f"{key}_{layer}"]
                n, p = paired_test(a, b)
                d = b.mean() - a.mean()
                better = "-" if not np.isfinite(d) or abs(d) < 1e-9 else (
                    "B" if (d > 0) == (good == "higher") else "A")
                rows.append(dict(layer=layer.upper(), metric=label, subset=subset, n=n, mean_a=a.mean(),
                                 mean_b=b.mean(), diff_b_minus_a=d, better=better, p_value=p))
                lines.append(f"{layer.upper():5s} {label:24s} {subset:9s} {a.mean():8.3f} {b.mean():8.3f} "
                             f"{d:+8.3f} {better:>7s} {fmt_p(p):>13s}")
    lines += ["", "INTEGRITY AUC (paired bootstrap, 2,000 resamples)"]
    for layer in ("elm", "ez"):
        r = auc_diff(A, B, layer)
        if r is None:
            continue
        auc_a, auc_b, d, lo, hi, p = r
        rows.append(dict(layer=layer.upper(), metric="Integrity AUC", subset="all", n=len(common), mean_a=auc_a,
                         mean_b=auc_b, diff_b_minus_a=d, better="B" if d > 0 else "A", p_value=p,
                         ci_low=lo, ci_high=hi))
        lines.append(f"  {layer.upper():3s} A {auc_a:.3f} | B {auc_b:.3f} | difference {d:+.3f} "
                     f"(95% CI {lo:+.3f} to {hi:+.3f}) | p {fmt_p(p)}")
    lines += ["", "How to read: p < 0.05 means the difference is unlikely to be due to chance.",
              "If the 95% CI of the AUC difference includes 0, the AUCs are not significantly different.",
              "Many metrics are tested here; treat single p-values just under 0.05 with caution."]
    text = "\n".join(lines)
    stem = C.OUTPUT_DIR / f"stats_{args.a}_vs_{args.b}"
    pd.DataFrame(rows).round(4).to_csv(f"{stem}.csv", index=False)
    stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nSaved {stem}.csv and {stem}.txt")


if __name__ == "__main__":
    main()
