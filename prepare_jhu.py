"""Convert the JHU OCT dataset (healthy controls and MS) into ELM/EZ masks
in the same format as your own masks (0 = background, 1 = ELM, 2 = EZ).

Step 1, look at the data first (nothing is written except a check image):
    python prepare_jhu.py --src "D:\\path\\to\\JHU" --inspect

Step 2, convert:
    python prepare_jhu.py --src "D:\\path\\to\\JHU"

The source folder must contain train/, val/ and test/, each with original/
and mask/ (annotation/ and overlay/ are optional), plus patients.csv.

How the layers are found: each mask is split into its separate lines (by
colour, or by grey value), and the lines are ordered from top to bottom.
  - 2 lines  -> the upper one is the ELM, the lower one is the EZ
  - 9 lines  -> the JHU boundaries; the 6th is the ELM, the 7th is the EZ (IS-OS)
  - otherwise, tell the script which line is which with --elm-rank and --ez-rank
    (1 = top line). The --inspect output shows the lines in order.
The lines are then redrawn at a fixed thickness so they match your masks.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as C
from utils import write_png, write_jpg, overlay, column_coverage

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------- files
def read_any(path):
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"Could not read {path}")
    return m


def to_gray(img):
    if img.ndim == 2:
        g = img
    elif img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g


def index_folder(folder: Path, suffix: str):
    """Map base name (e.g. hc10_bscan_01) -> file, stripping suffixes like _mask."""
    out = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.suffix.lower() in IMG_EXT or p.suffix.lower() == ".csv":
            stem = p.stem
            if suffix and stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
            out[stem] = p
    return out


def collect(src: Path):
    items = []
    for split in SPLITS:
        d = src / split
        orig = index_folder(d / "original", "")
        masks = index_folder(d / "mask", "_mask")
        if not masks:
            masks = index_folder(d / "masks", "_mask")
        ann = index_folder(d / "annotation", "_annotation")
        for stem, p in sorted(orig.items()):
            items.append(dict(split=split, stem=stem, patient=stem.split("_")[0], image=p,
                              mask=masks.get(stem), annotation=ann.get(stem)))
    return items


# ---------------------------------------------------------------- layer detection
def _merge_adjacent(bins, counts, min_count, circular):
    """Group neighbouring non-empty histogram bins into clusters."""
    keep = [b for b in bins if counts[b] >= min_count]
    clusters, cur = [], []
    for b in keep:
        if cur and b - cur[-1] > 1:
            clusters.append(cur); cur = []
        cur.append(b)
    if cur:
        clusters.append(cur)
    if circular and len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == max(bins):
        clusters[0] = clusters.pop() + clusters[0]
    return clusters


def find_lines(mask, min_frac=0.15):
    """Split a mask into separate lines. Returns list of dicts sorted top to bottom."""
    H, W = mask.shape[:2]
    labels = np.zeros((H, W), np.int32)   # 0 = none
    descs = []
    if mask.ndim == 3 and mask.shape[2] >= 3:
        bgr = np.ascontiguousarray(mask[..., :3])
        if bgr.dtype != np.uint8:
            bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fg = bgr.max(axis=2) > 60
        if mask.shape[2] == 4:
            fg &= mask[..., 3] > 0
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        colored = fg & (hsv[..., 1] >= 60)
        if colored.sum() >= (fg & ~colored).sum():
            hb = hsv[..., 0].astype(int) // 6          # 30 hue bins of 12 degrees
            counts = np.bincount(hb[colored], minlength=30)
            clusters = _merge_adjacent(list(range(30)), counts, max(3, int(0.01 * W)), circular=True)
            for k, cl in enumerate(clusters, 1):
                sel = colored & np.isin(hb, cl)
                labels[sel] = k
                mean_bgr = bgr[sel].mean(axis=0).round().astype(int)
                descs.append(f"colour BGR{tuple(int(v) for v in mean_bgr)}")
            return _finish(labels, descs, W, min_frac)
        value = bgr.max(axis=2)
        value[~fg] = 0
    else:
        value = mask if mask.ndim == 2 else mask[..., 0]
    value = value.astype(np.int64)
    nz = value > 0
    if not nz.any():
        return []
    vmax = int(value.max())
    counts = np.bincount(value[nz], minlength=vmax + 1)
    major = [v for v in range(1, vmax + 1) if counts[v] >= max(3, int(0.01 * W))]
    if len(major) <= 30 and counts[major].sum() >= 0.95 * nz.sum():
        clusters = [[v] for v in major]           # clean label values: one line per value
    else:                                         # noisy (e.g. JPEG): group neighbouring values
        clusters = _merge_adjacent(list(range(vmax + 1)), counts, max(3, int(0.01 * W)), circular=False)
    for k, cl in enumerate(clusters, 1):
        sel = nz & np.isin(value, cl)
        labels[sel] = k
        descs.append(f"value {cl[0]}" + (f"-{cl[-1]}" if len(cl) > 1 else ""))
    return _finish(labels, descs, W, min_frac)


def _finish(labels, descs, W, min_frac):
    rows = np.arange(labels.shape[0])[:, None]
    lines = []
    for k, d in enumerate(descs, 1):
        px = labels == k
        cols = px.any(axis=0)
        if cols.sum() < min_frac * W:
            continue
        n = px.sum(axis=0)
        mean_row = float((px * rows).sum() / px.sum())
        lines.append(dict(desc=d, pixels=px, mean_row=mean_row, count=int(px.sum()),
                          cover=float(cols.mean() * 100), thickness=float(n[cols].mean())))
    lines.sort(key=lambda L: L["mean_row"])
    return lines


def choose(lines, elm_rank, ez_rank):
    n = len(lines)
    if elm_rank and ez_rank:
        if max(elm_rank, ez_rank) > n:
            return None, f"only {n} lines found"
        return (lines[elm_rank - 1], lines[ez_rank - 1]), f"ranks {elm_rank}/{ez_rank} (set by you)"
    if n == 2:
        return (lines[0], lines[1]), "2 lines: upper = ELM, lower = EZ"
    if n == 9:
        return (lines[5], lines[6]), "9 boundaries: 6th = ELM, 7th = EZ (IS-OS)"
    return None, f"{n} lines found; use --elm-rank and --ez-rank"


# ---------------------------------------------------------------- redraw
def column_rows(px):
    n = px.sum(axis=0)
    rows = (px * np.arange(px.shape[0])[:, None]).sum(axis=0)
    return np.where(n > 0, rows / np.maximum(n, 1), np.nan)


def draw_line(canvas, rows, value, thickness, max_gap=3):
    """Draw a line column by column with a fixed vertical thickness, so it matches the
    masks made by the grader app. Gaps wider than max_gap columns are kept as gaps."""
    H, W = canvas.shape
    h = max(0, (thickness - 1) // 2)
    extra = thickness - 1 - 2 * h
    cols = np.where(~np.isnan(rows))[0]
    if len(cols) == 0:
        return
    for seg in np.split(cols, np.where(np.diff(cols) > max_gap + 1)[0] + 1):
        full = np.arange(seg[0], seg[-1] + 1)
        r = np.interp(full, seg, rows[seg])            # fill tiny gaps inside a piece
        ri = np.round(r).astype(int)
        for i, c in enumerate(full):
            lo, hi = ri[i], ri[i]
            if i > 0:                                   # connect steep steps between columns
                lo, hi = min(lo, (ri[i] + ri[i - 1]) // 2), max(hi, (ri[i] + ri[i - 1] + 1) // 2)
            y0, y1 = max(0, lo - h), min(H - 1, hi + h + extra)
            canvas[y0:y1 + 1, c] = value


def make_mask(shape, elm_px, ez_px, thickness):
    m = np.zeros(shape, np.uint8)
    draw_line(m, column_rows(elm_px), 1, thickness)
    draw_line(m, column_rows(ez_px), 2, thickness)    # EZ drawn last, wins any overlap
    return m


# ---------------------------------------------------------------- modes
def inspect(src, items, args):
    print(f"Source: {src}")
    pc = src / "patients.csv"
    if pc.exists():
        p = pd.read_csv(pc)
        print(f"patients.csv: {len(p)} rows; per split: {p.groupby('split').size().to_dict()}")
    for split in SPLITS:
        its = [i for i in items if i["split"] == split]
        print(f"  {split:5s}: {len(its)} images, {sum(i['mask'] is not None for i in its)} masks, "
              f"{sum(i['annotation'] is not None for i in its)} annotation files, "
              f"{len({i['patient'] for i in its})} patients")
    ex = next((i for i in items if i["mask"] is not None), None)
    if ex is None:
        print("No image/mask pairs found. Check the folder names (original/, mask/).")
        return
    img, m = read_any(ex["image"]), read_any(ex["mask"])
    print(f"\nExample: {ex['stem']}")
    print(f"  image {ex['image'].name}: shape {img.shape}, dtype {img.dtype}")
    print(f"  mask  {ex['mask'].name}: shape {m.shape}, dtype {m.dtype}")
    lines = find_lines(m)
    print(f"  lines found (top to bottom): {len(lines)}")
    for r, L in enumerate(lines, 1):
        print(f"   {r}. {L['desc']:28s} mean row {L['mean_row']:6.1f} | covers {L['cover']:5.1f}% of width "
              f"| ~{L['thickness']:.1f} px thick")
    pick, why = choose(lines, args.elm_rank, args.ez_rank)
    print(f"  choice: {why}")
    if ex["annotation"] is not None:
        try:
            a = pd.read_csv(ex["annotation"])
            print(f"\n  annotation {ex['annotation'].name}: {a.shape[0]} rows x {a.shape[1]} columns")
            print("  columns:", list(a.columns)[:15], "..." if a.shape[1] > 15 else "")
            print(a.head(3).to_string(max_cols=12))
        except Exception as e:
            print(f"  could not read annotation csv: {e}")
    # consistency over a sample
    rng = random.Random(0)
    sample = rng.sample([i for i in items if i["mask"] is not None], min(40, len(items)))
    counts = Counter(len(find_lines(read_any(i["mask"]))) for i in sample)
    print(f"\nLines per mask in {len(sample)} random masks: {dict(counts)}")
    # check image
    out = C.OUTPUT_DIR / "jhu_inspect"
    out.mkdir(parents=True, exist_ok=True)
    g = to_gray(img)
    vis = np.repeat(g[..., None], 3, 2).copy()
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
               (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
    for r, L in enumerate(lines, 1):
        if L["pixels"].shape == g.shape:
            vis[L["pixels"]] = palette[(r - 1) % len(palette)]
            ys, xs = np.nonzero(L["pixels"])
            x0 = int(xs.min()); y0 = int(ys[xs == xs.min()].mean())
            cv2.putText(vis, str(r), (x0 + 2, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        palette[(r - 1) % len(palette)], 1, cv2.LINE_AA)
    if pick and pick[0]["pixels"].shape == g.shape:
        conv = make_mask(g.shape, pick[0]["pixels"], pick[1]["pixels"], args.thickness)
        vis = np.concatenate([vis, overlay(g, conv)], axis=0)
    write_jpg(out / f"{ex['stem']}_lines.jpg", vis)
    print(f"\nSaved {out / (ex['stem'] + '_lines.jpg')}")
    print("  top: every line found, numbered from the top; bottom: the converted ELM (pink) and EZ (blue).")
    print("If that looks right, run the same command without --inspect.")


def convert(src, items, args):
    dst = Path(args.out)
    for sub in ("images", "masks", "qa"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    qa_done = set()
    for n, it in enumerate(items, 1):
        if it["mask"] is None:
            skipped["no mask"] += 1
            continue
        img, m = to_gray(read_any(it["image"])), read_any(it["mask"])
        if m.shape[:2] != img.shape:
            skipped["mask size differs from image"] += 1
            continue
        lines = find_lines(m)
        pick, why = choose(lines, args.elm_rank, args.ez_rank)
        if pick is None:
            skipped[why] += 1
            continue
        mask = make_mask(img.shape, pick[0]["pixels"], pick[1]["pixels"], args.thickness)
        gap = np.nanmedian(column_rows(pick[1]["pixels"]) - column_rows(pick[0]["pixels"]))
        write_png(dst / "images" / f"{it['stem']}.png", img)
        write_png(dst / "masks" / f"{it['stem']}.png", mask)
        if it["patient"] not in qa_done:
            qa_done.add(it["patient"])
            write_jpg(dst / "qa" / f"{it['split']}_{it['stem']}.jpg", overlay(img, mask))
        rows.append(dict(split=it["split"], patient=it["patient"], filename=f"{it['stem']}.png", stem=it["stem"],
                         width=img.shape[1], height=img.shape[0],
                         elm_coverage_pct=round(column_coverage(mask, 1), 1),
                         ez_coverage_pct=round(column_coverage(mask, 2), 1),
                         elm_ez_gap_px=round(float(gap), 1) if np.isfinite(gap) else np.nan))
        if n % 100 == 0:
            print(f"  {n}/{len(items)}")
    df = pd.DataFrame(rows)
    df.to_csv(dst / "index.csv", index=False)
    print(f"\nConverted {len(df)} of {len(items)} B-scans into {dst}")
    if skipped:
        print("Skipped:", dict(skipped))
    if len(df):
        print(df.groupby("split").agg(images=("stem", "size"), patients=("patient", "nunique")).to_string())
        bad = df[~(df.elm_ez_gap_px > 0)]
        print(f"Median ELM-to-EZ distance: {df.elm_ez_gap_px.median():.1f} px "
              f"({len(bad)} images where the ELM is not above the EZ; check these)")
        leak = set(df[df.split == "train"].patient) & set(df[df.split != "train"].patient)
        print("Patient overlap between splits:", sorted(leak) if leak else "none (good)")
        print(f"Check the overlays in {dst / 'qa'} (one per patient).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="JHU folder containing train/, val/, test/")
    ap.add_argument("--out", default=str(C.JHU_DIR))
    ap.add_argument("--inspect", action="store_true", help="only look at the data")
    ap.add_argument("--elm-rank", type=int, default=None, help="which line is the ELM (1 = top)")
    ap.add_argument("--ez-rank", type=int, default=None, help="which line is the EZ (1 = top)")
    ap.add_argument("--thickness", type=int, default=C.MASK_LINE_THICKNESS)
    args = ap.parse_args()
    src = Path(args.src)
    items = collect(src)
    if not items:
        raise FileNotFoundError(f"No images found under {src}/<train|val|test>/original")
    (inspect if args.inspect else convert)(src, items, args)


if __name__ == "__main__":
    main()
