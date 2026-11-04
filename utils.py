"""Image loading, augmentation and metrics (NumPy/OpenCV only, no PyTorch)."""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as C

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- loading
def read_gray(path) -> np.ndarray:
    """Read an image as 8-bit grayscale. Works with non-English Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_mask(path) -> np.ndarray:
    """Read a label mask (0 background, 1 ELM, 2 EZ). Accepts old RGBA exports too."""
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def write_png(path, arr) -> None:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))


def write_jpg(path, arr_rgb, quality=92) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))


def load_table() -> pd.DataFrame:
    """Read grades.csv and keep rows whose image and mask both exist."""
    if not Path(C.GRADES_CSV).exists():
        raise FileNotFoundError(
            f"{C.GRADES_CSV} not found. Unzip the dataset into the data folder "
            "so that data/images, data/masks and data/grades.csv exist.")
    df = pd.read_csv(C.GRADES_CSV, encoding="utf-8-sig", sep=None, engine="python")
    # tolerate files re-saved by Excel: extra BOMs, spaces or different capitalisation
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    if "filename" not in df.columns:
        raise KeyError(
            f"{C.GRADES_CSV} has no 'filename' column. Columns found: {list(df.columns)[:8]}. "
            "Use the grades.csv exported by the grader app, and do not re-save it in Excel.")
    df["stem"] = df["filename"].str.replace(r"\.[^.]+$", "", regex=True)
    df["image_path"] = df["filename"].apply(lambda f: C.IMAGES_DIR / f)
    df["mask_path"] = df["stem"].apply(lambda s: C.MASKS_DIR / f"{s}.png")
    ok = df["image_path"].apply(Path.exists) & df["mask_path"].apply(Path.exists)
    missing = df.loc[~ok, "filename"].tolist()
    if missing:
        print(f"Skipping {len(missing)} rows without image or mask: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    df = df[ok].reset_index(drop=True)
    if "disease_class" not in df.columns:
        df["disease_class"] = df["stem"].str.split("_").str[0]
    if len(df) == 0:
        raise RuntimeError("No usable images found. Check the data folder.")
    return df


# ---------------------------------------------------------- preprocessing
def resize_pair(img, mask, h, w):
    img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_r = None if mask is None else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def normalize(img_gray: np.ndarray) -> np.ndarray:
    """Grayscale uint8 (H, W) -> float32 (3, H, W) with ImageNet normalization."""
    x = img_gray.astype(np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).copy()


def augment(img, mask, rng: random.Random):
    """Random flips, small rotations/shifts/zoom, brightness, contrast, gamma and noise."""
    h, w = img.shape
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.7:
        angle = rng.uniform(-10, 10)
        scale = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += rng.uniform(-0.05, 0.05) * w
        M[1, 2] += rng.uniform(-0.05, 0.05) * h
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = img.astype(np.float32)
    if rng.random() < 0.7:
        img = img * rng.uniform(0.75, 1.3) + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.3:
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


# ---------------------------------------------------------------- metrics
def dice(pred, gt, c):
    """Dice for class c. 1.0 when the layer is absent in both."""
    p, g = pred == c, gt == c
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p & g).sum() / s)


def tolerant_f1(pred, gt, c, tol=C.TOLERANCE_PX):
    """Dice-like score that accepts a small vertical/horizontal offset (tol pixels)."""
    p, g = (pred == c).astype(np.uint8), (gt == c).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gd, pd_ = cv2.dilate(g, k), cv2.dilate(p, k)
    precision = (p & gd).sum() / p.sum()
    recall = (g & pd_).sum() / g.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def column_coverage(mask, c):
    """Percentage of image columns in which layer c is present."""
    return float((mask == c).any(axis=0).mean() * 100)


def boundary_error(pred, gt, c):
    """Mean absolute vertical distance (pixels) between predicted and traced layer,
    over columns where both contain the layer. NaN if they never overlap."""
    p, g = pred == c, gt == c
    cols = p.any(axis=0) & g.any(axis=0)
    if not cols.any():
        return float("nan")
    rows = np.arange(pred.shape[0])[:, None]
    p_row = (p * rows).sum(axis=0)[cols] / p.sum(axis=0)[cols]
    g_row = (g * rows).sum(axis=0)[cols] / g.sum(axis=0)[cols]
    return float(np.abs(p_row - g_row).mean())


def boundary_offset(pred, gt, c):
    """Mean signed vertical offset (pixels) of the predicted layer relative to the traced one.
    Positive = prediction lies lower in the image. Reveals systematic shifts."""
    p, g = pred == c, gt == c
    cols = p.any(axis=0) & g.any(axis=0)
    if not cols.any():
        return float("nan")
    rows = np.arange(pred.shape[0])[:, None]
    p_row = (p * rows).sum(axis=0)[cols] / p.sum(axis=0)[cols]
    g_row = (g * rows).sum(axis=0)[cols] / g.sum(axis=0)[cols]
    return float((p_row - g_row).mean())


def overlay(img_gray, mask):
    """Colour overlay: pink = ELM, blue = EZ."""
    rgb = np.repeat(img_gray[..., None], 3, axis=2).copy()
    rgb[mask == 1] = (255, 60, 200)
    rgb[mask == 2] = (40, 230, 255)
    return rgb


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
