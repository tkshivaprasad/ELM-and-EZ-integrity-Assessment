"""PyTorch parts: dataset, model and prediction."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

import config as C
from utils import read_gray, read_mask, resize_pair, normalize, augment


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OCTLayerDataset(Dataset):
    """Loads all images and masks once (the dataset is small), resized to the model size."""

    def __init__(self, rows, height, width, train: bool, seed: int = 0):
        self.train = train
        self.rng = random.Random(seed)
        self.items = []
        for r in rows:
            img, mask = resize_pair(read_gray(r["image_path"]), read_mask(r["mask_path"]), height, width)
            self.items.append((img, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]
        if self.train:
            img, mask = augment(img, mask, self.rng)
        x = torch.from_numpy(normalize(img))
        y = torch.from_numpy(mask.astype(np.int64))
        return x, y


def build_model(encoder_weights=C.ENCODER_WEIGHTS):
    return smp.Unet(encoder_name=C.ENCODER, encoder_weights=encoder_weights,
                    in_channels=3, classes=C.NUM_CLASSES)


def load_models(run_dir: Path, device):
    """Load every fold's best model from a training run (used as an ensemble)."""
    paths = sorted(Path(run_dir).glob("fold*/best.pt"))
    if not paths:
        raise FileNotFoundError(f"No trained models in {run_dir}. Run train_cv.py first.")
    models = []
    for p in paths:
        ckpt = torch.load(p, map_location=device)
        m = build_model(encoder_weights=None)
        m.load_state_dict(ckpt["model"])
        m.to(device).eval()
        models.append((m, ckpt.get("img_size", (C.IMG_HEIGHT, C.IMG_WIDTH))))
    return models


@torch.no_grad()
def predict_probs(models, img_gray, device, tta: bool = True) -> np.ndarray:
    """Average softmax probabilities of the models (and a flipped copy) at the
    original image size. Returns an array of shape (classes, H, W)."""
    H, W = img_gray.shape
    total = np.zeros((C.NUM_CLASSES, H, W), dtype=np.float32)
    n = 0
    for model, (h, w) in models:
        img_r, _ = resize_pair(img_gray, None, h, w)
        x = torch.from_numpy(normalize(img_r))[None].to(device)
        prob = torch.softmax(model(x), dim=1)
        if tta:
            prob_f = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
            prob = (prob + torch.flip(prob_f, dims=[3])) / 2
        prob = prob[0].float().cpu().numpy()
        for c in range(C.NUM_CLASSES):
            total[c] += cv2.resize(prob[c], (W, H), interpolation=cv2.INTER_LINEAR)
        n += 1
    return total / n
