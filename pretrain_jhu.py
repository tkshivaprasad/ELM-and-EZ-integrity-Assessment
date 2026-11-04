"""Pretrain the U-Net on the JHU dataset (converted by prepare_jhu.py), test it on
the JHU test patients, and try it directly on your own 85 images.

Run:   python pretrain_jhu.py
       python pretrain_jhu.py --every 2      (use every 2nd B-scan: faster, less memory)

Outputs:
  outputs/jhu_pretrain/fold0/best.pt       pretrained model (use it with train_cv.py --init)
  outputs/jhu_pretrain/fold0/log.csv       training log
  outputs/jhu_pretrain/test_metrics.csv    results for every JHU test B-scan
  outputs/jhu_pretrain/test_summary.txt    results on the JHU test patients (patient-level split)
  outputs/jhu_zeroshot/                    predictions on your own images without any
                                           fine-tuning; evaluate with: python evaluate.py --name jhu_zeroshot
"""
import argparse
import contextlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config as C
from utils import (read_gray, read_mask, write_png, seed_everything, dice, tolerant_f1,
                   boundary_error, boundary_offset, column_coverage, load_table)
from model_utils import OCTLayerDataset, build_model, get_device, predict_probs
from train_cv import run_epoch


def rows_for(index, split, every):
    d = index[index.split == split].sort_values("stem")
    if every > 1:
        d = d[d.groupby("patient").cumcount() % every == 0]
    return [dict(r, image_path=C.JHU_DIR / "images" / r["filename"], mask_path=C.JHU_DIR / "masks" / r["filename"])
            for r in d.to_dict("records")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1, help="use every n-th B-scan per patient for training")
    ap.add_argument("--epochs", type=int, default=C.PRETRAIN_EPOCHS)
    ap.add_argument("--no-zero-shot", action="store_true", help="skip predicting your own images")
    args = ap.parse_args()

    idx_path = C.JHU_DIR / "index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} not found. Run prepare_jhu.py first.")
    index = pd.read_csv(idx_path)
    seed_everything(C.SEED)
    device = get_device()
    tr, va, te = rows_for(index, "train", args.every), rows_for(index, "val", 1), rows_for(index, "test", 1)
    print(f"Device: {device} | JHU B-scans: train {len(tr)}, val {len(va)}, test {len(te)} | "
          f"patients: train {len({r['patient'] for r in tr})}, val {len({r['patient'] for r in va})}, "
          f"test {len({r['patient'] for r in te})}")

    out = C.OUTPUT_DIR / "jhu_pretrain"
    fold_dir = out / "fold0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    h, w = C.IMG_HEIGHT, C.IMG_WIDTH
    print("Loading images...")
    tr_ds = OCTLayerDataset(tr, h, w, train=True, seed=C.SEED)
    va_ds = OCTLayerDataset(va, h, w, train=False)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                                        num_workers=C.NUM_WORKERS, drop_last=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    ce = nn.CrossEntropyLoss(weight=torch.tensor(C.CE_CLASS_WEIGHTS, dtype=torch.float32, device=device))
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2], from_logits=True)
    loss_fn = lambda logits, y: ce(logits, y) + dice_loss(logits, y)

    best, best_epoch, log = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, tr_dl, device, loss_fn, opt, scaler)
        va_loss, va_d = run_epoch(model, va_dl, device, loss_fn)
        sched.step()
        score = float(va_d.mean())
        log.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                        val_dice_elm=va_d[0], val_dice_ez=va_d[1], val_dice_mean=score))
        pd.DataFrame(log).to_csv(fold_dir / "log.csv", index=False)
        flag = ""
        if score > best:
            best, best_epoch, flag = score, epoch, "  <- best"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": score,
                        "img_size": (h, w), "encoder": C.ENCODER, "source": "JHU"}, fold_dir / "best.pt")
        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f} | "
              f"val Dice ELM {va_d[0]:.3f} EZ {va_d[1]:.3f} | {time.time() - t0:.0f}s{flag}")
        if epoch - best_epoch >= C.PRETRAIN_PATIENCE:
            print(f"No improvement for {C.PRETRAIN_PATIENCE} epochs, stopping.")
            break

    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device)["model"])
    model.eval()
    ens = [(model, (h, w))]

    # ---- test on JHU test patients
    print("\nTesting on the JHU test patients...")
    res = []
    for r in te:
        gt = read_mask(r["mask_path"])
        pred = predict_probs(ens, read_gray(r["image_path"]), device).argmax(0).astype(np.uint8)
        m = dict(patient=r["patient"], filename=r["filename"])
        for c, n in ((1, "elm"), (2, "ez")):
            m[f"dice_{n}"] = dice(pred, gt, c)
            m[f"tol_dice_{n}"] = tolerant_f1(pred, gt, c)
            m[f"boundary_err_px_{n}"] = boundary_error(pred, gt, c)
            m[f"offset_px_{n}"] = boundary_offset(pred, gt, c)
            m[f"cov_traced_{n}"] = column_coverage(gt, c)
            m[f"cov_pred_{n}"] = column_coverage(pred, c)
        res.append(m)
    res = pd.DataFrame(res)
    res.to_csv(out / "test_metrics.csv", index=False)
    cols = [f"{k}_{n}" for n in ("elm", "ez") for k in ("dice", "tol_dice", "boundary_err_px")]
    per_pat = res.groupby("patient")[cols].mean()
    lines = [f"JHU pretraining: best validation Dice {best:.3f} (epoch {best_epoch})",
             f"Test set: {len(res)} B-scans from {res.patient.nunique()} patients not used in training",
             "", "Mean ± SD over test patients (each patient averaged first):"]
    for n in ("elm", "ez"):
        seg = lambda k: f"{per_pat[f'{k}_{n}'].mean():.3f} ± {per_pat[f'{k}_{n}'].std(ddof=1):.3f}"
        lines.append(f"  {n.upper():3s} Dice {seg('dice')} | tolerant Dice {seg('tol_dice')} | "
                     f"boundary error {seg('boundary_err_px')} px")
    lines += ["", "Per patient:", per_pat.round(3).to_string()]
    text = "\n".join(lines)
    (out / "test_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)

    # ---- zero-shot on the user's own images
    if not args.no_zero_shot:
        try:
            df = load_table()
        except Exception as e:
            print(f"\nSkipping zero-shot test on your images: {e}")
            return
        zs = C.run_dir(False, "jhu_zeroshot")
        (zs / "oof_masks").mkdir(parents=True, exist_ok=True)
        for r in df.to_dict("records"):
            pred = predict_probs(ens, read_gray(r["image_path"]), device).argmax(0).astype(np.uint8)
            write_png(zs / "oof_masks" / f"{r['stem']}.png", pred)
        df.assign(fold=0)[["filename", "disease_class", "fold"]].to_csv(zs / "folds.csv", index=False)
        print(f"\nPredicted your {len(df)} images with the JHU model (no fine-tuning).")
        print("Next: python evaluate.py --name jhu_zeroshot")
    print("Then fine-tune: python train_cv.py --init outputs/jhu_pretrain/fold0/best.pt --name cv_pretrained")


if __name__ == "__main__":
    main()
