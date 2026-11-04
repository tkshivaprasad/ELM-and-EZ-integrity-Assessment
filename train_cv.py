"""Train a U-Net to segment the ELM and EZ with k-fold cross-validation.

Run:   python train_cv.py            (full training)
       python train_cv.py --quick    (2-minute check that everything works)
       python train_cv.py --fold 3   (train or re-train one fold only)

Outputs (in outputs/cv):
  folds.csv                 which fold each image was validated in
  fold<k>/best.pt           best model of each fold
  fold<k>/log.csv           loss and Dice per epoch
  oof_masks/<image>.png     prediction for every image made by the model that
                            did NOT see it during training (used by evaluate.py)
"""
import argparse
import contextlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp

import config as C
from utils import load_table, read_gray, write_png, seed_everything
from model_utils import OCTLayerDataset, build_model, get_device, predict_probs


def batch_dice(logits, y, classes=(1, 2)):
    pred = logits.argmax(1)
    out = []
    for c in classes:
        p, g = pred == c, y == c
        inter = (p & g).sum(dim=(1, 2)).float()
        denom = (p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))).float()
        d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
        out.append(d)
    return torch.stack(out, dim=1)  # (batch, n_classes)


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    losses, dices = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        amp = torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
        with torch.set_grad_enabled(train), amp:
            logits = model(x)
            loss = loss_fn(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        dices.append(batch_dice(logits.detach(), y).cpu())
    d = torch.cat(dices).mean(0).numpy()
    return float(np.mean(losses)), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run to test the setup")
    ap.add_argument("--fold", type=int, default=None, help="train only this fold (0-based)")
    ap.add_argument("--init", default=None, help="start from these weights, e.g. outputs/jhu_pretrain/fold0/best.pt")
    ap.add_argument("--name", default="cv", help="name of this run (results go to outputs/<name>)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default from config.py)")
    args = ap.parse_args()

    cfg = dict(N_FOLDS=C.N_FOLDS, EPOCHS=C.EPOCHS, IMG_HEIGHT=C.IMG_HEIGHT,
               IMG_WIDTH=C.IMG_WIDTH, PATIENCE=C.PATIENCE)
    if args.quick:
        cfg.update(C.QUICK)
    out = C.run_dir(args.quick, args.name)
    lr = args.lr or C.LEARNING_RATE
    init_state = None
    if args.init:
        ckpt0 = torch.load(args.init, map_location="cpu")
        init_state = ckpt0["model"]
        if tuple(ckpt0.get("img_size", (cfg["IMG_HEIGHT"], cfg["IMG_WIDTH"]))) != (cfg["IMG_HEIGHT"], cfg["IMG_WIDTH"]):
            print(f"Note: pretrained at {ckpt0.get('img_size')}, fine-tuning at {(cfg['IMG_HEIGHT'], cfg['IMG_WIDTH'])}")
        print(f"Starting every fold from pretrained weights: {args.init}")
    (out / "oof_masks").mkdir(parents=True, exist_ok=True)

    seed_everything(C.SEED)
    device = get_device()
    df = load_table()
    print(f"Device: {device} | images: {len(df)} | classes: {df['disease_class'].value_counts().to_dict()}")
    if device.type == "cpu":
        print("No GPU found: training runs on the CPU and will be slow. Try --quick first.")

    skf = StratifiedKFold(n_splits=cfg["N_FOLDS"], shuffle=True, random_state=C.SEED)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, va) in enumerate(skf.split(df, df["disease_class"])):
        folds[va] = k
    df["fold"] = folds
    df[["filename", "disease_class", "fold"]].to_csv(out / "folds.csv", index=False)

    ce = nn.CrossEntropyLoss(weight=torch.tensor(C.CE_CLASS_WEIGHTS, dtype=torch.float32, device=device))
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2], from_logits=True)
    loss_fn = lambda logits, y: ce(logits, y) + dice_loss(logits, y)

    for k in range(cfg["N_FOLDS"]):
        if args.fold is not None and k != args.fold:
            continue
        fold_dir = out / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr_rows = df[df.fold != k].to_dict("records")
        va_rows = df[df.fold == k].to_dict("records")
        print(f"\n=== Fold {k + 1}/{cfg['N_FOLDS']}: {len(tr_rows)} train, {len(va_rows)} validation ===")

        h, w = cfg["IMG_HEIGHT"], cfg["IMG_WIDTH"]
        tr_ds = OCTLayerDataset(tr_rows, h, w, train=True, seed=C.SEED + k)
        va_ds = OCTLayerDataset(va_rows, h, w, train=False)
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                                            num_workers=C.NUM_WORKERS, drop_last=len(tr_ds) > C.BATCH_SIZE)
        va_dl = torch.utils.data.DataLoader(va_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                                            num_workers=C.NUM_WORKERS)

        if init_state is not None:
            model = build_model(encoder_weights=None)
            model.load_state_dict(init_state)
            model = model.to(device)
        else:
            model = build_model().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best, best_epoch, log = -1.0, 0, []
        for epoch in range(1, cfg["EPOCHS"] + 1):
            t0 = time.time()
            tr_loss, tr_d = run_epoch(model, tr_dl, device, loss_fn, opt, scaler)
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
                            "img_size": (h, w), "encoder": C.ENCODER}, fold_dir / "best.pt")
            print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f} | "
                  f"val Dice ELM {va_d[0]:.3f} EZ {va_d[1]:.3f} | {time.time() - t0:.0f}s{flag}")
            if epoch - best_epoch >= cfg["PATIENCE"]:
                print(f"No improvement for {cfg['PATIENCE']} epochs, stopping this fold.")
                break

        # out-of-fold predictions at the original image size
        ckpt = torch.load(fold_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for r in va_rows:
            probs = predict_probs([(model, (h, w))], read_gray(r["image_path"]), device)
            write_png(out / "oof_masks" / f"{r['stem']}.png", probs.argmax(0).astype(np.uint8))
        print(f"Fold {k + 1} best validation Dice {best:.3f} (epoch {best_epoch})")

    nxt = " --quick" if args.quick else (f" --name {args.name}" if args.name != "cv" else "")
    print(f"\nDone. Results are in {out}. Next: python evaluate.py{nxt}")


if __name__ == "__main__":
    main()
