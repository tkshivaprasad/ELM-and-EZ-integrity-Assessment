"""Run the JHU-pretrained model on your own images WITHOUT fine-tuning
(no retraining needed; uses outputs/jhu_pretrain/fold0/best.pt).

Run:   python zero_shot.py
Then:  python evaluate.py --name jhu_zeroshot
"""
import numpy as np

import config as C
from utils import load_table, read_gray, write_png
from model_utils import load_models, predict_probs, get_device


def main():
    device = get_device()
    models = load_models(C.OUTPUT_DIR / "jhu_pretrain", device)
    df = load_table()
    out = C.run_dir(False, "jhu_zeroshot")
    (out / "oof_masks").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(df.to_dict("records"), 1):
        pred = predict_probs(models, read_gray(r["image_path"]), device).argmax(0).astype(np.uint8)
        write_png(out / "oof_masks" / f"{r['stem']}.png", pred)
        if i % 20 == 0:
            print(f"  {i}/{len(df)}")
    df.assign(fold=0)[["filename", "disease_class", "fold"]].to_csv(out / "folds.csv", index=False)
    print(f"Predicted {len(df)} images with the JHU model (no fine-tuning) on {device}.")
    print("Next: python evaluate.py --name jhu_zeroshot")


if __name__ == "__main__":
    main()
