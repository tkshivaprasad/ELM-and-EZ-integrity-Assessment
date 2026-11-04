# ELM and EZ segmentation (U-Net, 5-fold cross-validation)

This project trains a U-Net to trace the ELM and EZ on OCT B-scans, using the
masks you made with the OCT Integrity Grader. From the predicted lines it
measures coverage (how much of the image width each layer covers) and tests how
well that coverage separates Intact from Disrupted/Absent layers.

```
oct_elm_ez_segmentation/
  config.py        all settings (paths, image size, epochs, batch size)
  train_cv.py      trains the U-Net with 5-fold cross-validation
  evaluate.py      Dice, boundary error, coverage agreement, ROC, example figures
  predict.py       runs the trained models on new images
  make_figures.py  publication-quality figures from the results
  prepare_jhu.py   converts the JHU public dataset into ELM/EZ masks
  pretrain_jhu.py  pretrains the U-Net on JHU and tests it on JHU test patients
  compare_runs.py  puts the results of different runs side by side
  utils.py         image loading, augmentation, metrics
  model_utils.py   dataset, model and prediction code
  data/            put the dataset here
  outputs/         created automatically; all results go here
```

## Setup in VS Code (Windows)

**1. Install Python 3.11 from python.org.** During setup tick "Add python.exe to
PATH" and, on the last screen, click "Disable path length limit" if it appears.
(The Microsoft Store Python caused the long-path error earlier, so avoid it.)

**2. Put this folder at a short path**, for example `D:\octseg`, and open it in
VS Code with File > Open Folder. Install the "Python" extension by Microsoft if
VS Code suggests it.

**3. Create a virtual environment.** Open Terminal > New Terminal and run:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the line. If PowerShell says scripts
are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, type Y,
and activate again.

**4. Install PyTorch.** First check whether you have an NVIDIA graphics card:

```
nvidia-smi
```

- If it prints a table, you have an NVIDIA GPU. Go to https://pytorch.org/get-started/locally/,
  choose Stable, Windows, Pip, Python and the newest CUDA version, and run the
  command it shows.
- If it says the command is not recognized, install the CPU version:
  `pip install torch torchvision`

**5. Install the other packages:**

```
pip install -r requirements.txt
```

**6. Add the data.** Unzip `oct_elm_ez_dataset_cleaned_v2_2026-09-29.zip` into the
`data` folder so you have `data\images`, `data\masks` and `data\grades.csv`.

**7. In VS Code, choose the interpreter:** press Ctrl+Shift+P, type "Python:
Select Interpreter" and pick the one in `.venv`.

## Running

You can use the terminal commands below, or the Run and Debug panel
(Ctrl+Shift+D), which has ready-made entries numbered 1 to 5.

| Step | Command | What it does |
|---|---|---|
| Check setup | `python train_cv.py --quick` | 2 folds, 2 epochs, small images. Confirms everything works. Results are not meaningful. |
| | `python evaluate.py --quick` | Checks the evaluation step. |
| Train | `python train_cv.py` | Full 5-fold training |
| Evaluate | `python evaluate.py` | Makes the numbers and figures for the paper |
| Figures | `python make_figures.py` (add `--name cv_pretrained` for the fine-tuned run) | Publication figures (PNG 300 dpi + PDF) in outputs/figures |
| Predict | `python predict.py --input "D:\path\to\images"` | Runs the models on new images |

The first run downloads the ImageNet weights for the encoder (about 85 MB), so
you need internet the first time.

### How long training takes

- **NVIDIA GPU:** roughly 20 to 60 minutes for all 5 folds.
- **CPU only:** many hours. To make it manageable, in `config.py` set
  `IMG_HEIGHT = IMG_WIDTH = 384` and `EPOCHS = 40`, and train one fold at a time:
  `python train_cv.py --fold 0`, then `--fold 1`, and so on up to `--fold 4`.
  Run `evaluate.py` after all five are done.

## Results

`outputs/cv/`
- `fold<k>/best.pt`: the trained model for each fold
- `fold<k>/log.csv`: loss and validation Dice for every epoch
- `oof_masks/`: a prediction for every image, made by the model that did not
  train on it (this is what makes the evaluation fair)

`outputs/evaluation/`
- `summary.txt`: the main numbers
- `metrics_per_image.csv`: all metrics for every image
- `coverage_agreement.png`, `bland_altman.png`, `roc_integrity.png`: figures
- `examples/`: image, your tracing and the model's prediction, stacked
  (pink = ELM, blue = EZ), for the best, median and worst image of each class

`outputs/predictions/` (from predict.py)
- `masks/`, `overlays/`, and `predictions.csv` with coverage and an uncertainty
  score. The most uncertain images are listed first; those are the most useful
  ones to annotate next.

## Using the JHU dataset (pretraining)

The JHU dataset (healthy controls and MS, 35 patients, split by patient into
train/val/test) teaches the model where the ELM and EZ are before it learns
from your own images.

| Step | Command |
|---|---|
| 1. Look at the data | `python prepare_jhu.py --src "D:\path\to\JHU" --inspect` |
| 2. Convert it | `python prepare_jhu.py --src "D:\path\to\JHU"` |
| 3. Pretrain and test on JHU test patients | `python pretrain_jhu.py` |
| 4. Evaluate the JHU model on your images, no fine-tuning | `python evaluate.py --name jhu_zeroshot` |
| 5. Fine-tune on your images (5-fold) | `python train_cv.py --init outputs/jhu_pretrain/fold0/best.pt --name cv_pretrained` |
| 6. Evaluate the fine-tuned model | `python evaluate.py --name cv_pretrained` |
| 7. Re-run the baseline evaluation (adds new columns) | `python evaluate.py` |
| 8. Compare all three | `python compare_runs.py` |

Step 1 prints how many lines each mask contains and which ones it will use as
ELM and EZ, and saves a check image in `outputs/jhu_inspect/`. Look at it before
converting. After step 2, check the overlays in `data_jhu/qa/` (one per patient).

Pretraining (step 3) takes roughly 30 to 60 minutes on an RTX 3070. If memory is
short, use `python pretrain_jhu.py --every 2`.

The "offset" in the results is the average vertical shift between the model's
line and yours (positive = the model's line is lower). A consistent offset in the
zero-shot results means JHU marks the layers at a slightly different position
than you do (for example the edge of a band rather than its centre).
Fine-tuning corrects this.

Cite the JHU dataset in the paper and follow its terms of use.

## Understanding the metrics

- **Dice:** pixel overlap between the model's line and your line (0 to 1). Thin
  lines make this strict: a line shifted by one pixel can drop Dice to about
  0.65 even though it is visually correct.
- **Dice with 2 px tolerance:** counts a pixel as correct if it is within 2
  pixels of the other line. Report both; this one reflects visual quality better.
- **Boundary error:** average vertical distance in pixels between the model's
  line and yours, where both exist.
- **Coverage agreement:** correlation, mean absolute error and Bland-Altman
  limits between predicted and traced coverage.
- **Integrity AUC:** how well predicted coverage separates Intact from
  Disrupted/Absent. The cut-off is chosen on the same images, so it is optimistic;
  confirm it on new images later.

## Troubleshooting

- **Weight download fails:** set `ENCODER_WEIGHTS = None` in `config.py`
  (results will be worse), or check your internet connection or proxy.
- **Out of memory:** set `BATCH_SIZE = 2` (or 1) in `config.py`.
- **"No usable images found":** the `data` folder is not set up as in step 6.
- **Training is very slow:** see "How long training takes" above.

## Limitations to state in the paper

- Patient identity is unknown, so folds are split by image, not by patient.
- Image size and quality differ between disease classes.
- Almost all diseased images have disrupted layers and all normals are intact, so
  disease and integrity are confounded in this pilot dataset.
