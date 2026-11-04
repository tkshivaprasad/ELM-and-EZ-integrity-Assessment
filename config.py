"""All settings for the ELM/EZ segmentation project. Edit values here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Data --------------------------------------------------------------
# Unzip the cleaned dataset (v2) into the "data" folder so you have:
#   data/images/*.jpg   data/masks/*.png   data/grades.csv
DATA_DIR = ROOT / "data1"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
GRADES_CSV = DATA_DIR / "grades.csv"

# ---- Outputs -----------------------------------------------------------
OUTPUT_DIR = ROOT / "outputs"

# ---- Classes in the masks ---------------------------------------------
CLASS_NAMES = {1: "ELM", 2: "EZ"}   # 0 = background
NUM_CLASSES = 3

# ---- Model -------------------------------------------------------------
ENCODER = "resnet34"          # ImageNet-pretrained backbone of the U-Net
ENCODER_WEIGHTS = "imagenet"  # set to None if the weight download fails

# ---- Training ----------------------------------------------------------
IMG_HEIGHT = 512              # every image is resized to this size for the model
IMG_WIDTH = 512
N_FOLDS = 5                   # 5-fold cross-validation
SEED = 42
EPOCHS = 60
BATCH_SIZE = 4                # lower to 2 if you run out of memory
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15                 # stop a fold early if validation Dice stops improving
CE_CLASS_WEIGHTS = [0.2, 1.0, 1.0]   # background, ELM, EZ (thin lines need more weight)
NUM_WORKERS = 0               # keep 0 on Windows

# ---- Evaluation --------------------------------------------------------
TOLERANCE_PX = 2              # a predicted pixel within 2 px of the traced line counts as correct

# ---- Quick test (python train_cv.py --quick) ---------------------------
QUICK = dict(N_FOLDS=2, EPOCHS=2, IMG_HEIGHT=256, IMG_WIDTH=256, PATIENCE=2)


# ---- JHU public dataset (pretraining) ------------------------------------
JHU_DIR = ROOT / "data_jhu"      # created by prepare_jhu.py
MASK_LINE_THICKNESS = 3          # same thickness as the grader app's masks
PRETRAIN_EPOCHS = 30
PRETRAIN_PATIENCE = 8


def run_dir(quick: bool = False, name: str = "cv") -> Path:
    """Folder where a training run is stored."""
    return OUTPUT_DIR / ("quick_test" if quick else name)


def eval_dir(quick: bool = False, name: str = "cv") -> Path:
    """Folder where evaluate.py writes results for a run."""
    if quick:
        return OUTPUT_DIR / "evaluation_quick"
    return OUTPUT_DIR / ("evaluation" if name == "cv" else f"evaluation_{name}")
