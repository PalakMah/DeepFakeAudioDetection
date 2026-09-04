"""
Central configuration for the ASVspoof 2019 LA hybrid spoof-detection project.

All constants here were taken directly from the original Kaggle notebook.
The only change from the notebook is that hardcoded ``/kaggle/input/...`` and
``/kaggle/working/...`` paths have been replaced with project-relative,
environment-overridable paths so the project runs outside of Kaggle.
"""

import os
import random

import numpy as np
import torch

# ============================================================
# Reproducibility (Cell 4)
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root of the *extracted* ASVspoof2019 LA dataset. On Kaggle this was
# "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA" (Cells 5-7, 47).
# Override with the ASVSPOOF_DATA_ROOT environment variable, or place the
# dataset at data/raw/LA/LA (see data/README.md).
DATA_ROOT = os.environ.get(
    "ASVSPOOF_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "data", "raw", "LA", "LA"),
)

TRAIN_AUDIO_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev", "flac")
EVAL_AUDIO_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval", "flac")
PROTOCOL_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")

TRAIN_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL = os.path.join(PROTOCOL_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")

# Where extracted hybrid feature (.pt) files are cached. On Kaggle this was
# "/kaggle/working/hybrid_features" (Cell 33).
PROCESSED_DIR = os.environ.get(
    "ASVSPOOF_PROCESSED_DIR",
    os.path.join(PROJECT_ROOT, "data", "processed", "hybrid_features"),
)
TRAIN_FEATURES_DIR = os.path.join(PROCESSED_DIR, "train")
DEV_FEATURES_DIR = os.path.join(PROCESSED_DIR, "dev")
EVAL_FEATURES_DIR = os.path.join(PROCESSED_DIR, "eval")

# Where trained model checkpoints and evaluation artifacts are written. On
# Kaggle this was "/kaggle/working/" (Cells 51, 58, 59).
OUTPUTS_DIR = os.environ.get(
    "ASVSPOOF_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs")
)
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

GENERAL_MODEL_PATH = os.path.join(MODELS_DIR, "general_model.pt")
PERSONALIZER_MODEL_PATH = os.path.join(MODELS_DIR, "personalizer.pt")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")
CONFUSION_MATRIX_PATH = os.path.join(FIGURES_DIR, "confusion_matrices.png")

# ============================================================
# Audio settings (Cell 4, Cell 10)
# ============================================================
SR = 16000
DURATION = 4  # seconds
MAX_LEN = SR * DURATION

BATCH_SIZE = 16          # raw-audio DataLoader batch size (Cell 12)
NUM_WORKERS = 2          # raw-audio DataLoader workers (Cell 12)

FEATURE_EXTRACTION_BATCH_SIZE = 64   # batch size while caching features (Cell 33)
FEATURE_PRINT_EVERY = 100            # progress print interval (Cell 33)
FEATURE_CHECKPOINT_EVERY = 2500      # periodic checkpoint interval (Cell 33)

# ============================================================
# Handcrafted feature dimensions (Cells 14-19)
# ============================================================
N_MFCC = 80
N_CQCC = 80
N_LFCC = 80

# ============================================================
# XLS-R deep embedding (Cells 21-22, 26, 29)
# ============================================================
XLSR_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# ============================================================
# Hybrid feature layout (Cell 48)
# ============================================================
# mfcc(240) + cqcc(240) + lfcc(240) + spectral(13) + prosody(5) + physics(5)
# + xlsr(1024) = 1767
FEATURE_GROUPS = {
    "mfcc": (0, 240),
    "cqcc": (240, 480),
    "lfcc": (480, 720),
    "spectral": (720, 733),
    "prosody": (733, 738),
    "physics": (738, 743),
    "xlsr": (743, 1767),
}
TOTAL_FEATURE_DIM = 1767
PHYSICS_SLICE = FEATURE_GROUPS["physics"]

# ============================================================
# General detector training (Cell 50-51)
# ============================================================
GENERAL_EMBEDDING_DIM = 256
GENERAL_TRAIN_BATCH_SIZE = 64
GENERAL_EPOCHS = 12
GENERAL_LR = 1e-4
GENERAL_WEIGHT_DECAY = 1e-4
GENERAL_TRIPLET_MARGIN = 0.3
GENERAL_TRIPLET_WEIGHT = 0.5
GENERAL_GRAD_CLIP_NORM = 1.0
GENERAL_EARLY_STOP_PATIENCE = 5

# ============================================================
# Personalizer training (Cell 57-58)
# ============================================================
PERSONALIZER_EPOCHS = 6
PERSONALIZER_LR = 1e-4
PERSONALIZER_WEIGHT_DECAY = 1e-4
PERSONALIZER_CONTEXT_DROPOUT_P = 0.4
PERSONALIZER_MAX_K = 10
PERSONALIZER_GRAD_CLIP_NORM = 1.0
PERSONALIZER_BATCH_SIZE = 32
PERSONALIZER_REINDEX_EVERY = 25

# ============================================================
# Memory bank subsystem defaults (Cell 55)
# ============================================================
MEMORY_BOOTSTRAP_CONFIDENCE = 0.99
MEMORY_CONFIDENCE_THRESH = 0.98
MEMORY_PHYSICS_THRESH = 0.55
MEMORY_SIMILARITY_THRESH = 0.80
MEMORY_DECAY_RATE = 0.40
MEMORY_FLOOR_WEIGHT = 0.15
MEMORY_SECONDS_PER_YEAR = 1000.0  # kept as-is from the notebook (demo-scale aging)
MEMORY_PROTOTYPE_TRIGGER_SIZE = 30
MEMORY_PROTOTYPE_N = 20
MEMORY_PROTOTYPE_MINIBATCH_THRESHOLD = 2000
MEMORY_RETRIEVER_K_MIN = 3
MEMORY_RETRIEVER_K_MAX = 10

# ============================================================
# Deployment replay evaluation (Cell 59)
# ============================================================
BONAFIDE_LABEL = 1
SPOOF_LABEL = 0
