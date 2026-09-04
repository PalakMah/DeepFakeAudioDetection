"""
Stage 2 — Training.

Trains the general (speaker-agnostic) detector on the cached hybrid
features, then trains the cross-attention personalization head on top of
the frozen general model's embeddings. Saves both checkpoints to
outputs/models/.

Usage
-----
    python scripts/train.py [--skip-general] [--skip-personalizer]

Requires that ``scripts/prepare_data.py`` has already produced feature
caches for at least the train and dev splits.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.dataset import CachedFeatureDataset, find_pt_dir
from src.models.general_detector import GeneralDetector
from src.training.train_general import train_general_model
from src.training.train_personalizer import train_personalizer


def main():
    parser = argparse.ArgumentParser(description="Train the general detector and personalizer.")
    parser.add_argument("--skip-general", action="store_true", help="Skip general-detector training and load an existing checkpoint.")
    parser.add_argument("--skip-personalizer", action="store_true", help="Skip personalizer training.")
    args = parser.parse_args()

    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading cached feature datasets...")
    train_dataset = CachedFeatureDataset(find_pt_dir(config.TRAIN_FEATURES_DIR))
    val_dataset = CachedFeatureDataset(find_pt_dir(config.DEV_FEATURES_DIR))
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if args.skip_general:
        general_model = GeneralDetector(embedding_dim=config.GENERAL_EMBEDDING_DIM).to(config.DEVICE)
        general_model.load_state_dict(torch.load(config.GENERAL_MODEL_PATH, map_location=config.DEVICE))
        print(f"Loaded general model from {config.GENERAL_MODEL_PATH}")
    else:
        general_model, _general_history = train_general_model(train_dataset, val_dataset)
        torch.save(general_model.state_dict(), config.GENERAL_MODEL_PATH)
        print(f"Saved general model to {config.GENERAL_MODEL_PATH}")

    if not args.skip_personalizer:
        personalizer, _p_history = train_personalizer(train_dataset, general_model)
        torch.save(personalizer.state_dict(), config.PERSONALIZER_MODEL_PATH)
        print(f"Saved personalizer to {config.PERSONALIZER_MODEL_PATH}")


if __name__ == "__main__":
    main()
