"""
Stage 1 — Data preparation.

Loads the official ASVspoof2019 LA protocol splits, builds the raw-audio
datasets (with training-time augmentation), and extracts + caches the
1767-d hybrid feature vector (MFCC+CQCC+LFCC+spectral+prosody+physics+XLS-R)
for every clip in every split.

Usage
-----
    python scripts/prepare_data.py [--splits train dev eval]

Before running, place the ASVspoof2019 LA dataset under ``data/raw/LA/LA``
(or point ``ASVSPOOF_DATA_ROOT`` at it) — see data/README.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.augmentation import train_augment
from src.data.dataset import ASVSpoofDataset
from src.data.protocol import load_official_splits, print_split_summary
from src.features.extraction_pipeline import process_split
from src.features.xlsr import load_xlsr_model


def main():
    parser = argparse.ArgumentParser(description="Extract and cache hybrid features.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "eval"],
        choices=["train", "dev", "eval"],
        help="Which splits to process.",
    )
    args = parser.parse_args()

    print("Loading official protocol splits...")
    train_df, dev_df, eval_df = load_official_splits()
    print_split_summary(train_df, dev_df, eval_df)

    datasets = {
        "train": ASVSpoofDataset(train_df, augment=train_augment),
        "dev": ASVSpoofDataset(dev_df, augment=None),
        "eval": ASVSpoofDataset(eval_df, augment=None),
    }

    print(f"\nLoading XLS-R model ({config.XLSR_MODEL_NAME})...")
    feature_extractor, xlsr_model = load_xlsr_model()

    for split in args.splits:
        process_split(
            datasets[split],
            split,
            feature_extractor,
            xlsr_model,
            save_dir=getattr(config, f"{split.upper()}_FEATURES_DIR"),
        )


if __name__ == "__main__":
    main()
