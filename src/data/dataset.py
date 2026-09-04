"""
Dataset classes.

``ASVSpoofDataset`` loads raw waveforms and applies padding/cropping and
(optionally) augmentation. It corresponds to notebook Cell 10, which is the
final, refined version of the class (Cell 9 held an earlier draft of the same
class that was superseded before the rest of the notebook ran — see the
project mapping notes in the README for details).

``CachedFeatureDataset`` loads pre-extracted hybrid feature vectors (``.pt``
files produced by ``src/features``) from disk. It corresponds to notebook
Cell 48.
"""

import glob
import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


class ASVSpoofDataset(Dataset):
    """Loads raw audio for a protocol DataFrame, pads/crops to a fixed
    length, and (optionally) applies waveform augmentation."""

    def __init__(self, dataframe, augment=None):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found:\n{path}")

        audio, _ = librosa.load(path, sr=config.SR, mono=True)

        # -------------------------
        # Crop or Pad
        # -------------------------
        if len(audio) > config.MAX_LEN:
            start = random.randint(0, len(audio) - config.MAX_LEN)
            audio = audio[start:start + config.MAX_LEN]
        elif len(audio) < config.MAX_LEN:
            pad = config.MAX_LEN - len(audio)
            audio = np.pad(audio, (0, pad), mode="constant")

        return audio.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]

        audio = self.load_audio(path)

        # -------------------------
        # Augmentation (Train Only)
        # -------------------------
        if self.augment is not None:
            audio = self.augment(samples=audio, sample_rate=config.SR)
            audio = np.asarray(audio, dtype=np.float32)

        sample = {
            "audio": torch.from_numpy(audio),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "speaker": str(row["speaker"]),
            "attack": str(row["attack"]),
            "file": str(row["file"]),
        }

        return sample


class CachedFeatureDataset(Dataset):
    """Loads pre-extracted hybrid feature vectors cached as ``.pt`` files."""

    def __init__(self, feature_dir):
        self.paths = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No .pt feature files found under {feature_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = torch.load(self.paths[idx], map_location="cpu")
        return {
            "feature": item["feature"].float(),
            "label": int(item["label"]),
            "speaker": item["speaker"],
            "attack": item["attack"],
            "file": item["file"],
        }


def collate_features(batch):
    features = torch.stack([b["feature"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    speakers = [b["speaker"] for b in batch]
    attacks = [b["attack"] for b in batch]
    files = [b["file"] for b in batch]
    return {"feature": features, "label": labels, "speaker": speakers, "attack": attacks, "file": files}


def find_pt_dir(root):
    """Return ``root`` itself if it directly contains ``.pt`` files,
    otherwise descend one level to find the folder that does.

    This mirrors the notebook's helper (Cells 44/46), which was needed
    because a Kaggle-dataset restore sometimes nested the feature files one
    directory deeper than expected. It is kept for robustness when pointing
    ``PROCESSED_DIR`` at a directory produced by a different extraction run.
    """
    if any(f.endswith(".pt") for f in os.listdir(root)):
        return root
    for sub in os.listdir(root):
        subpath = os.path.join(root, sub)
        if os.path.isdir(subpath) and any(f.endswith(".pt") for f in os.listdir(subpath)):
            return subpath
    raise RuntimeError(f"No .pt files found under {root}")
