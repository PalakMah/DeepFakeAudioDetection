"""
Reading the official ASVspoof 2019 LA protocol files.

Corresponds to the notebook's "Cell 3 : Read Official ASVspoof 2019 Protocol
Files" (the version kept is the one in notebook Cell 7, which is identical in
logic to the earlier stray Cell 47 re-read used later for cross-checking).
"""

import os

import pandas as pd

from src import config


def read_protocol(protocol_path: str, audio_folder: str) -> pd.DataFrame:
    """Parse a single ASVspoof2019 CM protocol file into a DataFrame.

    Each line of the protocol file has the format:
        <speaker> <file> <system_id> <attack> <key>
    where <key> is either "bonafide" or "spoof".
    """
    rows = []

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            rows.append(
                {
                    "speaker": parts[0],
                    "file": parts[1],
                    "attack": parts[3],
                    "label": 1 if parts[4] == "bonafide" else 0,
                    "path": os.path.join(audio_folder, parts[1] + ".flac"),
                }
            )

    return pd.DataFrame(rows)


def load_official_splits():
    """Load the official train/dev/eval protocol DataFrames.

    Returns
    -------
    train_df, dev_df, eval_df : pd.DataFrame
    """
    train_df = read_protocol(config.TRAIN_PROTOCOL, config.TRAIN_AUDIO_DIR)
    dev_df = read_protocol(config.DEV_PROTOCOL, config.DEV_AUDIO_DIR)
    eval_df = read_protocol(config.EVAL_PROTOCOL, config.EVAL_AUDIO_DIR)
    return train_df, dev_df, eval_df


def print_split_summary(train_df: pd.DataFrame, dev_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Print the same summary the notebook printed after loading protocols."""
    print("=" * 60)
    print("Official ASVspoof 2019 Dataset")
    print("=" * 60)

    print(f"Train Samples : {len(train_df)}")
    print(f"Dev Samples   : {len(dev_df)}")
    print(f"Eval Samples  : {len(eval_df)}")
    print()

    print("Train Speakers :", train_df.speaker.nunique())
    print("Dev Speakers   :", dev_df.speaker.nunique())
    print("Eval Speakers  :", eval_df.speaker.nunique())
    print()

    print("Train Labels")
    print(train_df.label.value_counts())
    print()

    print("Dev Labels")
    print(dev_df.label.value_counts())
    print()

    print("Eval Labels")
    print(eval_df.label.value_counts())
