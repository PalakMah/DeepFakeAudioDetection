"""
Batched hybrid-feature extraction and on-disk caching.

Corresponds to notebook Cell 33 ("process_split"). The Kaggle-specific
checkpoint push/restore to a Kaggle Dataset (notebook Cells 30-32, 34-36) has
been removed, since that mechanism only existed to survive ephemeral Kaggle
kernel sessions. The core resumability logic — skipping files that were
already extracted on a previous run — is preserved unchanged, so re-running
this pipeline after an interruption still resumes correctly using the local
filesystem as the cache.
"""

import gc
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from src import config
from src.features.handcrafted import extract_handcrafted_batch
from src.features.xlsr import extract_xlsr_batch


def process_split(
    dataset,
    split: str,
    feature_extractor,
    xlsr_model,
    save_dir: str = None,
    batch_size: int = config.FEATURE_EXTRACTION_BATCH_SIZE,
    print_every: int = config.FEATURE_PRINT_EVERY,
    device=config.DEVICE,
):
    """Extract and cache hybrid features for every sample in ``dataset``.

    Already-extracted files (matched by ``<file>.pt``) are skipped, so the
    function is safe to re-run after an interruption.
    """
    save_dir = save_dir or os.path.join(config.PROCESSED_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".pt")])

    print("=" * 60)
    print(f"{split.upper()} SPLIT")
    print("=" * 60)
    print(f"Dataset Size     : {len(dataset)}")
    print(f"Already Extracted: {existing}")
    print(f"Remaining        : {len(dataset) - existing}")
    print("=" * 60)

    processed = 0
    skipped = 0

    pbar = tqdm(total=len(dataset), desc=split)

    for batch_idx, start in enumerate(range(0, len(dataset), batch_size)):
        end = min(start + batch_size, len(dataset))

        samples = []
        audio_batch = []

        for i in range(start, end):
            try:
                sample = dataset[i]
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                pbar.update(1)
                continue

            save_path = os.path.join(save_dir, sample["file"] + ".pt")

            if os.path.exists(save_path):
                skipped += 1
                pbar.update(1)
                continue

            samples.append(sample)
            audio_batch.append(sample["audio"].numpy())

        if len(samples) == 0:
            continue

        # Feature Extraction
        handcrafted_batch = extract_handcrafted_batch(audio_batch)

        with torch.inference_mode():
            xlsr_batch = extract_xlsr_batch(audio_batch, feature_extractor, xlsr_model, device)

        # Save Features
        for sample, handcrafted, xlsr in zip(samples, handcrafted_batch, xlsr_batch):
            feature = np.concatenate([handcrafted, xlsr]).astype(np.float32)

            torch.save(
                {
                    "feature": torch.from_numpy(feature),
                    "label": int(sample["label"]),
                    "speaker": sample["speaker"],
                    "attack": sample["attack"],
                    "file": sample["file"],
                },
                os.path.join(save_dir, sample["file"] + ".pt"),
            )

            processed += 1

        pbar.update(len(samples))

        if (batch_idx + 1) % print_every == 0:
            print(f"[{split}] Processed={processed} Skipped={skipped}")

        # Memory Cleanup
        del samples
        del audio_batch
        del handcrafted_batch
        del xlsr_batch

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    print("\n" + "=" * 60)
    print(f"{split.upper()} COMPLETED")
    print(f"New Features : {processed}")
    print(f"Skipped      : {skipped}")
    print("=" * 60)
