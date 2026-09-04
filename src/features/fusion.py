"""
Hybrid feature fusion: concatenates all handcrafted features with the XLS-R
deep embedding into a single 1767-d vector.

Corresponds to notebook Cell 25 (single-sample fusion). The batched
extraction pipeline used for caching features to disk lives in
``src/features/extraction_pipeline.py`` (notebook Cell 33), and reuses the
same feature ordering defined here.
"""

import numpy as np

from src.features.handcrafted import (
    extract_cqcc,
    extract_lfcc,
    extract_mfcc,
    extract_physics,
    extract_prosody,
    extract_spectral,
)
from src.features.xlsr import extract_xlsr


def extract_all_features(audio, feature_extractor, xlsr_model, device):
    """Build the full hybrid feature vector for one audio clip.

    Order (must match ``config.FEATURE_GROUPS``):
        mfcc(240) + cqcc(240) + lfcc(240) + spectral(13) + prosody(5)
        + physics(5) + xlsr(1024) = 1767
    """
    # Handcrafted Features
    mfcc_feature = extract_mfcc(audio)
    cqcc_feature = extract_cqcc(audio)
    lfcc_feature = extract_lfcc(audio)
    spectral_feature = extract_spectral(audio)
    prosody_feature = extract_prosody(audio)
    physics_feature = extract_physics(audio)

    # Deep Representation
    xlsr_feature = extract_xlsr(audio, feature_extractor, xlsr_model, device)

    # Concatenate Everything
    hybrid_feature = np.concatenate(
        [
            mfcc_feature,
            cqcc_feature,
            lfcc_feature,
            spectral_feature,
            prosody_feature,
            physics_feature,
            xlsr_feature,
        ],
        axis=0,
    )

    return hybrid_feature.astype(np.float32)
