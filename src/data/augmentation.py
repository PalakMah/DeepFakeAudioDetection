"""
Waveform augmentation pipeline used only for the training split.

Corresponds to notebook Cell 8.
"""

from audiomentations import (
    AddGaussianNoise,
    AirAbsorption,
    Compose,
    Gain,
    PitchShift,
    Shift,
    TimeStretch,
)

# ==========================================================
# Train Augmentation
# ==========================================================
train_augment = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.40),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.30),
        TimeStretch(min_rate=0.90, max_rate=1.10, p=0.30),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.30),
        Shift(min_shift=-0.10, max_shift=0.10, p=0.25),
        AirAbsorption(p=0.20),
    ]
)

# ==========================================================
# Validation / Test
# ==========================================================
val_augment = None
test_augment = None
