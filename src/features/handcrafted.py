"""
Handcrafted acoustic feature extractors.

Corresponds to notebook Cells 14-19 (MFCC, CQCC, LFCC, spectral, prosody,
physics-guided features) plus the parallel batch wrapper from Cell 28.
"""

from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import parselmouth
from scipy.fftpack import dct
from spafe.features.lfcc import lfcc as spafe_lfcc

from src import config

SR = config.SR
N_MFCC = config.N_MFCC
N_CQCC = config.N_CQCC
N_LFCC = config.N_LFCC


# ============================================================
# Cell 14 : MFCC Feature Extraction
# ============================================================
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=160, win_length=400
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    feature = np.concatenate([mfcc_mean, delta_mean, delta2_mean])

    return feature.astype(np.float32)


# ============================================================
# Cell 15 : CQCC Feature Extraction
# ============================================================
def extract_cqcc(audio):
    # Constant-Q Transform
    cqt = np.abs(
        librosa.cqt(y=audio, sr=SR, hop_length=160, bins_per_octave=12, n_bins=84)
    )

    # Log Power
    log_cqt = np.log(cqt + 1e-8)

    # Cepstral Coefficients
    cqcc = dct(log_cqt, axis=0, norm="ortho")
    cqcc = cqcc[:N_CQCC]

    # Delta / Delta-Delta
    delta = librosa.feature.delta(cqcc)
    delta2 = librosa.feature.delta(cqcc, order=2)

    # Mean Pooling
    cqcc_mean = np.mean(cqcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    feature = np.concatenate([cqcc_mean, delta_mean, delta2_mean])

    return feature.astype(np.float32)


# ============================================================
# Cell 16 : LFCC Feature Extraction
# ============================================================
def extract_lfcc(audio):
    lfcc_feat = spafe_lfcc(
        sig=audio,
        fs=SR,
        num_ceps=N_LFCC,
        nfilts=80,
        nfft=512,
        low_freq=0,
        high_freq=SR // 2,
    )

    # (frames, coeffs) -> (coeffs, frames)
    lfcc_feat = lfcc_feat.T

    delta = librosa.feature.delta(lfcc_feat)
    delta2 = librosa.feature.delta(lfcc_feat, order=2)

    feature = np.concatenate(
        [np.mean(lfcc_feat, axis=1), np.mean(delta, axis=1), np.mean(delta2, axis=1)]
    )

    return feature.astype(np.float32)


# ============================================================
# Cell 17 : Spectral Feature Extraction
# ============================================================
def extract_spectral(audio):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
    flatness = librosa.feature.spectral_flatness(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=SR)

    feature = np.concatenate(
        [
            np.mean(centroid, axis=1),
            np.mean(bandwidth, axis=1),
            np.mean(rolloff, axis=1),
            np.mean(flatness, axis=1),
            np.mean(zcr, axis=1),
            np.mean(rms, axis=1),
            np.mean(contrast, axis=1),
        ]
    )

    return feature.astype(np.float32)


# ============================================================
# Cell 18 : Prosodic Feature Extraction
# ============================================================
def extract_prosody(audio):
    # Pitch (Fundamental Frequency)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=SR
    )

    pitch = f0[~np.isnan(f0)]

    if len(pitch) == 0:
        pitch_mean = 0
        pitch_std = 0
    else:
        pitch_mean = np.mean(pitch)
        pitch_std = np.std(pitch)

    # RMS Energy
    energy = librosa.feature.rms(y=audio)[0]
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # Voiced Ratio
    voiced_ratio = np.mean(voiced_flag.astype(np.float32))

    feature = np.array(
        [pitch_mean, pitch_std, energy_mean, energy_std, voiced_ratio], dtype=np.float32
    )

    return feature


# ============================================================
# Cell 19 : Physics-Guided Features (jitter / shimmer / HNR / intensity)
# ============================================================
def extract_physics(audio):
    sound = parselmouth.Sound(audio, sampling_frequency=SR)

    # Pitch
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]
    mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) else 0.0

    # Point Process (required for jitter & shimmer)
    point_process = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)", 75, 500
    )

    # Jitter
    jitter = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    # Shimmer
    shimmer = parselmouth.praat.call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # Harmonic-to-Noise Ratio
    harmonicity = sound.to_harmonicity()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    # Intensity
    intensity = sound.to_intensity()
    mean_intensity = parselmouth.praat.call(intensity, "Get mean", 0, 0, "energy")

    feature = np.array([mean_pitch, jitter, shimmer, hnr, mean_intensity], dtype=np.float32)
    feature = np.nan_to_num(feature)

    return feature


# ============================================================
# Cell 28 : Parallel Handcrafted Feature Extraction
# ============================================================
def extract_handcrafted(audio):
    mfcc = extract_mfcc(audio)
    cqcc = extract_cqcc(audio)
    lfcc = extract_lfcc(audio)
    spectral = extract_spectral(audio)
    prosody = extract_prosody(audio)
    physics = extract_physics(audio)

    return np.concatenate([mfcc, cqcc, lfcc, spectral, prosody, physics]).astype(np.float32)


def extract_handcrafted_batch(audio_batch, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        features = list(executor.map(extract_handcrafted, audio_batch))
    return features
