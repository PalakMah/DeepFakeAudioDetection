"""
XLS-R (wav2vec2-xls-r-300m) deep embedding extraction.

Corresponds to notebook Cells 21/26 (model loading) and Cells 22/29
(single-sample and batched extraction).
"""

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src import config


def load_xlsr_model(model_name: str = config.XLSR_MODEL_NAME, device=config.DEVICE):
    """Load the pretrained XLS-R feature extractor and model.

    Returns
    -------
    feature_extractor : Wav2Vec2FeatureExtractor
    xlsr_model : Wav2Vec2Model (in eval mode, moved to ``device``)
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    xlsr_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    xlsr_model.eval()
    return feature_extractor, xlsr_model


@torch.no_grad()
def extract_xlsr(audio, feature_extractor, xlsr_model, device=config.DEVICE):
    """Extract a single XLS-R embedding (mean-pooled over time)."""
    inputs = feature_extractor(audio, sampling_rate=config.SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    outputs = xlsr_model(input_values)
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_xlsr_batch(audio_batch, feature_extractor, xlsr_model, device=config.DEVICE):
    """Extract XLS-R embeddings for a batch of waveforms (mean-pooled)."""
    inputs = feature_extractor(
        audio_batch, sampling_rate=config.SR, padding=True, return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)

    outputs = xlsr_model(input_values)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()
