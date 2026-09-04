"""
General (speaker-agnostic) spoof detector: a gated feature-fusion module
feeding a small MLP encoder and classifier.

Corresponds to notebook Cell 48 (``FeatureFusion``, ``GeneralDetector``).
"""

import torch
import torch.nn as nn

from src.config import FEATURE_GROUPS, TOTAL_FEATURE_DIM


class FeatureFusion(nn.Module):
    """Per-group LayerNorm followed by a learned softmax gate over the
    normalized feature groups (mfcc/cqcc/lfcc/spectral/prosody/physics/xlsr).
    """

    def __init__(self, groups=FEATURE_GROUPS, gate_hidden=128):
        super().__init__()
        self.groups = groups
        self.group_names = list(groups.keys())
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(end - start) for name, (start, end) in groups.items()}
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(TOTAL_FEATURE_DIM, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(groups)),
        )

    def forward(self, x):
        normed_chunks = []
        for name, (start, end) in self.groups.items():
            chunk = x[:, start:end]
            normed_chunks.append(self.norms[name](chunk))
        normed_concat = torch.cat(normed_chunks, dim=1)

        gate_logits = self.gate_mlp(normed_concat)
        gate_weights = torch.softmax(gate_logits, dim=1)

        fused_chunks = []
        for i, (name, (start, end)) in enumerate(self.groups.items()):
            w = gate_weights[:, i:i + 1]
            fused_chunks.append(normed_chunks[i] * w)
        fused = torch.cat(fused_chunks, dim=1)

        return fused, gate_weights


class GeneralDetector(nn.Module):
    """Fusion -> MLP encoder -> linear classifier, with an optional
    L2-normalized embedding output used both for the speaker-triplet loss
    during training and as the query representation for the personalization
    stage."""

    def __init__(self, embedding_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.fusion = FeatureFusion()
        self.encoder = nn.Sequential(
            nn.Linear(TOTAL_FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        fused, gate_weights = self.fusion(x)
        embedding = self.encoder(fused)
        embedding_norm = nn.functional.normalize(embedding, dim=1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding_norm, gate_weights
        return logits
