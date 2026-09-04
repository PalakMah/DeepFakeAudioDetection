"""
Cross-attention personalization head.

Corresponds to notebook Cell 22 ("Cross-Attention Personalization Head"):
``CrossAttentionPersonalizer`` and ``pad_memory_batch``.
"""

import torch
import torch.nn as nn


class CrossAttentionPersonalizer(nn.Module):
    """Attends the query embedding over a speaker's retrieved memory rows
    (age-weighted), then fuses query + attended context into a final
    per-speaker prediction."""

    def __init__(self, embedding_dim=256, num_heads=4, context_dropout_p=0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_dropout_p = context_dropout_p
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.project = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, query_embedding, memory_embeddings, memory_mask, age_weights, training=None):
        training = self.training if training is None else training
        B = query_embedding.size(0)
        query = query_embedding.unsqueeze(1)

        if training and self.context_dropout_p > 0:
            drop = torch.rand(B, device=query_embedding.device) < self.context_dropout_p
            if drop.any():
                memory_mask = memory_mask.clone()
                memory_mask[drop] = True

        weighted_memory = memory_embeddings * age_weights.unsqueeze(-1)
        fully_masked = memory_mask.all(dim=1)

        context = torch.zeros_like(query_embedding)
        if (~fully_masked).any():
            active = ~fully_masked
            attn_out, _ = self.attn(
                query[active],
                weighted_memory[active],
                weighted_memory[active],
                key_padding_mask=memory_mask[active],
            )
            context[active] = attn_out.squeeze(1)

        fused = torch.cat([query_embedding, context], dim=1)
        personalized_repr = self.project(fused)
        logits = self.classifier(personalized_repr)
        return logits, personalized_repr


def pad_memory_batch(retrieved_list, embedding_dim, max_k):
    """Pad a list of per-sample retrieved-memory lists into fixed-size
    batched tensors (embeddings, boolean padding mask, age weights)."""
    B = len(retrieved_list)
    memory_embeddings = torch.zeros(B, max_k, embedding_dim, dtype=torch.float32)
    memory_mask = torch.ones(B, max_k, dtype=torch.bool)
    age_weights = torch.zeros(B, max_k, dtype=torch.float32)

    for i, entries in enumerate(retrieved_list):
        for j, e in enumerate(entries[:max_k]):
            memory_embeddings[i, j] = torch.from_numpy(e["embedding"])
            age_weights[i, j] = e["age_weight"]
            memory_mask[i, j] = False

    return memory_embeddings, memory_mask, age_weights
