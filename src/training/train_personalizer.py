"""
Training loop for the cross-attention personalization head.

Corresponds to notebook Cell 23 ("Train the Personalization Head"):
``_embed_and_confidence``, ``build_replay_records``, ``train_personalizer``.
"""

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.models.memory_bank import (
    AdaptiveMemoryRetriever,
    ConfidenceGate,
    MemoryAgingManager,
    MemoryBank,
    MemoryQualityController,
    MemoryUpdateController,
    PrototypeMemoryManager,
)
from src.models.personalizer import CrossAttentionPersonalizer, pad_memory_batch


@torch.no_grad()
def _embed_and_confidence(general_model, feature_tensor):
    logits, embedding, _ = general_model(feature_tensor, return_embedding=True)
    probs = torch.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values.item()
    return embedding.squeeze(0).cpu().numpy(), confidence, logits.squeeze(0).cpu().numpy()


def build_replay_records(dataset):
    """Order sample indices so that all clips from the same speaker are
    replayed contiguously (simulating a deployment stream), in order of
    first appearance of each speaker in the dataset."""
    speaker_order = []
    seen = set()
    buckets = {}
    for i in range(len(dataset)):
        spk = dataset[i]["speaker"]
        if spk not in seen:
            seen.add(spk)
            speaker_order.append(spk)
            buckets[spk] = []
        buckets[spk].append(i)

    ordered_indices = []
    for spk in speaker_order:
        ordered_indices.extend(buckets[spk])
    return ordered_indices


def train_personalizer(
    train_dataset,
    general_model,
    embedding_dim=config.GENERAL_EMBEDDING_DIM,
    epochs=config.PERSONALIZER_EPOCHS,
    lr=config.PERSONALIZER_LR,
    weight_decay=config.PERSONALIZER_WEIGHT_DECAY,
    context_dropout_p=config.PERSONALIZER_CONTEXT_DROPOUT_P,
    max_k=config.PERSONALIZER_MAX_K,
    grad_clip_norm=config.PERSONALIZER_GRAD_CLIP_NORM,
    reindex_every=config.PERSONALIZER_REINDEX_EVERY,
    batch_size=config.PERSONALIZER_BATCH_SIZE,
    device=config.DEVICE,
):
    """Train ``CrossAttentionPersonalizer`` by replaying ``train_dataset``
    speaker-by-speaker through a fresh memory bank each epoch, retrieving
    each speaker's accumulated memory before writing the current sample.
    """
    general_model.eval().to(device)
    personalizer = CrossAttentionPersonalizer(
        embedding_dim=embedding_dim, context_dropout_p=context_dropout_p
    ).to(device)
    optimizer = torch.optim.AdamW(personalizer.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss()

    order = build_replay_records(train_dataset)
    history = {"epoch_loss": []}

    for epoch in range(1, epochs + 1):
        bank = MemoryBank(embedding_dim=embedding_dim)
        quality = MemoryQualityController(bank)
        gate = ConfidenceGate()
        aging = MemoryAgingManager()
        prototype = PrototypeMemoryManager()
        retriever = AdaptiveMemoryRetriever(bank)
        controller = MemoryUpdateController(bank, quality, gate, aging, prototype, reindex_every=reindex_every)

        personalizer.train()
        running_loss, n_seen = 0.0, 0

        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            embeddings_batch, retrieved_batch, labels_batch = [], [], []

            for idx in idxs:
                sample = train_dataset[idx]
                feat = sample["feature"].unsqueeze(0).to(device)
                embedding, confidence, _ = _embed_and_confidence(general_model, feat)
                speaker = sample["speaker"]

                retrieved, complexity, top_k = retriever.retrieve(speaker, embedding)
                physics_vec = sample["feature"][config.PHYSICS_SLICE[0]:config.PHYSICS_SLICE[1]].numpy()
                controller.maybe_write(speaker, embedding, physics_vec, confidence)

                embeddings_batch.append(embedding)
                retrieved_batch.append(retrieved)
                labels_batch.append(sample["label"])

            controller.flush()

            query_emb = torch.tensor(np.stack(embeddings_batch), dtype=torch.float32, device=device)
            mem_emb, mem_mask, age_w = pad_memory_batch(retrieved_batch, embedding_dim, max_k)
            mem_emb, mem_mask, age_w = mem_emb.to(device), mem_mask.to(device), age_w.to(device)
            labels = torch.tensor(labels_batch, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = personalizer(query_emb, mem_emb, mem_mask, age_w, training=True)
            loss = ce_loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(personalizer.parameters(), grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * len(idxs)
            n_seen += len(idxs)

        epoch_loss = running_loss / n_seen
        history["epoch_loss"].append(epoch_loss)
        print(f"[Personalizer] Epoch {epoch}/{epochs} loss={epoch_loss:.4f}")

    return personalizer, history
