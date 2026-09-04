"""
Training loop for the general (speaker-agnostic) detector.

Corresponds to notebook Cell 19 ("Train the General Embedding +
Classification Model"): ``compute_class_weights``, ``mine_speaker_triplets``,
``train_general_model``.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import collate_features
from src.models.general_detector import GeneralDetector


def compute_class_weights(dataset):
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    labels = np.array(labels)
    n_bonafide = (labels == 1).sum()
    n_spoof = (labels == 0).sum()
    total = n_bonafide + n_spoof
    w_spoof = total / (2.0 * max(n_spoof, 1))
    w_bonafide = total / (2.0 * max(n_bonafide, 1))
    return torch.tensor([w_spoof, w_bonafide], dtype=torch.float32)


def mine_speaker_triplets(embeddings, labels, speakers):
    """Hard-negative triplet mining: anchor/positive share speaker+label,
    anchor/negative share speaker but differ in label (falling back to any
    opposite-label sample if no same-speaker negative exists)."""
    n = embeddings.size(0)
    labels_np = labels.detach().cpu().numpy()
    speakers_np = np.array(speakers)

    anchors, positives, negatives = [], [], []

    for i in range(n):
        same_speaker = speakers_np == speakers_np[i]
        same_label = labels_np == labels_np[i]
        opp_label = ~same_label

        pos_mask = same_speaker & same_label
        pos_mask[i] = False
        pos_candidates = np.where(pos_mask)[0]
        if len(pos_candidates) == 0:
            continue

        hard_neg_mask = same_speaker & opp_label
        neg_candidates = np.where(hard_neg_mask)[0]
        if len(neg_candidates) == 0:
            neg_candidates = np.where(opp_label)[0]
        if len(neg_candidates) == 0:
            continue

        anchors.append(i)
        positives.append(np.random.choice(pos_candidates))
        negatives.append(np.random.choice(neg_candidates))

    if len(anchors) == 0:
        return None

    return (
        torch.tensor(anchors, dtype=torch.long, device=embeddings.device),
        torch.tensor(positives, dtype=torch.long, device=embeddings.device),
        torch.tensor(negatives, dtype=torch.long, device=embeddings.device),
    )


def train_general_model(
    train_dataset,
    val_dataset,
    embedding_dim=config.GENERAL_EMBEDDING_DIM,
    batch_size=config.GENERAL_TRAIN_BATCH_SIZE,
    epochs=config.GENERAL_EPOCHS,
    lr=config.GENERAL_LR,
    weight_decay=config.GENERAL_WEIGHT_DECAY,
    triplet_margin=config.GENERAL_TRIPLET_MARGIN,
    triplet_weight=config.GENERAL_TRIPLET_WEIGHT,
    grad_clip_norm=config.GENERAL_GRAD_CLIP_NORM,
    early_stop_patience=config.GENERAL_EARLY_STOP_PATIENCE,
    num_workers=config.NUM_WORKERS,
    device=config.DEVICE,
):
    """Train ``GeneralDetector`` with cross-entropy + speaker-triplet loss,
    cosine LR schedule, gradient clipping and early stopping on val loss.

    Returns the best model (by val loss) and its training history.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_features,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_features,
    )

    model = GeneralDetector(embedding_dim=embedding_dim).to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    triplet_loss_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch in train_loader:
            x = batch["feature"].to(device)
            y = batch["label"].to(device)
            speakers = batch["speaker"]

            optimizer.zero_grad()
            logits, embeddings, _ = model(x, return_embedding=True)
            loss = ce_loss_fn(logits, y)

            triplet_idx = mine_speaker_triplets(embeddings, y, speakers)
            if triplet_idx is not None:
                a_idx, p_idx, n_idx = triplet_idx
                t_loss = triplet_loss_fn(embeddings[a_idx], embeddings[p_idx], embeddings[n_idx])
                loss = loss + triplet_weight * t_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += x.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["feature"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                loss = ce_loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history
