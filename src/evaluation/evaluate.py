"""
Evaluation metrics and deployment-style replay evaluation.

Corresponds to notebook Cell 20 ("Evaluation Metrics") and Cell 24
("Deployment-Style Replay Evaluation + Final Reporting").
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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
from src.models.personalizer import pad_memory_batch
from src.training.train_personalizer import _embed_and_confidence, build_replay_records

BONAFIDE_LABEL = config.BONAFIDE_LABEL
SPOOF_LABEL = config.SPOOF_LABEL


# ============================================================
# Cell 20 : Evaluation Metrics (Accuracy, F1, AUC, EER)
# ============================================================
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def evaluate_predictions(y_true, y_prob_spoof, threshold=0.5):
    y_pred = (y_prob_spoof >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_spoof": f1_score(y_true, y_pred, zero_division=0),
        "f1_bonafide": f1_score(1 - np.array(y_true), 1 - y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob_spoof),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    eer, eer_thr = compute_eer(y_true, y_prob_spoof)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_thr
    return metrics


def best_f1_threshold(y_true, y_prob_spoof, num_steps=200):
    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_thr, best_macro_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_prob_spoof >= thr).astype(int)
        f1_spoof = f1_score(y_true, y_pred, zero_division=0)
        f1_bona = f1_score(1 - np.array(y_true), 1 - y_pred, zero_division=0)
        macro_f1 = (f1_spoof + f1_bona) / 2.0
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thr = thr
    return best_thr, best_macro_f1


def plot_confusion_matrices(y_true, general_scores, personalized_scores, threshold=0.5, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scores, title in zip(
        axes, [general_scores, personalized_scores], ["General Detector", "Personalized Detector"]
    ):
        y_pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bona fide", "spoof"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["bona fide", "spoof"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# Cell 24 : Deployment-Style Replay Evaluation + Final Reporting
# ============================================================
def run_deployment_replay(
    test_dataset,
    general_model,
    personalizer,
    embedding_dim=config.GENERAL_EMBEDDING_DIM,
    max_k=config.PERSONALIZER_MAX_K,
    reindex_every=config.PERSONALIZER_REINDEX_EVERY,
    device=config.DEVICE,
):
    """Replay a test split speaker-by-speaker through a fresh memory bank,
    scoring each clip with both the general and personalized detectors
    before (optionally) writing it into memory — simulating an online
    deployment scenario."""
    general_model.eval().to(device)
    personalizer.eval().to(device)

    bank = MemoryBank(embedding_dim=embedding_dim)
    quality = MemoryQualityController(bank)
    gate = ConfidenceGate()
    aging = MemoryAgingManager()
    prototype = PrototypeMemoryManager()
    retriever = AdaptiveMemoryRetriever(bank)
    controller = MemoryUpdateController(bank, quality, gate, aging, prototype, reindex_every=reindex_every)

    order = build_replay_records(test_dataset)
    records = []
    accepted_writes = 0
    complexities, top_ks = [], []

    with torch.no_grad():
        for idx in order:
            sample = test_dataset[idx]
            feat = sample["feature"].unsqueeze(0).to(device)
            embedding, confidence, general_logits = _embed_and_confidence(general_model, feat)
            speaker = sample["speaker"]
            label = sample["label"]

            general_probs = torch.softmax(torch.tensor(general_logits), dim=0).numpy()
            general_spoof_prob = float(general_probs[SPOOF_LABEL])

            retrieved, complexity, top_k = retriever.retrieve(speaker, embedding)
            complexities.append(complexity)
            top_ks.append(top_k)

            query_emb = torch.tensor(embedding, dtype=torch.float32, device=device).unsqueeze(0)
            mem_emb, mem_mask, age_w = pad_memory_batch([retrieved], embedding_dim, max_k)
            mem_emb, mem_mask, age_w = mem_emb.to(device), mem_mask.to(device), age_w.to(device)

            p_logits, _ = personalizer(query_emb, mem_emb, mem_mask, age_w, training=False)
            p_probs = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
            personalized_spoof_prob = float(p_probs[SPOOF_LABEL])

            physics_vec = sample["feature"][config.PHYSICS_SLICE[0]:config.PHYSICS_SLICE[1]].numpy()
            accepted, _ = controller.maybe_write(speaker, embedding, physics_vec, confidence)
            if accepted:
                accepted_writes += 1

            records.append(
                {
                    "file": sample["file"],
                    "speaker": speaker,
                    "label": label,
                    "general_prob_spoof": general_spoof_prob,
                    "personalized_prob_spoof": personalized_spoof_prob,
                    "memory_size_at_query": bank.count(speaker),
                    "retrieved_k": top_k,
                    "complexity": complexity,
                    "written_to_memory": accepted,
                }
            )

    controller.flush()
    df = pd.DataFrame.from_records(records)
    print(f"Replayed {len(df)} clips across {df['speaker'].nunique()} speakers.")
    print(f"Accepted memory writes: {accepted_writes} ({accepted_writes / len(df):.1%} of clips)")
    print(f"Dynamic Top-K: mean={np.mean(top_ks):.2f}, range=[{min(top_ks)},{max(top_ks)}]")
    print(f"Acoustic complexity: mean={np.mean(complexities):.3f}")
    return df


def report_results(df, save_prefix=None):
    """Compute general/personalized metrics, plot confusion matrices, and
    write a submission CSV with per-clip predictions."""
    save_prefix = save_prefix or (config.OUTPUTS_DIR.rstrip("/") + "/")

    y_true_is_spoof = (df["label"].values == SPOOF_LABEL).astype(int)

    general_metrics = evaluate_predictions(y_true_is_spoof, df["general_prob_spoof"].values, threshold=0.5)
    personalized_metrics = evaluate_predictions(y_true_is_spoof, df["personalized_prob_spoof"].values, threshold=0.5)

    print("\n=== General Detector (threshold=0.5) ===")
    for k, v in general_metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Personalized Detector (threshold=0.5) ===")
    for k, v in personalized_metrics.items():
        print(f"  {k}: {v}")

    best_thr, best_macro_f1 = best_f1_threshold(y_true_is_spoof, df["personalized_prob_spoof"].values)
    print(f"\nPersonalized best-F1 threshold: {best_thr:.4f} (macro-F1={best_macro_f1:.4f})")

    plot_confusion_matrices(
        y_true_is_spoof,
        df["general_prob_spoof"].values,
        df["personalized_prob_spoof"].values,
        threshold=0.5,
        save_path=save_prefix + "figures/confusion_matrices.png",
    )

    submission = df.copy()
    submission["general_pred"] = np.where(submission["general_prob_spoof"] >= 0.5, SPOOF_LABEL, BONAFIDE_LABEL)
    submission["personalized_pred"] = np.where(
        submission["personalized_prob_spoof"] >= 0.5, SPOOF_LABEL, BONAFIDE_LABEL
    )
    submission["general_correct"] = (submission["general_pred"] == submission["label"]).astype(int)
    submission["personalized_correct"] = (submission["personalized_pred"] == submission["label"]).astype(int)
    submission.to_csv(save_prefix + "submission.csv", index=False)
    print(f"\nWrote {save_prefix}submission.csv")

    return general_metrics, personalized_metrics, submission
