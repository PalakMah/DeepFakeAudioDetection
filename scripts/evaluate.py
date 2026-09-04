"""
Stage 3 — Evaluation.

Loads the trained general detector and personalizer, then replays the eval
split speaker-by-speaker through a fresh memory bank (simulating an online
deployment scenario), reporting metrics for both the general and
personalized detectors and writing a submission CSV.

Usage
-----
    python scripts/evaluate.py

Requires that ``scripts/prepare_data.py`` has produced a feature cache for
the eval split, and that ``scripts/train.py`` has produced both model
checkpoints.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.dataset import CachedFeatureDataset, find_pt_dir
from src.evaluation.evaluate import report_results, run_deployment_replay
from src.models.general_detector import GeneralDetector
from src.models.personalizer import CrossAttentionPersonalizer


def main():
    Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading cached eval features...")
    eval_dataset = CachedFeatureDataset(find_pt_dir(config.EVAL_FEATURES_DIR))
    print(f"Eval samples: {len(eval_dataset)}")

    general_model = GeneralDetector(embedding_dim=config.GENERAL_EMBEDDING_DIM).to(config.DEVICE)
    general_model.load_state_dict(torch.load(config.GENERAL_MODEL_PATH, map_location=config.DEVICE))

    personalizer = CrossAttentionPersonalizer(embedding_dim=config.GENERAL_EMBEDDING_DIM).to(config.DEVICE)
    personalizer.load_state_dict(torch.load(config.PERSONALIZER_MODEL_PATH, map_location=config.DEVICE))

    replay_df = run_deployment_replay(eval_dataset, general_model, personalizer)
    general_metrics, personalized_metrics, submission = report_results(
        replay_df, save_prefix=config.OUTPUTS_DIR.rstrip("/") + "/"
    )


if __name__ == "__main__":
    main()
