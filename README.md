# Speaker-Personalized Audio Spoof Detection (ASVspoof 2019 LA)

A hybrid-feature audio deepfake / spoofing countermeasure for the ASVspoof
2019 Logical Access (LA) task, extended with an **online speaker-memory
personalization** stage that adapts predictions to each speaker as more of
their clips are seen.


## Key features

- **Hybrid acoustic feature extraction**: MFCC, CQCC, LFCC, spectral
  statistics, prosodic features, and Praat-derived physics-guided features
  (jitter, shimmer, HNR, intensity), fused with a deep **XLS-R
  (`wav2vec2-xls-r-300m`)** embedding into a single 1767-d vector per clip.
- **Gated feature fusion + general detector**: a learned softmax gate
  weights each feature group before an MLP encoder/classifier, trained with
  cross-entropy plus a speaker-aware triplet loss.
- **Speaker memory bank**: a FAISS-backed, SQLite-persisted store of past
  speaker embeddings with confidence gating, exponential aging, and
  K-means prototype compression to bound memory growth.
- **Cross-attention personalization head**: attends each new clip's
  embedding over that speaker's retrieved memory to produce a
  speaker-adapted prediction.
- **Deployment-style replay evaluation**: replays a held-out split
  speaker-by-speaker through a fresh memory bank (as a live system would
  see it), comparing general vs. personalized performance.

## Project architecture / workflow

```
Raw audio (.flac)
      │
      ▼
Protocol parsing (speaker, file, attack, label)
      │
      ▼
Waveform loading, pad/crop to 4s @ 16kHz, augmentation (train only)
      │
      ▼
Hybrid feature extraction ─────────────────────────────────┐
  MFCC(240) + CQCC(240) + LFCC(240) + spectral(13)          │
  + prosody(5) + physics(5) + XLS-R(1024) = 1767-d          │
      │                                                     │
      ▼                                                     │
Cached .pt feature files (data/processed/hybrid_features/)  │
      │                                                     │
      ▼                                                     │
┌─────────────────────────┐                                 │
│  General Detector        │  gated fusion → MLP → 256-d    │
│  (CE + speaker triplet)  │  embedding → 2-way classifier   │
└─────────────────────────┘                                 │
      │ embeddings                                          │
      ▼                                                     │
┌─────────────────────────┐                                 │
│  Speaker Memory Bank      │  quality scoring, confidence   │
│  + Personalization Head   │  gating, aging, K-means         │
│  (cross-attention)        │  compression, adaptive top-K    │
└─────────────────────────┘                                 │
      │                                                     │
      ▼                                                     │
Deployment-style replay evaluation (general vs. personalized) 
      │
      ▼
Metrics (accuracy, precision, recall, F1, AUC, EER) + submission.csv
```

## Directory structure

```
project-root/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/                  # place the ASVspoof2019 LA dataset here (gitignored)
│   ├── processed/            # cached hybrid feature .pt files (gitignored)
│   └── README.md             # dataset download & placement instructions
├── notebooks/
│   └── original_kaggle_notebook.ipynb   # the original, unmodified notebook
├── src/
│   ├── config.py             # all paths, hyperparameters, constants
│   ├── data/
│   │   ├── protocol.py       # ASVspoof protocol file parsing
│   │   ├── augmentation.py   # training-time waveform augmentation
│   │   └── dataset.py        # ASVSpoofDataset, CachedFeatureDataset
│   ├── features/
│   │   ├── handcrafted.py    # MFCC / CQCC / LFCC / spectral / prosody / physics
│   │   ├── xlsr.py           # XLS-R model loading + embedding extraction
│   │   ├── fusion.py         # single-sample hybrid feature fusion
│   │   └── extraction_pipeline.py  # batched extraction + on-disk caching
│   ├── models/
│   │   ├── general_detector.py  # FeatureFusion + GeneralDetector
│   │   ├── memory_bank.py       # speaker memory bank subsystem
│   │   └── personalizer.py      # CrossAttentionPersonalizer
│   ├── training/
│   │   ├── train_general.py       # general detector training loop
│   │   └── train_personalizer.py  # personalizer training loop
│   └── evaluation/
│       └── evaluate.py       # metrics + deployment replay evaluation
├── scripts/
│   ├── prepare_data.py       # Stage 1: extract & cache hybrid features
│   ├── train.py               # Stage 2: train general model + personalizer
│   └── evaluate.py            # Stage 3: replay evaluation on eval split
└── outputs/
    ├── figures/               # confusion matrices, plots
    └── models/                # general_model.pt, personalizer.pt
```

### Dataset & Preprocessing

The preprocessing pipeline for the **FaceForensics++ (FF++) dataset**, including face detection, frame extraction, face cropping, alignment, and preprocessing, is available in the accompanying repository:

[PalakMah/FaceForensics](https://github.com/PalakMah/FaceForensics?utm_source=chatgpt.com)

The repository also provides the **extracted frames/processed face crops** generated from the FaceForensics++ dataset, which can be used directly for downstream deepfake detection experiments without repeating the preprocessing pipeline.

> **Note:** The original FaceForensics++ dataset is subject to its own terms of use. This repository does not redistribute the original raw dataset.


## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**GPU/CUDA**: XLS-R feature extraction and model training will use CUDA
automatically if `torch.cuda.is_available()` — install a CUDA-enabled build
of PyTorch for your system if you have a GPU (see
https://pytorch.org/get-started/locally/). The project also runs on CPU,
just considerably more slowly for the XLS-R extraction step.

Then place the dataset as described in [`data/README.md`](data/README.md).

## Usage

### 1. Prepare data (extract & cache hybrid features)

```bash
python scripts/prepare_data.py --splits train dev eval
```

This is resumable: re-running it skips any `.pt` files that already exist
under `data/processed/hybrid_features/<split>/`.

### 2. Train

```bash
python scripts/train.py
```

Trains the general detector first (saved to
`outputs/models/general_model.pt`), then the personalization head on top of
it (saved to `outputs/models/personalizer.pt`). Use `--skip-general` to
reuse an existing general-model checkpoint, or `--skip-personalizer` to
train only the general detector.

### 3. Evaluate

```bash
python scripts/evaluate.py
```

Runs the deployment-style replay evaluation on the eval split, prints
metrics for both the general and personalized detectors, saves confusion
matrices to `outputs/figures/confusion_matrices.png`, and writes
`outputs/submission.csv`.

## Results

The metrics below were produced by the original notebook run on the full
ASVspoof2019 LA **eval** split (71,237 clips, 67 speakers), threshold 0.5:

| Metric | General Detector | Personalized Detector |
|---|---:|---:|
| Accuracy | 0.9300 | 0.9716 |
| Precision | 0.9993 | 0.9988 |
| Recall | 0.9226 | 0.9696 |
| F1 (spoof) | 0.9594 | 0.9839 |
| F1 (bona fide) | 0.7458 | 0.8781 |
| AUC | 0.9966 | 0.9978 |
| EER | 2.35% | 1.90% |

During replay, 65.6% of eval clips were accepted into speaker memory, with
a dynamic top-K retrieval averaging 4.85 (range 3–10) and mean acoustic
complexity of 0.265. The personalization head was trained for 6 epochs,
with training loss dropping from 0.0710 (epoch 1) to 0.0131 (epoch 6).

These numbers are reported as-is from the notebook; re-running the
pipeline (which includes stochastic augmentation, triplet mining, and
K-means compression) may produce slightly different results.

## Technologies used

Python, PyTorch, Hugging Face Transformers (`wav2vec2-xls-r-300m`),
librosa, `spafe`, `praat-parselmouth`, `audiomentations`, scikit-learn,
FAISS, SQLite, pandas, matplotlib.

## Research methodology

1. **Feature engineering**: combine classical spoofing-countermeasure
   features (MFCC/CQCC/LFCC capture spectral artifacts typical of
   synthesis/vocoding; spectral and prosodic statistics; physics-guided
   voice-production features from Praat) with a self-supervised deep
   speech representation (XLS-R) that captures higher-level acoustic
   structure.
2. **Gated fusion**: rather than naively concatenating heterogeneous
   feature groups, a learned gate weights each group's contribution
   per-sample after per-group normalization.
3. **Speaker-aware triplet loss**: alongside the classification loss, hard
   triplets are mined within-speaker (same speaker, opposite label as
   negative) to encourage the embedding space to separate bona fide from
   spoofed speech *for the same speaker*, which is the harder
   discrimination problem than across speakers.
4. **Online personalization**: a per-speaker memory of accepted embeddings
   (gated by confidence + physics/embedding similarity to existing memory)
   is retrieved via FAISS and attended over by a cross-attention head,
   letting the model adapt to a speaker's specific bona fide voice
   characteristics as more of their clips are observed — evaluated with a
   realistic speaker-ordered replay rather than a shuffled test set.

## Limitations

- The memory bank's aging model uses a demo-scale `seconds_per_year`
  constant (1000 seconds), not real calendar time — appropriate for a
  same-session replay evaluation but not calibrated for long-term
  deployment; see [Ambiguities](#ambiguities--assumptions) below.
- Personalization is evaluated via replay on the same corpus (ASVspoof2019
  LA eval) rather than a separate deployment dataset; generalization to
  other corpora, languages, or spoofing attack types (LA vs. PA vs.
  newer TTS/voice-conversion systems) is untested here.
- The physics-guided features (jitter/shimmer/HNR via Praat) can be noisy
  or degenerate on very short or heavily distorted clips; the pipeline
  guards against NaNs but does not otherwise validate feature quality.
- XLS-R inference is the main compute/time bottleneck; extracting features
  for the full corpus (~120k clips) is time-consuming on CPU.

## Future work

- Evaluate on ASVspoof 2021/2019 PA, ASVspoof5, or in-the-wild deepfake
  audio datasets to test cross-corpus generalization.
- Replace the demo-scale memory aging constant with a real wall-clock
  decay calibrated for production deployment.
- Ablate the contribution of each handcrafted feature group vs. the XLS-R
  embedding alone.
- Explore lighter-weight deep embeddings (e.g. smaller wav2vec2/HuBERT
  variants) to reduce feature-extraction cost.

## Citation

If you use the ASVspoof 2019 dataset, please cite the official ASVspoof
2019 paper/organizers (see https://www.asvspoof.org/). If you use the
`wav2vec2-xls-r-300m` model, please cite:

> Babu, A. et al. "XLS-R: Self-supervised Cross-lingual Speech
> Representation Learning at Scale." (2021)

## Refactoring notes

This repository was refactored from a Kaggle notebook. The pipeline logic,
model architectures, hyperparameters, and feature definitions are
unchanged. The following Kaggle-only mechanics were removed, since they
only existed to survive ephemeral Kaggle kernel sessions and have no
equivalent outside Kaggle:

- Kaggle API authentication / `kaggle.json` setup.
- Periodic checkpoint **push** of extracted feature `.zip` archives to a
  private Kaggle Dataset (`push_checkpoint_to_dataset`), and the matching
  **restore** of dev/eval checkpoints from that dataset.
- Ad-hoc directory-listing/debugging cells used to inspect Kaggle's
  `/kaggle/input` mount.

The resumability that mattered (skipping already-extracted files on
re-run) is preserved unchanged in `src/features/extraction_pipeline.py`,
now checkpointed against the local filesystem instead of a Kaggle Dataset.

### Ambiguities / assumptions

- Two versions of `ASVSpoofDataset` existed in the notebook (an earlier
  draft and a refined version that added a file-existence check and used
  `random.randint`/`reset_index`). The refined, later-defined version was
  taken as canonical, since it is what the rest of the notebook actually
  executed against.
- `push_checkpoint_to_dataset` and the associated Kaggle-dataset restore
  cells were dropped rather than reimplemented against a generic remote
  store, since no such destination was specified in the notebook.
