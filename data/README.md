# Dataset: ASVspoof 2019 — Logical Access (LA)

This project trains and evaluates on the **ASVspoof 2019 Logical Access (LA)**
corpus, used via the Kaggle mirror `awsaf49/asvpoof-2019-dataset` in the
original notebook. You can obtain the data from either:

- Kaggle: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
- The official ASVspoof 2019 release: https://datashare.ed.ac.uk/handle/10283/3336

## Expected layout

Download and extract the dataset so that it looks like this under
`data/raw/` (or point the `ASVSPOOF_DATA_ROOT` environment variable at
wherever you extracted it):

```
data/raw/LA/LA/
├── ASVspoof2019_LA_train/
│   └── flac/                     # training .flac audio files
├── ASVspoof2019_LA_dev/
│   └── flac/                     # dev .flac audio files
├── ASVspoof2019_LA_eval/
│   └── flac/                     # eval .flac audio files
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

If you keep the dataset somewhere else, set:

```bash
export ASVSPOOF_DATA_ROOT=/path/to/LA/LA
```

## Dataset statistics (from the original notebook run)

| Split | Clips  | Speakers | Bona fide | Spoof |
|-------|-------:|---------:|----------:|------:|
| Train | 25,380 | 20       | 2,580     | 22,800|
| Dev   | 24,844 | 20       | 2,548     | 22,296|
| Eval  | 71,237 | 67       | 7,355     | 63,882|

## Processed features

`scripts/prepare_data.py` extracts a 1767-d hybrid feature vector (MFCC +
CQCC + LFCC + spectral + prosody + physics-guided features + XLS-R
embedding) for every clip and caches it as a `.pt` file under
`data/processed/hybrid_features/<split>/<file_id>.pt`. This directory is
**not** committed to git (see the top-level `.gitignore`) — regenerate it
locally by running the data-preparation script.
