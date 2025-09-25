# ADL HW1 — Chinese Extractive QA

## Project Overview
This project is the framework for **ADL HW1**, focusing on **Chinese Extractive Question Answering**. The process consists of two main stages:

1. **Paragraph Selection (psel)**: Select the most relevant paragraphs from `context.json`
2. **Span Selection (span)**: Extract the final answer from the selected paragraphs

The final output is a `submission.csv` file in the following format:
```
id,answer
5e7a923dd6e4ccb8730eb95230e0c908,1996年
...
```

## Directory Structure
```
adl-hw1/
├── scripts/
│   ├── download.sh     # Online stage (model download)
│   └── run.sh          # Offline stage (complete pipeline)
├── src/
│   ├── dataio/
│   │   └── load_adl.py # Data conversion to multiple-choice format
│   ├── psel/
│   │   ├── train.py    # Paragraph selection training (BERT multiple-choice)
│   │   └── infer.py    # Paragraph selection inference
│   ├── span/
│   │   ├── train.py    # Span extraction training (BERT QA)
│   │   └── infer.py    # Span extraction inference
│   └── utils/
│       └── format_kaggle.py # Conversion to submission.csv
├── models/
│   ├── psel-best/      # Trained paragraph selection model
│   └── span-best/      # Trained span extraction model
├── data/
│   ├── context.json
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── out/
│   ├── selected.json   # psel output (paragraph selection results)
│   ├── predictions.json # span output (answer extraction results)
│   └── submission.csv  # Final Kaggle submission file
├── report/figures/
├── README.md
└── requirements.txt
```

## Usage Instructions

### Online Stage (Download Models)
```bash
bash scripts/download.sh
# Validation
test -f models/psel-best/config.json && test -f models/span-best/meta.json && echo "[ok] download stage passed"
```

### Offline Stage (Complete Pipeline)
```bash
bash scripts/run.sh
```

The process will sequentially:
1. **Paragraph Selection (psel)**: Read `context.json` and `test.json`, select relevant paragraphs → `out/selected.json`
2. **Span Extraction (span)**: Extract answers from selected paragraphs → `out/predictions.json`
3. **Format Conversion**: Convert to Kaggle format → `out/submission.csv`

## Day 3 Progress
- ✅ **psel**: Paragraph selection training and inference completed, generates `out/selected.json` (train/valid/test)
- ✅ **span/train.py**: Supports No-Trainer training, BERT (bert-base-chinese) on Colab T4 running 2 epochs, valid F1 ≈ 85.55
- ✅ **span/infer.py**: Inference on test set, generates `out/predictions.json`
- ✅ **utils/format_kaggle.py**: Converts to Kaggle-compatible `submission.csv`
- ✅ **run.sh**: Complete pipeline successfully integrated, one-click from data to submission file

Initial Kaggle submission (S0) completed, Public Leaderboard score: approximately 0.71.

## Requirements
- Python 3.10
- Key packages:
  - transformers==4.50.0
  - accelerate==0.34.2
  - datasets==3.0.1
  - evaluate==0.4.3
  - tqdm==4.66.4

### Installation
```bash
pip install -r requirements.txt
```
