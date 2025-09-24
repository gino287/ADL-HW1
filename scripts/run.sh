#!/usr/bin/env bash
# Offline pipeline: test.jsonl -> selected.jsonl -> predictions.json -> submission.csv
set -euo pipefail
mkdir -p out

echo "[1/3] Paragraph Selection (psel)"
python3 src/psel/infer.py --test data/test.jsonl --out out/selected.jsonl --ckpt_dir models/psel-best

echo "[2/3] Span Selection (extractive QA)"
python3 src/span/infer.py --selected out/selected.jsonl --out out/predictions.json --ckpt_dir models/span-best

echo "[3/3] Format to Kaggle CSV"
python3 src/utils/format_kaggle.py --pred out/predictions.json --out out/submission.csv

echo "[ok] Wrote out/submission.csv"
