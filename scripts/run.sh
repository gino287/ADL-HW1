#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

mkdir -p out

echo "[1/3] Paragraph Selection (psel, top-k)"
"$PYTHON" src/psel/infer.py \
  --context data/context.json \
  --train data/train.json \
  --valid data/valid.json \
  --test  data/test.json \
  --ckpt_dir models/psel-best \
  --topk 3 \
  --out out/selected.json

echo "[2/3] Span Selection (extractive QA, cross-paragraph aggregation)"
"$PYTHON" src/span/infer.py \
  --test data/test.json \
  --selected out/selected.json \
  --context data/context.json \
  --ckpt_dir models/span-best \
  --out out/predictions.json

echo "[3/3] Format to Kaggle CSV (id,answer)"
"$PYTHON" src/utils/format_kaggle.py \
  --pred out/predictions.json \
  --test data/test.json \
  --out  out/submission.csv
