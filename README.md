# ADL HW1 — Chinese Extractive QA (Day 1: Skeleton & Repro)

Two-stage pipeline:
1) Paragraph Selection (psel, Multiple Choice) → out/selected.json
2) Span Selection (extractive QA) → out/predictions.json
Then → Kaggle CSV out/submission.csv (cols: id,prediction_text)

## Online (download ckpt)
# Provide URLs or create dummy zips if none provided
# PSEL_URL=<url1> SPAN_URL=<url2>
bash scripts/download.sh

This extracts:
- models/psel-best/
- models/span-best/

## Offline (inference, no internet)
bash scripts/run.sh
# inputs : data/context.json, data/test.json
# outputs: out/selected.json, out/predictions.json, out/submission.csv

## I/O Contracts
- context.json: list[str], paragraph_id = index
- test.json: list[ { id, question, paragraphs: list[int] } ]
- selected.json: list[ { id, question, selected_paragraph: str } ]
- predictions.json: list[ { id, prediction_text: str } ]
- submission.csv: header id,prediction_text

## Env
- Python 3.10
- (Later) Transformers 4.50.0, accelerate, datasets, evaluate
