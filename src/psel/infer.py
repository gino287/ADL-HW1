from __future__ import annotations
import os, json, argparse, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from src.dataio.load_adl import ADLPselDataset, psel_collate

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="out/selected.json")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    ds  = ADLPselDataset(args.context, args.test, tok, max_length=args.max_len, split="test")
    dl  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     collate_fn=lambda b: psel_collate(b, tok, args.max_len))

    model = AutoModelForMultipleChoice.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outs = []
    with torch.no_grad():
        for batch in dl:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            ).logits  # [B, C]
            pred_idx = logits.argmax(dim=-1).tolist()

            for i in range(len(batch["ids"])):
                chosen = batch["choices_text"][i][pred_idx[i]]
                outs.append({
                    "id": batch["ids"][i],
                    "question": batch["questions"][i],
                    "selected_paragraph": chosen
                })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(outs)} items to {args.out}")

if __name__ == "__main__":
    main()
