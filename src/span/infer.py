# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm

@dataclass
class QAExample:
    id: str
    question: str
    context: str

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_selected_context(sid: str, selected_obj, context_db) -> Optional[str]:
    if sid not in selected_obj: return None
    entry = selected_obj[sid]
    if isinstance(entry, dict) and isinstance(entry.get("context"), str):
        return entry["context"]
    if isinstance(entry, dict) and "pid" in entry:
        pid = entry["pid"]
        if isinstance(context_db, dict):
            return context_db.get(str(pid)) or context_db.get(pid)
        if isinstance(context_db, list):
            try: return context_db[int(pid)]
            except Exception: return None
    if isinstance(entry, str):
        return entry
    return None

def build_test_examples(test_path: str, selected_path: str, context_path: str) -> List[QAExample]:
    data = load_json(test_path)
    sel = load_json(selected_path)
    ctx = load_json(context_path)
    iterable = data if isinstance(data, list) else data.get("data", [])
    exs = []
    miss = 0
    for ex in iterable:
        sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
        q   = str(ex.get("question", ex.get("q",""))).strip()
        ctx_str = get_selected_context(sid, sel, ctx)
        if not ctx_str:
            miss += 1
            ctx_str = ""  # 沒選到也給空 context，至少不會炸
        exs.append(QAExample(sid, q, ctx_str))
    if miss: print(f"[warn] {miss} test examples have no selected context.")
    print(f"[build] test examples: {len(exs)}")
    return exs

def decode_best_span(start_logits, end_logits, offset_mapping, max_answer_length=64):
    # best by start+end，並限制長度
    best_s, best_e, best_score = 0, 0, -1e9
    # 用 top-k 可稍加速；也可全掃
    k = min(20, start_logits.shape[-1])
    s_top = torch.topk(start_logits, k=k).indices.tolist()
    e_top = torch.topk(end_logits,   k=k).indices.tolist()
    for s in s_top:
        for e in e_top:
            if e < s: continue
            if e - s + 1 > max_answer_length: continue
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_s, best_e, best_score = s, e, score
    s_off, _ = offset_mapping[best_s]
    _, e_off = offset_mapping[best_e]
    return s_off, e_off

def infer(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    exs = build_test_examples(args.test, args.selected, args.context)

    # 先 tokenize 成 features（滑窗）
    encoded = tok(
        [e.question for e in exs],
        [e.context  for e in exs],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded["offset_mapping"]

    # 前向、彙整每個 example 的多窗候選
    all_preds: Dict[str, List[str]] = {}
    bs = args.bs
    with torch.no_grad():
        pbar = tqdm(range(0, len(encoded["input_ids"]), bs), desc="Infer", unit="batch")
        for i in pbar:
            sl = slice(i, i+bs)
            input_ids = torch.tensor(encoded["input_ids"][sl]).to(device)
            attn_mask = torch.tensor(encoded["attention_mask"][sl]).to(device)
            token_type_ids = torch.tensor(encoded.get("token_type_ids", [0]*len(encoded["input_ids"]))[sl]).to(device)
            out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids, return_dict=True)
            start_logits = out.start_logits.detach().cpu()
            end_logits   = out.end_logits.detach().cpu()
            for j in range(start_logits.shape[0]):
                smp_idx = sample_mapping[i+j]
                ex = exs[smp_idx]
                offsets = offset_mapping[i+j]
                s_char, e_char = decode_best_span(start_logits[j], end_logits[j], offsets, args.max_answer_length)
                pred_text = ex.context[s_char:e_char]
                all_preds.setdefault(ex.id, []).append(pred_text)

    # 整理成單一答案（這裡取最長非空字串；可替換為投票）
    final: Dict[str,str] = {}
    for ex in exs:
        cands = [c for c in all_preds.get(ex.id, [""]) if c is not None]
        cands = sorted(cands, key=lambda s: len(s), reverse=True)
        final[ex.id] = cands[0] if cands else ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)
    print(f"[OK] wrote {len(final)} predictions to {args.out}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--selected", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="out/predictions.json")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--max_answer_length", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    return ap.parse_args()

if __name__ == "__main__":
    infer(parse_args())
