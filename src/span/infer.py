# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm

@dataclass
class QAItem:
    id: str
    question: str
    paragraph: str  # 單一段落文本

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_paragraphs_for_id(sid: str, selected_obj, context_db) -> List[str]:
    """支援多種 selected.json 格式，回傳該題的段落列表。"""
    if isinstance(selected_obj, dict):
        entry = selected_obj.get(sid)
    else:
        entry = None
        # list 形式：[{id, paragraphs:[...]}]
        if isinstance(selected_obj, list):
            for e in selected_obj:
                if str(e.get("id")) == sid:
                    entry = e
                    break

    if not entry:
        return []

    # 優先 paragraphs
    if isinstance(entry, dict) and isinstance(entry.get("paragraphs"), list):
        paras = entry["paragraphs"]
        # paras 可能已是文字；或是 pid/obj
        out = []
        for p in paras:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, (int,)):
                # pid → text
                if isinstance(context_db, dict):
                    t = context_db.get(str(p)) or context_db.get(p)
                else:
                    try:
                        t = context_db[int(p)]
                    except Exception:
                        t = None
                if isinstance(t, str):
                    out.append(t)
            elif isinstance(p, dict) and "text" in p:
                out.append(p["text"])
        return out

    # 相容舊格式：單一 context
    if isinstance(entry, dict) and isinstance(entry.get("context"), str):
        return [entry["context"]]

    # 相容 pid 欄位
    if isinstance(entry, dict) and "pid" in entry:
        pid = entry["pid"]
        if isinstance(context_db, dict):
            t = context_db.get(str(pid)) or context_db.get(pid)
        else:
            try:
                t = context_db[int(pid)]
            except Exception:
                t = None
        return [t] if isinstance(t, str) else []

    # 直接字串
    if isinstance(entry, str):
        return [entry]

    return []

def build_items(test_path: str, selected_path: str, context_path: str) -> List[QAItem]:
    data = load_json(test_path)
    sel = load_json(selected_path)
    ctx = load_json(context_path)
    iterable = data if isinstance(data, list) else data.get("data", [])
    items: List[QAItem] = []
    miss = 0
    for ex in iterable:
        sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
        q   = str(ex.get("question", ex.get("q",""))).strip()
        paras = extract_paragraphs_for_id(sid, sel, ctx)
        if not paras:
            miss += 1
            paras = [""]  # 保底
        for p in paras:
            items.append(QAItem(sid, q, p))
    if miss:
        print(f"[warn] {miss} test examples have no selected paragraphs.")
    print(f"[build] total QA items (with paragraphs): {len(items)}")
    return items

def decode_best_span(start_logits, end_logits, offset_mapping, max_answer_length=64):
    best_s, best_e, best_score = 0, 0, -1e9
    # 小幅剪枝：top-k 索引；也可改成全掃
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
    return s_off, e_off, best_score

def infer(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    items = build_items(args.test, args.selected, args.context)

    # 先把每個 item tokenize（滑窗）
    encoded = tok(
        [it.question for it in items],
        [it.paragraph for it in items],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded["offset_mapping"]

    # 推論：對每個 window 算分，回填到「題目層級」，取全域最高分 span
    best_for_id: Dict[str, Tuple[float, str]] = {}  # id -> (score, text)

    bs = args.bs
    with torch.no_grad():
        pbar = tqdm(range(0, len(encoded["input_ids"]), bs), desc="Infer", unit="batch")
        for i in pbar:
            sl = slice(i, i+bs)
            input_ids = torch.tensor(encoded["input_ids"][sl]).to(device)
            attn_mask = torch.tensor(encoded["attention_mask"][sl]).to(device)
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = [[0]*len(encoded["input_ids"][0]) for _ in range(len(encoded["input_ids"]))]
            token_type_ids = torch.tensor(token_type_ids[sl]).to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids, return_dict=True)
            start_logits = out.start_logits.detach().cpu()
            end_logits   = out.end_logits.detach().cpu()

            for j in range(start_logits.shape[0]):
                smp_idx  = sample_mapping[i+j]   # 指回原 items 索引
                it       = items[smp_idx]
                offsets  = offset_mapping[i+j]
                s_char, e_char, score = decode_best_span(start_logits[j], end_logits[j], offsets, args.max_answer_length)
                pred_text = it.paragraph[s_char:e_char].strip()

                # 更新該題目前最佳
                old = best_for_id.get(it.id, (-1e9, ""))
                if score > old[0]:
                    best_for_id[it.id] = (score, pred_text)

    # 收斂成最終答案
    final: Dict[str, str] = {qid: txt for qid, (sc, txt) in best_for_id.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
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
