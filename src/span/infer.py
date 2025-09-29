# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

@dataclass
class QAItem:
    id: str
    question: str
    paragraph: str  # 單一段落文本

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_paragraphs_for_id(sid: str, selected_obj, context_db) -> List[str]:
    """支援多種 selected.json 格式，回傳該題的段落列表（文字）。"""
    if isinstance(selected_obj, dict):
        entry = selected_obj.get(sid)
    else:
        entry = None
        if isinstance(selected_obj, list):
            for e in selected_obj:
                if str(e.get("id")) == sid:
                    entry = e
                    break

    if not entry:
        return []

    # 優先 paragraphs
    if isinstance(entry, dict) and isinstance(entry.get("paragraphs"), list):
        out = []
        for p in entry["paragraphs"]:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, int):
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

    # 相容：單一 context
    if isinstance(entry, dict) and isinstance(entry.get("context"), str):
        return [entry["context"]]

    # 相容：pid 欄位
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
            paras = [""]  # 保底，至少會產出空字串答案
        for p in paras:
            items.append(QAItem(sid, q, p))
    if miss:
        print(f"[warn] {miss} test examples have no selected paragraphs.")
    print(f"[build] total QA items (with paragraphs): {len(items)}")
    return items

def infer(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 路徑健檢與容錯 ---
    ckpt = os.path.expanduser(args.ckpt_dir)
    if not os.path.isdir(ckpt):
        if os.path.isdir(os.path.join(ckpt, "best")):
            ckpt = os.path.join(ckpt, "best")
        else:
            raise FileNotFoundError(
                f"Checkpoint folder not found: {args.ckpt_dir}\n"
                f"請確認路徑（例如 models/span_roberta_large_a100/best）。"
            )

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt).to(device)
    model.eval()

    items = build_items(args.test, args.selected, args.context)

    encoded = tok(
        [it.question for it in items],
        [it.paragraph for it in items],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        padding="max_length"
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded["offset_mapping"]
    token_type_ids = encoded.get("token_type_ids", None)

    # 構建「有效 context token」mask：用 BatchEncoding.sequence_ids(i)
    valid_masks: List[List[bool]] = []
    total = len(encoded["input_ids"])
    for i in range(total):
        # 在某些版本中，sequence_ids 是 BatchEncoding 的方法
        seq_ids = encoded.sequence_ids(i) if hasattr(encoded, "sequence_ids") else None
        # 後備方案：用 token_type_ids 區分（BERT系常見：0=question/cls/sep, 1=context）
        if seq_ids is None:
            if token_type_ids is not None:
                seq_ids = [0 if tt == 0 else 1 for tt in token_type_ids[i]]
            else:
                # 保守策略：都當成 context（理論上不會走到這）
                seq_ids = [1] * len(encoded["input_ids"][i])

        offsets = offset_mapping[i]
        mask = []
        for sid, off in zip(seq_ids, offsets):
            if off is None:
                mask.append(False)
            else:
                s_off, e_off = off
                is_ctx = (sid == 1)
                mask.append(is_ctx and (e_off > s_off))
        valid_masks.append(mask)

    # 批次推論
    best_for_id: Dict[str, Tuple[float, str]] = {}
    B = args.bs
    with torch.no_grad():
        pbar = tqdm(range(0, total, B), desc="Infer", unit="batch")
        for i in pbar:
            sl = slice(i, i+B)
            input_ids = torch.tensor(encoded["input_ids"][sl]).to(device)
            attn_mask = torch.tensor(encoded["attention_mask"][sl]).to(device)
            if token_type_ids is None:
                tti = torch.zeros_like(input_ids)
            else:
                tti = torch.tensor(token_type_ids[sl]).to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=tti, return_dict=True)
            s_logits = out.start_logits.detach().cpu()
            e_logits = out.end_logits.detach().cpu()

            for j in range(s_logits.size(0)):
                vm = torch.tensor(valid_masks[i+j])
                s_logits[j][~vm] = float("-inf")
                e_logits[j][~vm] = float("-inf")

                k = min(20, s_logits.shape[1])
                s_top = torch.topk(s_logits[j], k=k).indices.tolist()
                e_top = torch.topk(e_logits[j], k=k).indices.tolist()

                smp_idx = sample_mapping[i+j]
                it = items[smp_idx]
                offsets = offset_mapping[i+j]

                best_s, best_e, best_score, best_text = 0, 0, float("-inf"), ""
                for s in s_top:
                    for e in e_top:
                        if e < s: continue
                        if e - s + 1 > args.max_answer_length: continue
                        off_s, off_e = offsets[s], offsets[e]
                        if off_s is None or off_e is None: continue
                        if off_e[1] <= off_s[0]: continue
                        sc = s_logits[j][s].item() + e_logits[j][e].item()
                        s_char, e_char = off_s[0], off_e[1]
                        pred = it.paragraph[s_char:e_char].strip()
                        if pred and sc > best_score:
                            best_s, best_e, best_score, best_text = s, e, sc, pred

                if best_text == "":
                    s_idx = int(torch.argmax(s_logits[j]))
                    e_idx = int(torch.argmax(e_logits[j]))
                    if e_idx < s_idx:
                        e_idx = s_idx
                    off_s, off_e = offsets[s_idx], offsets[e_idx]
                    s_char = 0 if off_s is None else off_s[0]
                    e_char = 0 if off_e is None else off_e[1]
                    best_text = it.paragraph[s_char:e_char].strip()
                    best_score = (s_logits[j][s_idx] + e_logits[j][e_idx]).item()

                old = best_for_id.get(it.id, (float("-inf"), ""))
                if best_score > old[0]:
                    best_for_id[it.id] = (best_score, best_text)

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
