from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice

try:
    from src.dataio.load_adl import ADLPselDataset, psel_collate
except Exception:
    ADLPselDataset = None
    psel_collate = None

# ---------- utils ----------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def extract_answer_string(ex: dict) -> Optional[str]:
    ans = ex.get("answer")
    if isinstance(ans, dict) and isinstance(ans.get("text"), str):
        s = ans["text"].strip()
        return s if s else None
    # 兼容其他寫法
    if isinstance(ex.get("answer"), str) and ex["answer"].strip():
        return ex["answer"].strip()
    a2 = ex.get("answers")
    if isinstance(a2, dict) and a2.get("text"):
        t = a2["text"][0]
        return t.strip() if isinstance(t, str) else str(t)
    if isinstance(a2, list) and a2:
        a0 = a2[0]
        if isinstance(a0, dict) and "text" in a0:
            return str(a0["text"]).strip()
    return None

def get_id_and_question(ex: dict) -> Tuple[str, str]:
    sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
    q = str(ex.get("question", ex.get("q", ""))).strip()
    return sid, q

def get_para_text_by_id(pid, context_db) -> Optional[str]:
    # context 可能是 list 或 dict
    if isinstance(context_db, list):
        try:
            return context_db[int(pid)]
        except Exception:
            return None
    if isinstance(context_db, dict):
        return context_db.get(str(pid)) or context_db.get(pid)
    return None

def normalize_para_ids_to_texts(paras, context_db) -> List[str]:
    out = []
    for p in paras:
        if isinstance(p, (int, str)):
            t = get_para_text_by_id(p, context_db)
            if isinstance(t, str):
                out.append(t)
        elif isinstance(p, dict) and "text" in p:
            out.append(p["text"])
    return out

# ---------- oracle selection (train/valid) ----------
def oracle_select(split_path: str, context_path: str) -> Dict[str, Dict[str, str]]:
    data = load_json(split_path)
    ctx_db = load_json(context_path)
    iterable = data if isinstance(data, list) else data.get("data", [])
    out = {}
    miss = 0
    for ex in iterable:
        sid, _ = get_id_and_question(ex)
        if not sid:
            continue
        ans = extract_answer_string(ex)
        # 收集候選段落字串
        candidates: List[str] = []
        if "paragraphs" in ex and isinstance(ex["paragraphs"], list):
            candidates = normalize_para_ids_to_texts(ex["paragraphs"], ctx_db)
        elif isinstance(ex.get("context"), str):
            candidates = [ex["context"]]
        elif isinstance(ex.get("contexts"), list):
            candidates = normalize_para_ids_to_texts(ex["contexts"], ctx_db)

        chosen = None

        # 1) 若有 relevant（ID/索引），直接取
        if "relevant" in ex:
            rel = ex["relevant"]
            t = get_para_text_by_id(rel, ctx_db)
            if isinstance(t, str):
                chosen = t

        # 2) 沒有 relevant 或讀不到時，用答案字串搜尋候選
        if chosen is None and ans and candidates:
            for c in candidates:
                if ans in c:
                    chosen = c
                    break

        # 3) 都找不到就退而求其次：選最長段（讓下游至少有內容可學）
        if chosen is None and candidates:
            chosen = max(candidates, key=len)

        if chosen:
            out[sid] = {"context": chosen}
        else:
            miss += 1
    print(f"[oracle] {split_path} → built {len(out)} entries; miss {miss}")
    return out

# ---------- model selection (test) ----------
def model_select(context_path: str, qa_path: str, ckpt_dir: str, max_len: int, bs: int) -> Dict[str, Dict[str, str]]:
    if ADLPselDataset is None or psel_collate is None:
        raise RuntimeError("需要 src.dataio.load_adl.ADLPselDataset / psel_collate。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    ds  = ADLPselDataset(context_path, qa_path, tok, max_length=max_len, split="test")
    dl  = DataLoader(ds, batch_size=bs, shuffle=False,
                     collate_fn=lambda b: psel_collate(b, tok, max_len))

    model = AutoModelForMultipleChoice.from_pretrained(ckpt_dir).to(device)
    model.eval()

    sel = {}
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
                sid = batch["ids"][i]
                sel[sid] = {"context": chosen}
    print(f"[model] {qa_path} → built {len(sel)} entries")
    return sel

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=str, required=True, help="context.json（供 oracle/model 兩模式查段落）")
    ap.add_argument("--train", type=str, help="train.json")
    ap.add_argument("--valid", type=str, help="valid.json")
    ap.add_argument("--test",  type=str, help="test.json")
    ap.add_argument("--ckpt_dir", type=str, help="psel 權重（跑 test 的 model 模式需要）")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","oracle","model"],
                    help="auto: train/valid→oracle, test→model；或強制全 oracle / 全 model")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", type=str, default="out/selected.json")
    ap.add_argument("--format", type=str, default="kv", choices=["kv","list"])
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out)
    results: Dict[str, Dict[str,str]] = {}

    # train
    if args.train:
        if args.mode in ("auto","oracle"):
            res = oracle_select(args.train, args.context)
        else:
            res = model_select(args.context, args.train, args.ckpt_dir, args.max_len, args.batch_size)
        results.update(res)

    # valid
    if args.valid:
        if args.mode in ("auto","oracle"):
            res = oracle_select(args.valid, args.context)
        else:
            res = model_select(args.context, args.valid, args.ckpt_dir, args.max_len, args.batch_size)
        results.update(res)

    # test
    if args.test:
        if args.mode in ("auto","model"):
            if not args.ckpt_dir:
                raise ValueError("test 用 model 模式需要 --ckpt_dir")
            res = model_select(args.context, args.test, args.ckpt_dir, args.max_len, args.batch_size)
        else:
            # test 通常沒有答案；若強制 oracle，可能全 miss
            res = oracle_select(args.test, args.context)
        results.update(res)

    if args.format == "kv":
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    else:
        arr = [{"id": k, "selected_paragraph": v.get("context","")} for k,v in results.items()]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(results)} entries to {args.out} ({args.format})")

