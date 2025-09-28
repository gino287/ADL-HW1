# src/psel/infer.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice

from src.dataio.load_adl import ADLPselDataset, psel_collate

# ---------- utils ----------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def extract_answer_string(ex: dict) -> Optional[str]:
    """盡量從常見欄位撈出文字答案（for oracle 過濾輔助）。"""
    ans = ex.get("answer")
    if isinstance(ans, dict) and isinstance(ans.get("text"), str):
        s = ans["text"].strip()
        return s if s else None
    if isinstance(ans, str) and ans.strip():
        return ans.strip()
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
    sid = str(ex.get("id", ex.get("qid", ex.get("uid", ""))))
    q = str(ex.get("question", ex.get("q", ""))).strip()
    return sid, q

def get_para_text_by_id(pid, context_db) -> Optional[str]:
    """context_db 允許 list[str] 或 {pid:str}。"""
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

def prefer_best_dir(ckpt_dir: str) -> str:
    """若 ckpt_dir/best 存在，優先使用；否則用 ckpt_dir 本身。"""
    best = os.path.join(ckpt_dir, "best")
    return best if os.path.isdir(best) else ckpt_dir

# ---------- oracle selection (train/valid 產參考答案集) ----------
def oracle_select(split_path: str, context_path: str, topk: int) -> Dict[str, Dict[str, List[str]]]:
    data = load_json(split_path)
    ctx_db = load_json(context_path)
    iterable = data if isinstance(data, list) else data.get("data", [])
    out: Dict[str, Dict[str, List[str]]] = {}
    miss = 0

    for ex in iterable:
        sid, _ = get_id_and_question(ex)
        if not sid:
            continue
        ans = extract_answer_string(ex)

        # 先從 candidate id 轉文字
        candidates: List[str] = []
        if isinstance(ex.get("paragraphs"), list):
            candidates = normalize_para_ids_to_texts(ex["paragraphs"], ctx_db)
        elif isinstance(ex.get("contexts"), list):
            candidates = normalize_para_ids_to_texts(ex["contexts"], ctx_db)
        elif isinstance(ex.get("context"), str):
            candidates = [ex["context"]]

        chosen: List[str] = []

        # 1) 有標註 relevant → 放進來
        if "relevant" in ex:
            t = get_para_text_by_id(ex["relevant"], ctx_db)
            if isinstance(t, str):
                chosen.append(t)

        # 2) 有文字答案 → 補上包含答案字串的段落
        if ans and candidates:
            for c in candidates:
                if ans in c and c not in chosen:
                    chosen.append(c)

        # 3) 不足 → 用較長的段落湊滿 topk（避免空）
        if candidates:
            for c in sorted(candidates, key=len, reverse=True):
                if len(chosen) >= topk:
                    break
                if c not in chosen and c.strip():
                    chosen.append(c)

        if chosen:
            out[sid] = {"paragraphs": chosen[:topk]}
        else:
            miss += 1

    print(f"[oracle] {split_path} → built {len(out)} entries; miss {miss}")
    return out

# ---------- model selection (test 用已訓練模型挑 topk) ----------
def model_select(context_path: str, qa_path: str, ckpt_dir: str, max_len: int, bs: int, topk: int) -> Dict[str, Dict[str, List[str]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = prefer_best_dir(ckpt_dir)

    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

    # 用 ADLPselDataset 對 test 組 batch：會回傳 ids / choices_text 等輔助欄位
    ds  = ADLPselDataset(context_path, qa_path, tok, max_length=max_len, split="test")
    dl  = DataLoader(ds, batch_size=bs, shuffle=False,
                     collate_fn=lambda b: psel_collate(b, tok, max_length=max_len))

    model = AutoModelForMultipleChoice.from_pretrained(ckpt_dir).to(device)
    model.eval()

    sel: Dict[str, Dict[str, List[str]]] = {}
    with torch.no_grad():
        for batch in dl:
            # 有些模型（如 RoBERTa）可能沒有 token_type_ids，這裡做保險
            model_inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            logits = model(**model_inputs).logits  # [B, C]
            k = min(topk, logits.size(-1))
            _, indices = torch.topk(logits, k=k, dim=-1)  # [B, k]

            for i in range(len(batch["ids"])):
                sid = batch["ids"][i]
                para_texts = batch["choices_text"][i]
                idx_list = indices[i].tolist()
                picked = [para_texts[j] for j in idx_list if j < len(para_texts)]
                sel[sid] = {"paragraphs": picked}

    print(f"[model] {qa_path} → built {len(sel)} entries with topk={topk}")
    return sel

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=str, required=True, help="context.json（list[str] 或 {pid:text}）")
    ap.add_argument("--train", type=str, help="train.json")
    ap.add_argument("--valid", type=str, help="valid.json")
    ap.add_argument("--test",  type=str, help="test.json")
    ap.add_argument("--ckpt_dir", type=str, help="psel 權重資料夾；若含 best 子資料夾會優先使用")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","oracle","model"],
                    help="auto: train/valid→oracle, test→model；或強制全 oracle / 全 model")
    ap.add_argument("--topk", type=int, default=3, help="每題輸出前 k 段落")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", type=str, default="out/selected.json")
    ap.add_argument("--format", type=str, default="kv", choices=["kv","list"])
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out)
    results: Dict[str, Dict[str, List[str]]] = {}

    # train
    if args.train:
        if args.mode in ("auto","oracle"):
            res = oracle_select(args.train, args.context, args.topk)
        else:
            if not args.ckpt_dir:
                raise ValueError("train 用 model 模式需要 --ckpt_dir")
            res = model_select(args.context, args.train, args.ckpt_dir, args.max_len, args.batch_size, args.topk)
        results.update(res)

    # valid
    if args.valid:
        if args.mode in ("auto","oracle"):
            res = oracle_select(args.valid, args.context, args.topk)
        else:
            if not args.ckpt_dir:
                raise ValueError("valid 用 model 模式需要 --ckpt_dir")
            res = model_select(args.context, args.valid, args.ckpt_dir, args.max_len, args.batch_size, args.topk)
        results.update(res)

    # test
    if args.test:
        if args.mode in ("auto","model"):
            if not args.ckpt_dir:
                raise ValueError("test 用 model 模式需要 --ckpt_dir")
            res = model_select(args.context, args.test, args.ckpt_dir, args.max_len, args.batch_size, args.topk)
        else:
            res = oracle_select(args.test, args.context, args.topk)
        results.update(res)

    if args.format == "kv":
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    else:
        arr = [{"id": k, "paragraphs": v.get("paragraphs", [])} for k,v in results.items()]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {len(results)} entries to {args.out} ({args.format}), topk={args.topk}")

if __name__ == "__main__":
    main()
