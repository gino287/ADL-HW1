# -*- coding: utf-8 -*-
"""
No-Trainer Extractive QA (BERT, Chinese)
- 讀 out/selected.json（含 train/valid/test）
- 建 start/end 監督，sliding window + offset 對齊
- 進度條 tqdm，最佳 F1 自動存 models/span-best/ + meta.json
"""

import os, json, math, time, random, argparse, hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

# ---------------- utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<16), b''):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer_string(ex: dict) -> Optional[str]:
    """
    支援多種格式：
      - ex['answer'] 是字串
      - ex['answer'] 是物件，含 'text'
      - ex['answers'] = {'text':[...]} 或 [{'text':...}]
    """
    if 'answer' in ex:
        a = ex['answer']
        if isinstance(a, str) and a.strip():
            return a.strip()
        if isinstance(a, dict) and isinstance(a.get('text'), str) and a['text'].strip():
            return a['text'].strip()
    if 'answers' in ex:
        ans = ex['answers']
        if isinstance(ans, dict) and 'text' in ans and len(ans['text'])>0:
            s = str(ans['text'][0]).strip()
            return s if s else None
        if isinstance(ans, list) and len(ans)>0:
            a0 = ans[0]
            if isinstance(a0, dict) and 'text' in a0:
                s = str(a0['text']).strip()
                return s if s else None
    return None

def get_selected_context(sample_id: str, selected_obj, context_db) -> Optional[str]:
    """
    支援：
      A) { "<id>": {"context": "..."} }
      B) { "<id>": {"pid": <paragraph_id>} }（透過 context.json 查）
      C) { "<id>": "<context string>" }
    """
    if sample_id not in selected_obj:
        return None
    entry = selected_obj[sample_id]
    if isinstance(entry, dict):
        if 'context' in entry and isinstance(entry['context'], str):
            return entry['context']
        if 'pid' in entry:
            pid = entry['pid']
            if isinstance(context_db, dict):
                return context_db.get(str(pid)) or context_db.get(pid)
            if isinstance(context_db, list):
                try:
                    return context_db[int(pid)]
                except Exception:
                    return None
    elif isinstance(entry, str):
        return entry
    return None

@dataclass
class QAExample:
    id: str
    question: str
    context: str
    answer: Optional[str] = None  # train/valid 有，test 無

def build_examples(split_path: str, selected_path: str, context_path: str, require_answer: bool, limit: Optional[int]=None) -> List[QAExample]:
    data = load_json(split_path)
    sel = load_json(selected_path)
    ctx_db = load_json(context_path)
    examples = []
    miss_ctx = 0
    miss_ans = 0
    iterable = data if isinstance(data, list) else data.get('data', [])
    for ex in iterable:
        sid = str(ex.get('id', ex.get('qid', ex.get('uid'))))
        q   = str(ex.get('question', ex.get('q', ''))).strip()
        if not sid or not q:
            continue
        ctx = get_selected_context(sid, sel, ctx_db)
        if not ctx:
            miss_ctx += 1
            if require_answer:
                continue
            else:
                ctx = ""
        ans = extract_answer_string(ex)
        if require_answer and not ans:
            miss_ans += 1
            continue
        examples.append(QAExample(id=sid, question=q, context=ctx, answer=ans))
        if limit and len(examples) >= limit:
            break
    if require_answer:
        print(f"[build] loaded {len(examples)} examples | miss_ctx={miss_ctx} miss_ans={miss_ans}")
    else:
        print(f"[build] loaded {len(examples)} examples (test)")
    return examples

def char_em_f1(pred: str, gold: str) -> Tuple[float, float]:
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if gold == "":
        return (1.0 if pred=="" else 0.0, 1.0 if pred=="" else 0.0)
    if pred == "":
        return (0.0, 0.0)
    if pred == gold:
        return 1.0, 1.0
    pc, gc = Counter(list(pred)), Counter(list(gold))
    inter = sum((pc & gc).values())
    prec = inter / max(1, len(pred))
    rec  = inter / max(1, len(gold))
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    em = 1.0 if pred == gold else 0.0
    return em, f1

def decode_best_span(all_start, all_end, offset_mapping, max_answer_length=64):
    best_score = -1e9; best = (0, 0)
    k = min(20, all_start.shape[-1])
    start_topk = torch.topk(all_start, k=k).indices.tolist()
    end_topk   = torch.topk(all_end,   k=k).indices.tolist()
    for s in start_topk:
        for e in end_topk:
            if e < s:
                continue
            if e - s + 1 > max_answer_length:
                continue
            score = all_start[s].item() + all_end[e].item()
            if score > best_score:
                best_score = score; best = (s, e)
    s, e = best
    os, _ = offset_mapping[s]
    _, oe = offset_mapping[e]
    return os, oe

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--valid", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--selected", type=str, required=True)
    parser.add_argument("--model", type=str, default="bert-base-chinese")
    parser.add_argument("--output_dir", type=str, default="models/span-best")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_answer_length", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # 1) 準備資料
    train_examples = build_examples(args.train, args.selected, args.context, require_answer=True)
    valid_examples = build_examples(args.valid, args.selected, args.context, require_answer=True)
    print(f"[info] train: {len(train_examples)}, valid: {len(valid_examples)}")
    assert len(train_examples) > 0 and len(valid_examples) > 0, "train/valid 為 0，請檢查 selected.json 與答案字段解析。"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)

    # 2) Feature 轉換（含 offset 映射）
    def tokenize_examples(examples: List[QAExample], train_mode: bool):
        questions = [e.question for e in examples]
        contexts  = [e.context  for e in examples]
        encoded = tokenizer(
            questions, contexts,
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        offset_mapping = encoded["offset_mapping"]
        start_positions = []
        end_positions   = []
        aligned = 0
        for i, off in enumerate(offset_mapping):
            if not train_mode:
                start_positions.append(0); end_positions.append(0);
                continue
            sample_idx = sample_mapping[i]
            answer = examples[sample_idx].answer or ""
            context = examples[sample_idx].context or ""

            ans_start_char = context.find(answer)
            if ans_start_char < 0:
                start_positions.append(0); end_positions.append(0)
                continue
            ans_end_char = ans_start_char + len(answer)

            sequence_ids = encoded.sequence_ids(i)
            idxs = [k for k, sid in enumerate(sequence_ids) if sid == 1]
            if not idxs:
                start_positions.append(0); end_positions.append(0);
                continue
            ctx_start, ctx_end = idxs[0], idxs[-1]

            token_start = token_end = None
            for k in range(ctx_start, ctx_end+1):
                s, e = off[k]
                if s <= ans_start_char < e:
                    token_start = k
                if s <  ans_end_char <= e:
                    token_end = k
                if token_start is not None and token_end is not None:
                    break
            if token_start is None or token_end is None:
                start_positions.append(0); end_positions.append(0)
            else:
                start_positions.append(token_start); end_positions.append(token_end)
                aligned += 1

        encoded["start_positions"] = start_positions
        encoded["end_positions"]   = end_positions
        encoded["overflow_to_sample_mapping"] = sample_mapping
        encoded["offset_mapping"] = offset_mapping
        if train_mode:
            total = len(offset_mapping)
            print(f"[align] {aligned}/{total} features aligned to gold span ({aligned/total*100:.1f}%)")
        return encoded

    train_feats = tokenize_examples(train_examples, train_mode=True)
    valid_feats = tokenize_examples(valid_examples, train_mode=True)

    class QADataset(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k not in []}

    train_loader = DataLoader(QADataset(train_feats), batch_size=args.bs, shuffle=True,  num_workers=args.num_workers)
    valid_loader = DataLoader(QADataset(valid_feats), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # 3) Optim / Scheduler / AMP
    optim = AdamW(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1*max_train_steps), num_training_steps=max_train_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_f1 = -1.0

    def evaluate():
        model.eval()
        total_loss = 0.0; n = 0
        with torch.no_grad():
            for batch in DataLoader(QADataset(valid_feats), batch_size=args.bs, shuffle=False, num_workers=0):
                for k in ["input_ids","attention_mask","token_type_ids","start_positions","end_positions"]:
                    if k in batch: batch[k] = batch[k].to(device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
                total_loss += outputs.loss.item(); n += 1
        avg_loss = total_loss/max(1,n)

        # 解碼 EM/F1
        all_preds = defaultdict(list)
        with torch.no_grad():
            for i in range(0, len(valid_feats["input_ids"]), args.bs):
                sl = slice(i, i+args.bs)
                input_ids = torch.tensor(valid_feats["input_ids"][sl]).to(device)
                attn_mask = torch.tensor(valid_feats["attention_mask"][sl]).to(device)
                token_type_ids = torch.tensor(valid_feats.get("token_type_ids", [0]*len(valid_feats["input_ids"]))[sl]).to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids, return_dict=True)
                start_logits = out.start_logits.detach().cpu()
                end_logits   = out.end_logits.detach().cpu()
                sample_mapping = valid_feats["overflow_to_sample_mapping"][sl]
                for j in range(start_logits.shape[0]):
                    smp_idx = sample_mapping[j]
                    ex = valid_examples[smp_idx]
                    offsets = valid_feats["offset_mapping"][i+j]
                    s_char, e_char = decode_best_span(start_logits[j], end_logits[j], offsets, args.max_answer_length)
                    pred_text = ex.context[s_char:e_char]
                    all_preds[ex.id].append(pred_text)

        ems, f1s = [], []
        for ex in valid_examples:
            cand_list = all_preds.get(ex.id, [""])
            cand_list = sorted(cand_list, key=lambda s: len(s), reverse=True)
            pred = cand_list[0] if cand_list else ""
            em, f1 = char_em_f1(pred, ex.answer or "")
            ems.append(em); f1s.append(f1)
        EM = float(np.mean(ems))*100.0
        F1 = float(np.mean(f1s))*100.0
        return avg_loss, EM, F1

    # 4) 訓練迴圈（tqdm 進度條 + 自動存檔 best）
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", unit="batch")
        run_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            for k in ["input_ids","attention_mask","token_type_ids","start_positions","end_positions"]:
                if k in batch: batch[k] = batch[k].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
                loss = outputs.loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(optim); scaler.update(); sched.step()
                optim.zero_grad(set_to_none=True); global_step += 1
            run_loss += loss.item()
            pbar.set_postfix(loss=f"{run_loss/step:.4f}")

        v_loss, v_em, v_f1 = evaluate()
        print(f"[valid] epoch {epoch} | loss {v_loss:.4f} | EM {v_em:.2f} | F1 {v_f1:.2f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            print(f"[save] new best F1={best_f1:.2f} → {args.output_dir}")
            tokenizer.save_pretrained(args.output_dir)
            model.save_pretrained(args.output_dir)
            meta = {
                "model_name": args.model,
                "epochs_trained": epoch,
                "batch_size": args.bs,
                "lr": args.lr,
                "max_length": args.max_length,
                "doc_stride": args.doc_stride,
                "max_answer_length": args.max_answer_length,
                "grad_accum": args.grad_accum,
                "seed": args.seed,
                "valid_em": v_em,
                "valid_f1": v_f1,
                "train_json": args.train,
                "valid_json": args.valid,
                "context_json": args.context,
                "selected_json": args.selected,
                "selected_sha1": sha1_of_file(args.selected) if os.path.exists(args.selected) else None,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] best F1: {best_f1:.2f} | saved at {args.output_dir}")

if __name__ == "__main__":
  main()

