# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math, json, argparse, random, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)

# ========== utils ==========

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_id_and_question(ex: dict) -> Tuple[str, str]:
    sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
    q = str(ex.get("question", ex.get("q", ""))).strip()
    return sid, q

def extract_answer_texts(ex: dict) -> List[str]:
    # 支援多種答案欄位格式
    if isinstance(ex.get("answer"), dict) and isinstance(ex["answer"].get("text"), str):
        t = ex["answer"]["text"].strip()
        return [t] if t else []
    if isinstance(ex.get("answer"), str):
        t = ex["answer"].strip()
        return [t] if t else []
    a2 = ex.get("answers")
    if isinstance(a2, dict) and a2.get("text"):
        if isinstance(a2["text"], list):
            return [str(t).strip() for t in a2["text"] if str(t).strip()]
        else:
            t = str(a2["text"]).strip()
            return [t] if t else []
    if isinstance(a2, list) and a2:
        outs = []
        for a in a2:
            if isinstance(a, dict) and "text" in a:
                t = str(a["text"]).strip()
                if t: outs.append(t)
        return outs
    return []

def get_para_text_by_id(pid, context_db) -> Optional[str]:
    if isinstance(context_db, dict):
        return context_db.get(str(pid)) or context_db.get(pid)
    if isinstance(context_db, list):
        try: return context_db[int(pid)]
        except Exception: return None
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

@dataclass
class QAExample:
    id: str
    question: str
    context: str
    answer_text: str
    answer_start: int  # char idx
    answer_end: int    # char idx (exclusive)

def _first_hit_span(contexts: List[str], answers: List[str]) -> Optional[Tuple[str,int,int,str]]:
    """在多個候選段中找出第一個包含任一答案字串的 (ctx, s, e, ans)。"""
    for a in answers:
        if not a: 
            continue
        for c in contexts:
            pos = c.find(a)
            if pos != -1:
                return c, pos, pos + len(a), a
    return None

def build_examples(split_path: str,
                   context_path: str,
                   psel_kv: Optional[Dict[str, Dict[str, List[str]]]] = None,
                   use_psel_only: bool = False) -> List[QAExample]:
    """優先用 psel 選出的段落（文字串），否則回退到原始的 paragraphs/contexts/context。"""
    data = load_json(split_path)
    ctx_db = load_json(context_path)
    iterable = data if isinstance(data, list) else data.get("data", [])
    out: List[QAExample] = []
    miss = 0

    for ex in iterable:
        sid, q = get_id_and_question(ex)
        answers = extract_answer_texts(ex)
        if not answers:
            miss += 1
            continue

        # 先嘗試用 psel 選出來的
        cands: List[str] = []
        if psel_kv is not None and sid in psel_kv:
            cands = list(psel_kv[sid].get("paragraphs", []) or [])

        # 不夠就回退到原始欄位
        if (not cands) and (not use_psel_only):
            if "paragraphs" in ex and isinstance(ex["paragraphs"], list):
                cands = normalize_para_ids_to_texts(ex["paragraphs"], ctx_db)
            elif isinstance(ex.get("contexts"), list):
                cands = normalize_para_ids_to_texts(ex["contexts"], ctx_db)
            elif isinstance(ex.get("context"), str):
                cands = [ex["context"]]

        # relevant（id/索引）在回退時優先
        if (not cands) and (not use_psel_only) and ("relevant" in ex):
            t = get_para_text_by_id(ex["relevant"], ctx_db)
            if isinstance(t, str):
                cands = [t]

        hit = _first_hit_span(cands, answers)
        if not hit:
            miss += 1
            continue

        ctx, s, e, a = hit
        out.append(QAExample(sid, q, ctx, a, s, e))

    print(f"[build] {split_path}: {len(out)} examples  (miss {miss})")
    return out

# ========== dataset ==========

class SpanQADataset(Dataset):
    def __init__(self,
                 items: List[QAExample],
                 tokenizer: AutoTokenizer,
                 max_length: int,
                 doc_stride: int,
                 is_train: bool = True,
                 max_answer_length: int = 64):
        self.items = items
        self.tok = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.is_train = is_train
        self.max_answer_length = max_answer_length

        self.enc = self.tok(
            [it.question for it in items],
            [it.context  for it in items],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        self.sample_mapping = self.enc.pop("overflow_to_sample_mapping")
        self.offset_mapping = self.enc["offset_mapping"]  # list of list[(start,end)]

        # 建 label（僅訓練用）
        if is_train:
            self.starts, self.ends = self._create_labels()

    def __len__(self): 
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items() if k != "offset_mapping"}
        if self.is_train:
            item["start_positions"] = torch.tensor(self.starts[i])
            item["end_positions"]   = torch.tensor(self.ends[i])
        return item

    def _create_labels(self):
        starts, ends = [], []
        for i in range(len(self.enc["input_ids"])):
            smp_idx = self.sample_mapping[i]
            it      = self.items[smp_idx]
            offsets = self.offset_mapping[i]
            seq_ids = self.enc.sequence_ids(i)  # None/0(question)/1(context)

            # 僅考慮 context 的 token，避免 special/question 影響
            valid_idx = [k for k, sid in enumerate(seq_ids) if sid == 1 and offsets[k] is not None]
            if not valid_idx:
                cls_index = self.enc["input_ids"][i].index(self.tok.cls_token_id) \
                            if self.tok.cls_token_id in self.enc["input_ids"][i] else 0
                starts.append(cls_index); ends.append(cls_index); continue

            start_char, end_char = it.answer_start, it.answer_end
            start_tok, end_tok = None, None

            for k in valid_idx:
                s_off, e_off = offsets[k]
                if s_off is None or e_off is None: 
                    continue
                if start_tok is None and s_off <= start_char < e_off:
                    start_tok = k
                if end_tok is None and s_off < end_char <= e_off:
                    end_tok = k
                # 極短答案可能被單一 token 完整包住
                if s_off <= start_char and end_char <= e_off:
                    start_tok = end_tok = k
                    break

            if start_tok is None or end_tok is None:
                cls_index = self.enc["input_ids"][i].index(self.tok.cls_token_id) \
                            if self.tok.cls_token_id in self.enc["input_ids"][i] else 0
                starts.append(cls_index); ends.append(cls_index); continue

            # answer 長度上限
            if end_tok - start_tok + 1 > self.max_answer_length:
                end_tok = start_tok + self.max_answer_length - 1
                end_tok = max(end_tok, start_tok)

            starts.append(start_tok); ends.append(end_tok)
        return starts, ends

# ========== metrics ==========

def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def compute_em_f1(pred: str, golds: List[str]) -> Tuple[float, float]:
    pred_n = _normalize_text(pred)
    golds_n = [_normalize_text(g) for g in golds if g is not None]
    if any(pred_n == g for g in golds_n):
        return 1.0, 1.0
    def to_units(x: str): return list(x)  # 中文用逐字
    p_set = to_units(pred_n)
    best_f1 = 0.0
    for g in golds_n:
        g_set = to_units(g)
        common = 0
        used = [False]*len(g_set)
        for ch in p_set:
            for j, gh in enumerate(g_set):
                if (not used[j]) and gh == ch:
                    used[j] = True
                    common += 1
                    break
        if common == 0:
            f1 = 0.0
        else:
            prec = common / max(1, len(p_set))
            rec  = common / max(1, len(g_set))
            f1 = 2*prec*rec/(prec+rec)
        best_f1 = max(best_f1, f1)
    return 0.0, best_f1

@torch.no_grad()
def evaluate(model, tok, dataset: SpanQADataset, batch_size: int, device, max_answer_length: int) -> Tuple[float, float]:
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample_map = dataset.sample_mapping
    offsets    = dataset.offset_mapping
    items      = dataset.items

    best_for_id: Dict[str, Tuple[float, str]] = {}
    for batch_idx, batch in enumerate(dl):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids, return_dict=True)
        s_logits = out.start_logits.detach().cpu()
        e_logits = out.end_logits.detach().cpu()

        B = s_logits.size(0)
        for j in range(B):
            gidx = batch_idx*dl.batch_size + j
            smp  = sample_map[gidx]
            it   = items[smp]
            off  = offsets[gidx]
            # 只挑有 offset 的 token，避免選到 special
            valid_idx = [k for k, (s_off, e_off) in enumerate(off) if (s_off is not None and e_off is not None and not (s_off == 0 and e_off == 0))]
            if not valid_idx:
                continue

            # 取前k名組合尋優
            k = min(20, len(off))
            s_top = torch.topk(s_logits[j][valid_idx], k=min(k, len(valid_idx))).indices.tolist()
            e_top = torch.topk(e_logits[j][valid_idx], k=min(k, len(valid_idx))).indices.tolist()
            s_top = [valid_idx[idx] for idx in s_top]
            e_top = [valid_idx[idx] for idx in e_top]

            best_s, best_e, best_score = 0, 0, -1e9
            for s in s_top:
                for e in e_top:
                    if e < s: continue
                    if e - s + 1 > max_answer_length: continue
                    score = s_logits[j][s].item() + e_logits[j][e].item()
                    if score > best_score:
                        best_s, best_e, best_score = s, e, score

            s_char, _ = off[best_s]
            _, e_char = off[best_e]
            pred_text = it.context[s_char:e_char].strip()
            old = best_for_id.get(it.id, (-1e9, ""))
            if best_score > old[0]:
                best_for_id[it.id] = (best_score, pred_text)

    em_sum, f1_sum, n = 0.0, 0.0, 0
    for it in items:
        if it.id not in best_for_id:
            continue
        pred = best_for_id[it.id][1]
        _, f1 = compute_em_f1(pred, [it.answer_text])
        em = 1.0 if _normalize_text(pred) == _normalize_text(it.answer_text) else 0.0
        em_sum += em; f1_sum += f1; n += 1
    model.train()
    return em_sum / max(1, n), f1_sum / max(1, n)

# ========== train ==========

def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--valid", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    # optional: psel selected（KV 格式：{id: {"paragraphs": [...]}}）
    ap.add_argument("--psel_selected", type=str, default=None,
                    help="使用段落選結果來限制候選（建議給 out/psel_selected_all.json）")
    ap.add_argument("--use_psel_only", action="store_true",
                    help="只用 psel 段落，不回退到原始 paragraphs/contexts/context")
    # model
    ap.add_argument("--model", type=str, default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--output_dir", type=str, default="models/span-best")
    # hparams
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--max_answer_length", type=int, default=64)
    # misc
    ap.add_argument("--eval_every", type=int, default=1000)  # 0 = 每個 epoch 評估一次
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀 psel 選段（可選）
    psel_kv = load_json(args.psel_selected) if args.psel_selected else None

    # tokenizer & data
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_items = build_examples(args.train, args.context, psel_kv, args.use_psel_only)
    valid_items = build_examples(args.valid, args.context, psel_kv, args.use_psel_only)

    train_ds = SpanQADataset(train_items, tok, args.max_length, args.doc_stride, is_train=True,  max_answer_length=args.max_answer_length)
    valid_ds = SpanQADataset(valid_items, tok, args.max_length, args.doc_stride, is_train=False, max_answer_length=args.max_answer_length)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=args.num_workers)
    # eval 用較大的 batch
    valid_dl = DataLoader(valid_ds, batch_size=max(2, args.bs), shuffle=False, num_workers=args.num_workers)

    # model / optim / sched
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = AdamW(param_groups, lr=args.lr)

    num_update_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    tot_updates = args.epochs * num_update_per_epoch
    warmups = int(tot_updates * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmups, num_training_steps=tot_updates)

    scaler = GradScaler()
    model.train()

    best_f1, best_path = -1.0, os.path.join(args.output_dir, "best")

    global_step, running = 0, 0.0
    optim.zero_grad(set_to_none=True)

    for ep in range(1, args.epochs + 1):
        for step, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            ttype     = batch.get("token_type_ids")
            if ttype is None:
                ttype = torch.zeros_like(input_ids)
            ttype     = ttype.to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos   = batch["end_positions"].to(device)

            with autocast():
                out  = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    token_type_ids=ttype,
                    start_positions=start_pos,
                    end_positions=end_pos,
                    return_dict=True
                )
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            running += loss.item()

            if step % args.grad_accum == 0:
                scaler.unscale_(optim)
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if global_step % 20 == 0:
                    lr_now = sched.get_last_lr()[0]
                    print(f"[epoch {ep}] update {global_step}/{tot_updates} loss={running/20:.4f} lr={lr_now:.2e}")
                    running = 0.0

                if args.eval_every > 0 and (global_step % args.eval_every == 0):
                    em, f1 = evaluate(model, tok, valid_ds, max(2, args.bs), device, args.max_answer_length)
                    print(f"[eval@step {global_step}] valid EM={em:.4f} F1={f1:.4f}")
                    if f1 > best_f1:
                        best_f1 = f1
                        model.save_pretrained(best_path)
                        tok.save_pretrained(best_path)
                        print(f"[save] new best F1={best_f1:.4f} → {best_path}")

        if args.eval_every == 0:
            em, f1 = evaluate(model, tok, valid_ds, max(2, args.bs), device, args.max_answer_length)
            print(f"[eval@epoch {ep}] valid EM={em:.4f} F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(best_path)
                tok.save_pretrained(best_path)
                print(f"[save] new best F1={best_f1:.4f} → {best_path}")

    if best_f1 < 0:
        # 至少留一份
        model.save_pretrained(best_path)
        tok.save_pretrained(best_path)
        best_f1 = 0.0

    print(f"[OK] finished training. best valid F1={best_f1:.4f}, saved to {best_path}")

if __name__ == "__main__":
    main()
