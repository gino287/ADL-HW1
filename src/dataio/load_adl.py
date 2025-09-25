from __future__ import annotations
import json, torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class ADLPselDataset(Dataset):
    """把 HW1 的段落選擇資料轉成 Multiple Choice 格式。"""
    def __init__(self, context_path: str, qa_path: str, tokenizer, max_length: int = 512, split: str = "train"):
        self.tok = tokenizer
        self.L = max_length
        self.split = split

        with open(context_path, "r", encoding="utf-8") as f:
            self.ctx = json.load(f)  # {pid: paragraph_text}

        with open(qa_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        for ex in raw:
            qid = ex.get("id")
            q = ex.get("question")
            para_ids: List[str] = ex.get("paragraphs") or ex.get("paragraph_ids") or []
            choices: List[str] = [self.ctx[pid] for pid in para_ids]

            label = None
            rel = ex.get("relevant")
            if rel is not None and rel in para_ids:
                label = para_ids.index(rel)

            self.samples.append({
                "id": qid,
                "question": q,
                "choices": choices,
                "label": -1 if label is None else label,
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def psel_collate(batch: List[Dict[str, Any]], tokenizer, max_length=512):
    """組成模型要的 [B, C, L] tensors。"""
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    ids, questions, choices_text = [], [], []

    for ex in batch:
        q = ex["question"]; ch = ex["choices"]
        enc = tokenizer([q]*len(ch), ch,
                        truncation=True, padding="max_length",
                        max_length=max_length, return_token_type_ids=True,
                        return_tensors="pt")
        input_ids.append(enc["input_ids"])               # [C, L]
        attention_mask.append(enc["attention_mask"])     # [C, L]
        token_type_ids.append(enc["token_type_ids"])     # [C, L]
        labels.append(ex["label"])
        ids.append(ex["id"]); questions.append(q); choices_text.append(ch)

    batch_input_ids   = torch.stack(input_ids, dim=0)    # [B, C, L]
    batch_attention   = torch.stack(attention_mask, 0)   # [B, C, L]
    batch_token_type  = torch.stack(token_type_ids, 0)   # [B, C, L]
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "token_type_ids": batch_token_type,
        "labels": labels,
        "ids": ids,
        "questions": questions,
        "choices_text": choices_text,
    }
