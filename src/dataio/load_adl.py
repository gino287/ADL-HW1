from __future__ import annotations
import json, random, torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class ADLPselDataset(Dataset):
    """
    把 HW1 的段落選擇資料轉成 Multiple Choice 格式。
    - context_file: 一個「list[str]」的 JSON，每個索引即為 pid（0-based 或資料中使用的整數）。
    - qa_file:      list[{"id","question","paragraphs":[pid...],"relevant": pid 或 None, ...}]
    - fixed_k:      若指定，會保留正解並從其餘候選隨機抽到固定個數（不足則全用）。
    """
    def __init__(
        self,
        context_path: str,
        qa_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        fixed_k: Optional[int] = None,
        rng_seed: int = 42,
    ):
        self.tok = tokenizer
        self.L = max_length
        self.split = split
        self.fixed_k = fixed_k

        # 讀 context（list[str]）
        with open(context_path, "r", encoding="utf-8") as f:
            ctx_list = json.load(f)
        if not isinstance(ctx_list, list):
            raise ValueError("context.json 應該是 list[str]，但讀到的不是 list。")
        # 直接用整數 pid -> text 的 mapping
        self.ctx_by_id: Dict[int, str] = {i: s for i, s in enumerate(ctx_list)}

        # 讀 QA
        with open(qa_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("train/valid 應該是 list[dict] 格式。")

        # 固定隨機種子（抽負例用）
        self.rng = random.Random(rng_seed)

        self.samples = []
        for ex in raw:
            qid = ex.get("id")
            q = ex.get("question", "")
            para_ids: List[int] = ex.get("paragraphs") or []
            # 轉成文字，pid 可能不連續，但必須存在於 context
            choices_full: List[str] = []
            valid_indices: List[int] = []
            for pid in para_ids:
                if pid in self.ctx_by_id:
                    choices_full.append(self.ctx_by_id[pid])
                    valid_indices.append(pid)

            # 找正解（可能沒有）
            rel = ex.get("relevant", None)
            gold_idx_in_full = None
            if rel is not None and rel in para_ids:
                # 用在 "full" 內的索引
                try:
                    gold_idx_in_full = para_ids.index(rel)
                except ValueError:
                    gold_idx_in_full = None

            # 如果有固定 K，做「保留正解 + 抽負例」
            if self.fixed_k is not None and len(choices_full) > 0:
                if (gold_idx_in_full is not None) and (0 <= gold_idx_in_full < len(choices_full)):
                    negatives = [i for i in range(len(choices_full)) if i != gold_idx_in_full]
                    need_neg = max(0, self.fixed_k - 1)
                    if len(negatives) > need_neg:
                        chosen_negs = self.rng.sample(negatives, need_neg)
                    else:
                        chosen_negs = negatives
                    pick_indices = [gold_idx_in_full] + chosen_negs
                    # 為了不把正解固定在 0 位，打亂後再算 label
                    self.rng.shuffle(pick_indices)
                    choices = [choices_full[i] for i in pick_indices]
                    label = pick_indices.index(gold_idx_in_full)
                else:
                    # 沒正解 或正解不在 choices_full 裡：只能截斷/抽樣前 fixed_k 個，label = -1
                    if len(choices_full) > self.fixed_k:
                        # 沒有正解時就隨機抽 fixed_k
                        pick_indices = list(range(len(choices_full)))
                        self.rng.shuffle(pick_indices)
                        pick_indices = pick_indices[: self.fixed_k]
                        choices = [choices_full[i] for i in pick_indices]
                    else:
                        choices = choices_full
                    label = -1
            else:
                # 不固定 K：全用
                choices = choices_full
                if gold_idx_in_full is not None:
                    label = gold_idx_in_full
                else:
                    label = -1

            # 保底：至少要有一個選項
            if len(choices) == 0:
                # 放一個空字串佔位，避免崩潰
                choices = ["（空段落）"]
                label = -1

            self.samples.append({
                "id": qid,
                "question": q,
                "choices": choices,
                "label": int(label),
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def psel_collate(batch: List[Dict[str, Any]], tokenizer, max_length=512):
    """
    組成模型要的 [B, C, L] tensors。
    兼容沒有 token_type_ids 的 tokenizer（例如 RoBERTa 類）。
    """
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    ids, questions, choices_text = [], [], []

    for ex in batch:
        q = ex["question"]
        ch = ex["choices"]
        enc = tokenizer(
            [q] * len(ch),
            ch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        input_ids.append(enc["input_ids"])               # [C, L]
        attention_mask.append(enc["attention_mask"])     # [C, L]
        if "token_type_ids" in enc:
            token_type_ids.append(enc["token_type_ids"])
        else:
            # 沒有 token_type_ids：補零
            token_type_ids.append(torch.zeros_like(enc["input_ids"]))
        labels.append(ex["label"])
        ids.append(ex.get("id"))
        questions.append(q)
        choices_text.append(ch)

    batch_input_ids  = torch.stack(input_ids, dim=0)       # [B, C, L]
    batch_attention  = torch.stack(attention_mask, 0)      # [B, C, L]
    batch_token_type = torch.stack(token_type_ids, 0)      # [B, C, L]
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
