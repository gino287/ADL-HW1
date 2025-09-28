# load_adl.py
from __future__ import annotations
import json, torch, random
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class ADLPselDataset(Dataset):
    """把 HW1 的段落選擇資料轉成 Multiple Choice 格式。"""
    def __init__(
        self,
        context_path: str,
        qa_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        fixed_k: Optional[int] = None,          # ← 選配：固定 K 候選
        neg_sampler: Optional[str] = None,      # ← 選配：'random' / 'bm25' / 'embed'
        seed: int = 42
    ):
        self.tok = tokenizer
        self.L = max_length
        self.split = split
        self.fixed_k = fixed_k
        self.neg_sampler = neg_sampler
        random.seed(seed)

        with open(context_path, "r", encoding="utf-8") as f:
            self.ctx = json.load(f)  # {pid: paragraph_text}

        with open(qa_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 若要做固定 K，需要全域可取的所有 pid
        self.all_pids = list(self.ctx.keys())

        self.samples = []
        for ex in raw:
            qid = ex.get("id")
            q = ex.get("question")
            para_ids: List[str] = ex.get("paragraphs") or ex.get("paragraph_ids") or []
            choices: List[str] = [self.ctx[pid] for pid in para_ids]

            # 標註正解（可能在 valid/test 沒有）
            label = -1
            rel = ex.get("relevant")
            if rel is not None and rel in para_ids:
                label = para_ids.index(rel)

            # 選配：固定 K 候選（示範 random，之後可換 BM25/embedding）
            if self.fixed_k is not None and len(choices) != self.fixed_k:
                choices, label = self._pad_or_trim_choices(para_ids, label, self.fixed_k)

            # 防呆
            if label != -1:
                assert 0 <= label < len(choices), f"label {label} out of range C={len(choices)} for qid={qid}"
            assert len(choices) >= 2, f"choices too few for qid={qid}"

            self.samples.append({
                "id": qid,
                "question": q,
                "choices": choices,
                "label": label,
            })

    def _pad_or_trim_choices(self, para_ids: List[str], label: int, K: int):
        """把候選修成固定 K：若不足，補負例；若過多，隨機下采樣（保留正例）"""
        choices_pids = para_ids[:]
        # 先確保正例在
        if label == -1:
            # 沒標 label 就隨機指定一個為「準正例」（train 可用，val/test 保持 -1）
            pass
        else:
            pos_pid = para_ids[label]

        # 補到 K
        while len(choices_pids) < K:
            # 簡單 random：之後可換 BM25 / embedding 相似
            pid = random.choice(self.all_pids)
            if pid not in choices_pids:
                choices_pids.append(pid)

        # 若超過 K，保留正例 + 隨機抽負例湊 K
        if len(choices_pids) > K:
            if label != -1:
                pos_pid = para_ids[label]
                neg_pool = [p for p in choices_pids if p != pos_pid]
                sampled_negs = random.sample(neg_pool, K - 1)
                choices_pids = [pos_pid] + sampled_negs
                random.shuffle(choices_pids)
                label = choices_pids.index(pos_pid)
            else:
                choices_pids = random.sample(choices_pids, K)
                label = -1

        choices = [self.ctx[pid] for pid in choices_pids]
        return choices, label

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def psel_collate(batch: List[Dict[str, Any]], tokenizer, max_length=512):
    """組成模型要的 [B, C, L] tensors。"""
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    ids, questions, choices_text = [], [], []

    for ex in batch:
        q = ex["question"]; ch = ex["choices"]
        enc = tokenizer([q]*len(ch), ch,
                        truncation="only_second",        # ← 關鍵：只截斷段落
                        padding="max_length",
                        max_length=max_length,
                        return_token_type_ids=True,
                        return_tensors="pt")
        input_ids.append(enc["input_ids"])               # [C, L]
        attention_mask.append(enc["attention_mask"])     # [C, L]
        # 有些 tokenizer 可能沒有 token_type_ids，視模型而定
        token_type_ids.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
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
