import os, math, argparse, random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)

from src.dataio.load_adl import ADLPselDataset, psel_collate


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dl, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        ttype     = batch["token_type_ids"].to(device)
        labels    = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttype).logits  # [B, C]
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total   += labels.numel()
    model.train()
    return correct / max(1, total)


def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--context", type=str, default="data/context.json")
    ap.add_argument("--train",   type=str, default="data/train.json")
    ap.add_argument("--valid",   type=str, default="data/valid.json")
    # model & out
    ap.add_argument("--model", type=str, default="hfl/chinese-roberta-wwm-ext",
                    help="預訓練骨幹（例：bert-base-chinese / hfl/chinese-roberta-wwm-ext）")
    ap.add_argument("--output_dir", type=str, default="models/psel-best")
    # train hyper-params
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2, help="每步的 micro-batch")
    ap.add_argument("--grad_accum", type=int, default=8, help="梯度累積步數（有效 batch = bs * grad_accum）")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    # eval / logging
    ap.add_argument("--eval_every", type=int, default=200, help="每幾個 update 做一次 valid 評估（0 表示每個 epoch 後評估）")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer & dataset
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_ds = ADLPselDataset(args.context, args.train, tok, max_length=args.max_len, split="train")
    valid_ds = ADLPselDataset(args.context, args.valid, tok, max_length=args.max_len, split="valid")

    train_dl = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        collate_fn=lambda b: psel_collate(b, tok, args.max_len)
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=max(2, args.bs), shuffle=False,
        collate_fn=lambda b: psel_collate(b, tok, args.max_len)
    )

    # model & optim & sched
    model = AutoModelForMultipleChoice.from_pretrained(args.model).to(device)
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

    # train loop
    best_acc, best_path = -1.0, os.path.join(args.output_dir, "best")
    global_step, running = 0, 0.0
    optim.zero_grad(set_to_none=True)

    for ep in range(1, args.epochs + 1):
        for step, batch in enumerate(train_dl, start=1):
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            ttype     = batch["token_type_ids"].to(device)
            labels    = batch["labels"].to(device)

            with autocast():
                out  = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttype, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            running += loss.item()

            if step % args.grad_accum == 0:
                # 梯度裁剪（以防不穩）
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
                    print(f"[epoch {ep}] update {global_step}/{tot_updates}  loss={running/20:.4f}  lr={lr_now:.2e}")
                    running = 0.0

                # 依據 eval_every 做中途評估
                if args.eval_every > 0 and (global_step % args.eval_every == 0):
                    acc = evaluate(model, valid_dl, device)
                    print(f"[eval@step {global_step}] valid MC-acc = {acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        model.save_pretrained(best_path)
                        tok.save_pretrained(best_path)
                        print(f"[save] new best acc={best_acc:.4f} → {best_path}")

        # 每個 epoch 結束後也評估一次（避免 eval_every 太大時沒有評估）
        if args.eval_every == 0:
            acc = evaluate(model, valid_dl, device)
            print(f"[eval@epoch {ep}] valid MC-acc = {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                model.save_pretrained(best_path)
                tok.save_pretrained(best_path)
                print(f"[save] new best acc={best_acc:.4f} → {best_path}")

    # 若整訓都沒觸發 best（理論上不會），至少存一份
    if best_acc < 0:
        model.save_pretrained(best_path)
        tok.save_pretrained(best_path)
        best_acc = 0.0

    print(f"[OK] finished training. best valid acc = {best_acc:.4f}, saved to {best_path}")


if __name__ == "__main__":
    main()
