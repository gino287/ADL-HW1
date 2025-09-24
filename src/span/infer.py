import json, argparse, pathlib, re

def cheap_answer(paragraph: str, question: str) -> str:
    # 超簡單規則（只為 Day1 验流程）：
    # 1) 若段落內有中文/英文引號內容，取第一個引號內片段（<=20 字）
    # 2) 否則若有連續數字（年份等），取第一組
    # 3) 否則回傳段落前 8 個非空白字元
    for pat in [r"「([^」]{1,20})」", r"\"([^\"]{1,20})\"", r"\d{2,10}"]:
        m = re.search(pat, paragraph)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    text = "".join(ch for ch in paragraph if not ch.isspace())
    return text[:8] if text else ""

def main(args):
    sel = json.load(open(args.selected, "r", encoding="utf-8"))  # list[ {id, question, selected_paragraph} ]
    preds = []
    for ex in sel:
        pid = ex["id"]
        q = ex.get("question", "")
        ptext = ex.get("selected_paragraph", "")
        ans = cheap_answer(ptext, q)
        preds.append({"id": pid, "prediction_text": ans})

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)   # out/selected.json (list)
    ap.add_argument("--out", required=True)        # out/predictions.json (list of {id, prediction_text})
    ap.add_argument("--ckpt_dir", required=False)  # Day1 不用
    args = ap.parse_args()
    main(args)
