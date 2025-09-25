import json, csv, argparse, os, unicodedata, re

def load_ids(test_path):
    with open(test_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    iterable = obj if isinstance(obj, list) else obj.get("data", [])
    ids = []
    for ex in iterable:
        sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
        ids.append(sid)
    return ids

def load_pred(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 支援 {id: answer} 或 [{"id":..,"answer":..}]
    if isinstance(obj, dict):
        return {str(k): ("" if v is None else str(v)) for k, v in obj.items()}
    d = {}
    for it in obj:
        sid = str(it.get("id"))
        ans = it.get("answer", "")
        d[sid] = "" if ans is None else str(ans)
    return d

_whitespace_re = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    # 1) 去前後空白
    s = s.strip()
    if not s:
        return s
    # 2) 全半形/相容分解 → 統一到 NFKC
    s = unicodedata.normalize("NFKC", s)
    # 3) 連續空白壓成單一空白（保留標點，不做多餘清洗）
    s = _whitespace_re.sub(" ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True, help="predictions.json（{id: answer} 或 list 格式）")
    ap.add_argument("--test", type=str, required=True, help="test.json（決定輸出列順序）")
    ap.add_argument("--out",  type=str, default="out/submission.csv")
    args = ap.parse_args()

    ids = load_ids(args.test)
    pred = load_pred(args.pred)

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 作業規範：header 必須是 id,answer
        w.writerow(["id", "answer"])
        for sid in ids:
            ans = normalize_answer(pred.get(sid, ""))
            w.writerow([sid, ans])

    print(f"[OK] wrote CSV: {args.out} ({len(ids)} rows, header=id,answer)")

if __name__ == "__main__":
    main()
