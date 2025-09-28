import json, csv, argparse, os, unicodedata, re, sys

def load_ids(test_path):
    with open(test_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    iterable = obj if isinstance(obj, list) else obj.get("data", [])
    ids = []
    for ex in iterable:
        sid = str(ex.get("id", ex.get("qid", ex.get("uid"))))
        ids.append(sid)
    return ids

def _pick_answer_field(it):
    # 盡量相容各家欄位名稱
    for key in ("answer", "prediction", "prediction_text", "text"):
        if key in it:
            return it[key]
    return ""

def load_pred(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return {str(k): ("" if v is None else str(v)) for k, v in obj.items()}
    d = {}
    for it in obj:
        sid = str(it.get("id", it.get("qid", it.get("uid"))))
        ans = _pick_answer_field(it)
        d[sid] = "" if ans is None else str(ans)
    return d

# 壓空白用
_whitespace_re = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    # 換行 / 製表 → 空白，避免 CSV 斷行
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.strip()
    if not s:
        return s
    # 全半形統一
    s = unicodedata.normalize("NFKC", s)
    # 連續空白壓成單一空白
    s = _whitespace_re.sub(" ", s)
    # 去除零寬字元（有時資料裡會藏）
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True, help="predictions.json（{id: answer} 或 list 格式）")
    ap.add_argument("--test", type=str, required=True, help="test.json（決定輸出列順序）")
    ap.add_argument("--out",  type=str, default="out/submission.csv")
    ap.add_argument("--no_normalize", action="store_true", help="不要正規化答案（預設會做輕量清洗）")
    args = ap.parse_args()

    ids = load_ids(args.test)
    pred = load_pred(args.pred)

    # 小提醒：缺漏/多餘的 id
    missing = [sid for sid in ids if sid not in pred]
    extra   = [sid for sid in pred.keys() if sid not in ids]
    if missing:
        print(f"[warn] {len(missing)} ids missing in predictions (將輸出空字串)。", file=sys.stderr)
    if extra:
        print(f"[info] {len(extra)} predictions have ids not in test（會被忽略）。", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 依作業規範輸出：表頭 id,answer
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "answer"])
        for sid in ids:
            ans = pred.get(sid, "")
            ans = ans if args.no_normalize else normalize_answer(ans)
            w.writerow([sid, ans])

    print(f"[OK] wrote CSV: {args.out} ({len(ids)} rows, header=id,answer)")

if __name__ == "__main__":
    main()
