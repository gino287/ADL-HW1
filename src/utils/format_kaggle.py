import json, csv, argparse

def load_ids(test_path):
    obj = json.load(open(test_path,"r",encoding="utf-8"))
    iterable = obj if isinstance(obj, list) else obj["data"]
    return [str(ex.get("id", ex.get("qid", ex.get("uid")))) for ex in iterable]

def load_pred(pred_path):
    obj = json.load(open(pred_path,"r",encoding="utf-8"))
    # 支援 {id: answer} 或 [{"id":..,"answer":..}]
    if isinstance(obj, dict):
        return obj
    d = {}
    for it in obj:
        d[str(it["id"])] = it.get("answer","")
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--out",  type=str, default="out/submission.csv")
    args = ap.parse_args()

    ids = load_ids(args.test)
    pred = load_pred(args.pred)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","answer"])
        for sid in ids:
            w.writerow([sid, pred.get(sid,"")])
    print(f"[OK] wrote CSV: {args.out} ({len(ids)} rows)")

if __name__ == "__main__":
    main()
