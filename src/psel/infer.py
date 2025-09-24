# src/psel/infer.py
import json, argparse, pathlib

def char_overlap_score(q, p):
    qs = set(ch for ch in q if not ch.isspace())
    return sum((ch in p) for ch in qs)

def main(args):
    context = json.load(open(args.context, "r", encoding="utf-8"))  # list of paragraphs
    test = json.load(open(args.test, "r", encoding="utf-8"))        # list of {id, question, paragraphs}

    results = []
    for ex in test:
        qid = ex["id"]
        q = ex["question"]
        cand_pids = ex["paragraphs"]
        # Day1 dummy: 就挑第一個段落 id
        pid = cand_pids[0]
        ptext = context[pid]
        results.append({"id": qid, "question": q, "selected_paragraph": ptext})

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)   # data/context.json
    ap.add_argument("--test", required=True)      # data/test.json
    ap.add_argument("--out", required=True)       # out/selected.json
    ap.add_argument("--ckpt_dir", required=False)
    args = ap.parse_args()
    main(args)
