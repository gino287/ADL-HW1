import json, csv, argparse, pathlib

def main(args):
    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction_text"])
        for p in preds:
            w.writerow([p["id"], p["prediction_text"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)  # out/predictions.json
    ap.add_argument("--out", required=True)   # prediction.csv
    args = ap.parse_args()
    main(args)
