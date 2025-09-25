ADL HW1 — Chinese Extractive QA
專案簡介

這個專案是 ADL HW1 的骨架專案，主題為 Chinese Extractive QA。
整個流程分兩階段：

Paragraph Selection (psel)：從 context.json 中挑出最相關的段落

Span Selection (span)：在挑出的段落中抽取答案

最終輸出一個 submission.csv，格式：

id,prediction_text


檔案架構
adl-hw1/
  scripts/
    download.sh        # 線上階段（下載模型）
    run.sh             # 離線階段（整個 pipeline）
  src/
    dataio/load_adl.py # Day2 新增，資料轉換成多選題格式
    psel/train.py      # 段落選擇訓練程式 (BERT 多選題)
    psel/infer.py      # 段落選擇推論程式
    span/infer.py      # 片段抽取 (目前還是 dummy)
    utils/format_kaggle.py # 轉成 submission.csv
  models/
    psel-best/         # 訓練好的段落選擇模型 (Day2 存這裡)
  data/
    context.json  
    train.json
    valid.json
    test.json     
  out/
    selected.json      # psel 輸出 (段落選擇結果)
    predictions.json   # span 輸出 (答案抽取結果)
    submission.csv     # Kaggle 最終上傳檔
  report/figures/
  README.md
  requirements.txt


  使用方式
線上階段 (下載模型)
bash scripts/download.sh
test -f models/psel-best/config.json && test -f models/span-best/meta.json && echo "[ok] download stage passed"

離線階段 (完整 pipeline)
bash scripts/run.sh data/context.json data/test.json out/submission.csv


流程會依序做：

段落選擇 (psel) → 讀 context.json 和 test.json，挑一段 → out/selected.json

片段抽取 (span) → 在選段裡抓答案 → out/predictions.json

格式轉換 → 轉成 Kaggle 要的 submission.csv → out/submission.csv


Day2 完成進度

新增 load_adl.py → 把 (question, paragraph) 轉成 多選題格式 [batch, num_choices, seq_len]

訓練：在 Colab (T4 GPU) 跑 1 epoch BERT (bert-base-chinese)，存到 models/psel-best/

推論：完成 psel/infer.py，能在 test.json 上輸出 out/selected.json

run.sh 已可用真正的 BERT 段落選擇模型替代 dummy，pipeline 跑通到 submission.csv

環境需求

Python 3.10

安裝套件：

transformers==4.50.0
accelerate==0.34.2
datasets==3.0.1
evaluate==0.4.3
tqdm==4.66.4


安裝方式：

pip install -r requirements.txt