專案簡介

ADL HW1 的骨架專案 
主題: Chinese Extractive QA

流程分兩階段

階段1:Paragraph Selection, psel (挑文章)
階段2:Span Selection, span (找答案)
輸出:submission.csv (欄位:id,prediction_text)

檔案架構
adl-hw1/
  scripts/
    download.sh   # 線上階段
    run.sh        # 離線階段
  src/
    psel/infer.py       # 段落選擇
    span/infer.py       # 片段抽取
    utils/format_kaggle.py # 轉成 submission.csv
  models/         # download.sh 會解壓到這裡
  data/
    context.json  
    test.json     
  out/            # 輸出檔案
  report/figures/ # 可放示圖
  README.md
  requirements.txt


使用方式


線上階段
# Online（可連網）
bash scripts/download.sh
test -f models/psel-best/meta.json && test -f models/span-best/meta.json && echo "[ok] download stage passed"

離線階段
bash scripts/run.sh
依序做
1段落選擇：讀 data/context.json 和 data/test.json，挑一段 → out/selected.json

2片段抽取：在選段裡抓答案 → out/predictions.json

3格式轉換：轉成 Kaggle 需要的 → out/submission.csv


環境需求

Python 3.10

需要安裝的套件：

transformers==4.50.0

accelerate==0.34.2

datasets==3.0.1

evaluate==0.4.3

tqdm==4.66.4
安裝:
pip install -r requirements.txt
