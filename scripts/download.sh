#!/usr/bin/env bash
# Download two ckpt zips (or synthesize dummy zips) and extract to models/
set -euo pipefail
mkdir -p models

PSEL_ZIP="psel-best.zip"
SPAN_ZIP="span-best.zip"

have_curl() { command -v curl >/dev/null 2>&1; }

download_or_dummy () {
  local url="\" zip="\" label="\"
  if [[ -n "\" ]]; then
    echo "[download] \ from: \"
    if have_curl; then
      curl -L "\" -o "\"
    else
      python3 - <<PY
import urllib.request
urllib.request.urlretrieve("\", "\")
PY
    fi
  else
    echo "[dummy] make \ zip locally"
    python3 - <<PY
import zipfile, json
with zipfile.ZipFile("\", "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("README.txt", "Dummy ckpt for Day1 skeleton.")
    z.writestr("meta.json", json.dumps({"name": "\", "version": "0.0.0"}))
print("created", "\")
PY
  fi
}

download_or_dummy "\" "\" "psel-best"
download_or_dummy "\" "\" "span-best"

python3 - <<PY
import zipfile, os
os.makedirs("models/psel-best", exist_ok=True)
os.makedirs("models/span-best", exist_ok=True)
with zipfile.ZipFile("psel-best.zip") as z: z.extractall("models/psel-best")
with zipfile.ZipFile("span-best.zip") as z: z.extractall("models/span-best")
print("extracted to models/psel-best and models/span-best")
PY

echo "[ok] download/extract complete."
