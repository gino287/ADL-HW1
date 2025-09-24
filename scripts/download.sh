#!/usr/bin/env bash
set -euo pipefail

# --- Helper: detect python ---
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

mkdir -p models

create_dummy_dir() {
  local dir="$1"
  mkdir -p "$dir"
  printf "Dummy ckpt for %s (Day1 skeleton)\n" "$dir" > "$dir/README.txt"
  cat > "$dir/meta.json" <<JSON
{"name":"$(basename "$dir")","version":"0.0.0","created_by":"download.sh"}
JSON
  echo "[dummy] created $dir/README.txt and $dir/meta.json"
}

download_zip_or_dummy() {
  local url="$1" zip="$2" dir="$3"
  if [[ -n "$url" ]]; then
    echo "[download] $dir from: $url"
    if command -v curl >/dev/null 2>&1; then
      curl -L "$url" -o "$zip" || { echo "[warn] curl failed, fallback to dummy"; create_dummy_dir "$dir"; return; }
    else
      "$PYTHON" - <<PY || { echo "[warn] python download failed, fallback to dummy"; create_dummy_dir "$dir"; return; }
import urllib.request
urllib.request.urlretrieve("$url", "$zip")
print("saved $zip")
PY
    fi
    # Try extract with Python zipfile; if fail, fallback to dummy
    mkdir -p "$dir"
    "$PYTHON" - <<PY || { echo "[warn] unzip failed, fallback to dummy"; rm -rf "$dir"; create_dummy_dir "$dir"; return; }
import zipfile,sys
with zipfile.ZipFile("$zip") as z: z.extractall("$dir")
print("extracted $zip -> $dir")
PY
  else
    create_dummy_dir "$dir"
  fi
}

download_zip_or_dummy "${PSEL_URL:-}" "psel-best.zip" "models/psel-best"
download_zip_or_dummy "${SPAN_URL:-}" "span-best.zip" "models/span-best"

echo "[ok] ckpt ready in models/psel-best and models/span-best"
