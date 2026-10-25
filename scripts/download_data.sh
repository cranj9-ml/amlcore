#!/usr/bin/env bash
# Download one AMLworld variant (transactions + laundering patterns) into data/raw.
#
# Uses the kaggle CLI, which needs an API token: on kaggle.com go to Settings > API >
# Create New Token, and save it as ~/.kaggle/access_token (or a legacy kaggle.json as
# ~/.kaggle/kaggle.json, with chmod 600).
# Kaggle often serves large single files zipped, so any <file>.zip is extracted and
# then deleted. Files that already exist are skipped unless --force is given.
#
# Usage: scripts/download_data.sh [VARIANT] [--force]     (VARIANT defaults to HI-Small)

set -euo pipefail

DATASET="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
VARIANT="HI-Small"
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) VARIANT="$arg" ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw"

# Prefer the project venv's kaggle CLI, so the script works without activating the venv.
KAGGLE="$PROJECT_ROOT/.venv/bin/kaggle"
[[ -x "$KAGGLE" ]] || KAGGLE="kaggle"

mkdir -p "$RAW_DIR"

for file_name in "${VARIANT}_Trans.csv" "${VARIANT}_Patterns.txt"; do
    target="$RAW_DIR/$file_name"
    if [[ -f "$target" && "$FORCE" -eq 0 ]]; then
        echo "Already present, skipping: $file_name"
        continue
    fi

    echo "Downloading $file_name ..."
    "$KAGGLE" datasets download "$DATASET" -f "$file_name" -p "$RAW_DIR" --force

    if [[ -f "$target.zip" ]]; then
        echo "Unzipping $file_name.zip ..."
        unzip -o -q "$target.zip" -d "$RAW_DIR"
        rm "$target.zip"
    fi

    if [[ ! -f "$target" ]]; then
        echo "Expected $target after download, but it is missing" >&2
        exit 1
    fi
done

echo "Done. Files are in $RAW_DIR"
