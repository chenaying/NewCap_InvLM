#!/usr/bin/env bash
# Fix CRLF in shell scripts on Linux servers (run once after git pull on Windows).
set -e
cd "$(dirname "$0")/.."
for f in scripts/*.sh; do
  sed -i 's/\r$//' "$f"
  echo "fixed: $f"
done
echo "Done. Re-run training with: FUSION_TYPE=crossattn bash scripts/train_coco.sh 0"
