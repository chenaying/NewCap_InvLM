#!/usr/bin/env bash
# Fix Windows CRLF line endings in shell scripts (run on Linux after git pull).
set -e
cd "$(dirname "$0")/.."
fixed=0
for f in scripts/*.sh; do
  if grep -q $'\r' "$f" 2>/dev/null; then
    sed -i 's/\r$//' "$f"
    echo "fixed CRLF: $f"
    fixed=$((fixed + 1))
  fi
done
if [[ "$fixed" -eq 0 ]]; then
  echo "All scripts/*.sh already use LF line endings."
else
  echo "Fixed $fixed file(s). You can now run train/eval scripts."
fi
