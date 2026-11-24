#!/usr/bin/env bash
# Downloads SubPipe archives from Zenodo (record 12666132) into data/raw, resuming if interrupted, and checks each MD5.
# Usage: scripts/download_subpipe.sh SubPipeMini.zip [SubPipeMini2.zip ...]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
meta=$(curl -sf --retry 3 "https://zenodo.org/api/records/12666132")

for name in "$@"; do
  read -r url md5 < <(python3 -c '
import json, sys
f = next(f for f in json.loads(sys.argv[1])["files"] if f["key"] == sys.argv[2])
print(f["links"]["self"], f["checksum"].split(":")[1])' "$meta" "$name")
  echo "Downloading $name"
  curl -fL -C - --retry 5 -o "data/raw/$name" "$url"
  got=$(md5 -q "data/raw/$name")
  if [ "$got" = "$md5" ]; then echo "$name: checksum OK"; else echo "$name: checksum mismatch ($got vs $md5)" >&2; exit 1; fi
done
