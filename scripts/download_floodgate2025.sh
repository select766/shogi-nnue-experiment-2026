#!/usr/bin/env bash
# Resume only the .part file; never replace an existing archive with different bytes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ARCHIVE="dataset/floodgate2025/raw/wdoor2025.7z"
EXPECTED="423504903f211316ec40f9c3d8dcc2e5faf392fc4d1334ede1489242a9d09baa"
URL="https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor2025.7z"
mkdir -p dataset/floodgate2025/raw
if [[ -e "$ARCHIVE" ]]; then
    printf '%s  %s\n' "$EXPECTED" "$ARCHIVE" | sha256sum --check
    exit 0
fi
curl --fail --location --connect-timeout 30 --retry 3 --continue-at - \
    --dump-header dataset/floodgate2025/raw/download-headers.txt \
    --output "$ARCHIVE.part" "$URL"
printf '%s  %s\n' "$EXPECTED" "$ARCHIVE.part" | sha256sum --check
mv -- "$ARCHIVE.part" "$ARCHIVE"
