#!/usr/bin/env bash
# Expert Blending 版やねうら王の起動ラッパ。
#
# 将棋 GUI (ShogiGUI / 将棋所など) は USI エンジンに環境変数を渡す機能を
# 持たないことが多い。一方 onnxruntime のリンクには `LD_LIBRARY_PATH` で
# `libonnxruntime.so.1` を見つけられる必要があるため、本スクリプト経由で
# 環境変数を補ってから本体バイナリを exec する。
#
# GUI 側ではこの .sh を「エンジンの実行ファイル」として登録すればよい。
#
# 任意の CWD で呼ばれても破綻しないよう、スクリプト自身の絶対パスから
# リポジトリ内の各種ディレクトリを解決する。

set -euo pipefail

# このスクリプト自身の絶対パス (symlink 経由で呼ばれても展開する)。
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# repo レイアウト前提:
#   <repo>/bin/YaneuraOu-expert-blending.sh   (本スクリプト)
#   <repo>/bin/YaneuraOu-expert-blending      (実行バイナリ)
#   <repo>/YaneuraOu/extra/onnxruntime/linux/current/lib/libonnxruntime.so.*
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE="$SCRIPT_DIR/YaneuraOu-expert-blending"
ONNX_LIB_DIR="$REPO_DIR/YaneuraOu/extra/onnxruntime/linux/current/lib"

if [[ ! -x "$ENGINE" ]]; then
    echo "Error: engine binary not found or not executable: $ENGINE" >&2
    exit 1
fi
if [[ ! -d "$ONNX_LIB_DIR" ]]; then
    echo "Error: onnxruntime lib dir not found: $ONNX_LIB_DIR" >&2
    echo "  まず YaneuraOu/extra/onnxruntime/fetch_onnxruntime.sh を実行してください。" >&2
    exit 1
fi

# 既存の LD_LIBRARY_PATH を尊重しつつ先頭に追加。
export LD_LIBRARY_PATH="${ONNX_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# stdout を /tmp のログファイルに tee しながら GUI へ流す。
# -a (append) にすることで対局をまたいでもログが蓄積される。
# 可視化サーバーはこのファイルを末尾スキャンして blending_weight を取得する。
LOG_FILE="/tmp/yaneuraou-expert-blending.log"
exec "$ENGINE" "$@" > >(tee -a "$LOG_FILE")
