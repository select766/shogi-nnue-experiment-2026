# やねうら王 (YaneuraOu) v9.01git ビルド手順 (Ubuntu 24.04)

## 前提条件

- Ubuntu 24.04
- clang++ (14以上) および lld
- AVX2対応CPU (Intel Haswell以降 / AMD Zen以降)

確認コマンド:

```bash
clang++ --version
ld.lld --version
```

## 1. リポジトリのクローン

```bash
cd /home/select766/shogi/train-nnue
git clone -b v9.01git https://github.com/yaneurao/YaneuraOu.git
```

## 2. ビルド

```bash
cd YaneuraOu/source
make normal
```

デフォルト設定:
- **エディション**: `YANEURAOU_ENGINE_NNUE` (標準NNUE型 = halfKP256)
- **ターゲットCPU**: `AVX2`
- **コンパイラ**: `clang++` (lldリンカ使用)
- **ビルドターゲット**: `normal` (通常使用版)

ビルドが成功すると `YaneuraOu-by-gcc` が `source/` ディレクトリに生成される。

### ビルドオプションの変更

Makefile先頭の変数を変更することで設定を変更可能:

```makefile
# コンパイラをg++に変更する場合
COMPILER = g++

# SSE4.2のみ対応のCPUの場合
TARGET_CPU = SSE42
```

## 3. 評価関数ファイルの配置

NNUE評価関数ファイル (`nn.bin`) を `eval/` ディレクトリに配置する:

```bash
mkdir -p eval
cp /path/to/nn.nnue eval/nn.bin
```

nnue-pytorchで学習した `.nnue` ファイルをそのまま `nn.bin` にリネームして使用できる。
標準NNUE (halfkp_256x2-32-32) のファイルサイズは 64,217,072 バイト。

## 4. 動作確認

### 方法A: 対話的に確認 (手動)

```bash
./YaneuraOu-by-gcc
```

**重要**: 各コマンドは、前のコマンドの応答が返ってから入力すること。
特に `isready` の後は `readyok` が表示されるまで待つ必要がある。
`readyok` の前に `position` や `go` を送ると正しく動作しない。

```
usi
(usiok が表示されるまで待つ)
isready
(readyok が表示されるまで待つ)
position startpos
go byoyomi 1000
(bestmove が表示されるまで待つ)
quit
```

### 方法B: スクリプトで確認 (推奨)

応答待ちを自動で行う検証スクリプト:

```bash
python3 /home/select766/shogi/train-nnue/run_yaneuraou.py ./YaneuraOu-by-gcc
```

正常に動作すれば以下のような出力が得られる:

```
>>> usi
<<< id name YaneuraOu NNUE 9.01git 64AVX2
<<< ...
<<< usiok
>>> isready
<<< info string loading eval file : .../eval/nn.bin
<<< ...
<<< readyok
>>> position startpos
>>> go byoyomi 1000
<<< info depth 16 ... score cp 27 ... pv 7g7f ...
<<< bestmove 7g7f ponder 3c3d
```

## ディレクトリ構成

```
YaneuraOu/
└── source/
    ├── Makefile
    ├── YaneuraOu-by-gcc    # ビルド済みバイナリ
    ├── eval/
    │   └── nn.bin          # NNUE評価関数ファイル
    └── book/               # 定跡ファイル (オプション)
```

## 補足

- 定跡ファイル (`standard_book.db`) がなくても動作する (警告は出るが無視してよい)
- `make evallearn` で学習用バイナリもビルド可能 (OpenBLASが必要)
- `make tournament` で大会用バイナリ (やや高速だが機能制限あり) をビルド可能
