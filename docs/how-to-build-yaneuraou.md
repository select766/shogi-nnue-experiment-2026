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
YaneuraOu サブモジュール内にビルド成果物を残さないため、`bin/` にコピーする:

```bash
cp YaneuraOu/source/YaneuraOu-by-gcc bin/YaneuraOu-by-gcc
```

### Expert Blending 用バイナリのビルド

`DNNServerCmd` を使う Expert Blending 対局では、修正済み `YaneuraOu` ソースから
`bin/YaneuraOu-expert-blending` を作り直す。

```bash
cd /home/select766/shogi/train-nnue/YaneuraOu/source
make clean
make -j4
cp YaneuraOu-by-gcc ../../bin/YaneuraOu-expert-blending
```

注意:
- `YaneuraOu/source/` を編集した後は、必ず再ビルドして `bin/YaneuraOu-expert-blending` を更新すること
- ソースだけ更新しても、`bin/` の実行バイナリは自動更新されない

### ビルドオプションの変更

Makefile先頭の変数を変更することで設定を変更可能:

```makefile
# コンパイラをg++に変更する場合
COMPILER = g++

# SSE4.2のみ対応のCPUの場合
TARGET_CPU = SSE42
```

## 3. 評価関数ファイルの配置

NNUE評価関数ファイル (`nn.bin`) を `bin/eval/` ディレクトリに配置する:

```bash
mkdir -p bin/eval
cp /path/to/nn.nnue bin/eval/nn.bin
```

nnue-pytorchで学習した `.nnue` ファイルをそのまま `nn.bin` にリネームして使用できる。
標準NNUE (halfkp_256x2-32-32) のファイルサイズは 64,217,072 バイト。

## 4. 動作確認

### 方法A: 対話的に確認 (手動)

`bin/` ディレクトリで実行する (エンジンが `eval/nn.bin` をカレントディレクトリ基準で探すため):

```bash
cd bin
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

応答待ちを自動で行う検証スクリプト (リポジトリルートから実行可能):

```bash
uv run python -m train_nnue.run_yaneuraou
```

引数なしでデフォルトの `bin/YaneuraOu-by-gcc` を使用する。
正常に動作すれば以下のような出力が得られる:

```
>>> usi
<<< id name YaneuraOu NNUE 9.01git 64AVX2
<<< ...
<<< usiok
>>> isready
<<< info string loading eval file : .../bin/eval/nn.bin
<<< ...
<<< readyok
>>> position startpos
>>> go byoyomi 1000
<<< info depth 16 ... score cp 27 ... pv 7g7f ...
<<< bestmove 7g7f ponder 3c3d
```

### 方法C: DNNServerCmd 起動確認 (Expert Blending向け)

`isready` 時に DNNBridge が外部プロセスを起動できることを確認する。

```bash
ROOT=/home/select766/shogi/train-nnue
{
  printf 'usi\n'
  sleep 0.2
  printf 'setoption name EvalDir value %s\n' "$ROOT/bin/eval"
  printf 'setoption name DNNServerCmd value /bin/echo\n'
  printf 'isready\n'
  sleep 0.2
  printf 'quit\n'
} | "$ROOT/bin/YaneuraOu-expert-blending"
```

期待結果:
- `info string DNNBridge: starting process: /bin/echo`
- `/bin/echo` は `ready` を返さないため、続いて
  `info string DNNBridge: did not receive 'ready' ...`
  が出る (この2行が出れば起動経路は動作している)

## ディレクトリ構成

```
train-nnue/
├── YaneuraOu/              # サブモジュール (変更を加えない)
│   └── source/
│       └── Makefile
└── bin/                    # ビルド成果物・実行環境 (gitignore)
    ├── YaneuraOu-by-gcc    # ビルド済みバイナリ (YaneuraOu/source/ からコピー)
    ├── eval/
    │   └── nn.bin          # 学習済みNNUE評価関数ファイル (検証用)
    └── shuffle/            # tanuki-learner 実行環境 (シャッフル用、別モデル)
        ├── tanuki-learner  # tanuki-learner バイナリ
        └── eval/
            └── nn.bin      # 既成品モデル (qsearch用)
```

## 補足

- 定跡ファイル (`standard_book.db`) がなくても動作する (警告は出るが無視してよい)
- `make evallearn` で学習用バイナリもビルド可能 (OpenBLASが必要)
- `make tournament` で大会用バイナリ (やや高速だが機能制限あり) をビルド可能
