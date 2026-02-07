# qsearch適用シャッフル → NNUE学習 → 動作検証の手順

## 概要

nnue-pytorchで学習する際、教師データのシャッフルは単純な40バイトレコードの並べ替えではなく、
各局面に対して静止探索(qsearch)を適用し、PVの末端局面に置き換えることがモデルの品質上重要である。

この処理には tanuki- フォーク (select766/tanuki-) の `shuffle_kifu` コマンドを使用する。

全体の流れ:

1. tanuki-learner のビルド (初回のみ)
2. 入力データの準備
3. `shuffle_kifu` の実行 (qsearch適用シャッフル)
4. train/val 分割
5. NNUE学習
6. モデル変換と動作検証

## 前提条件

- how-to-setup-nnue-pytorch.md に従って nnue-pytorch 環境が構築済み
- how-to-build-yaneuraou.md に従って YaneuraOu v9.01git がビルド済み
- 何らかの NNUE 評価関数ファイル (nn.bin) が存在する (qsearch実行に必要)

## 1. tanuki-learner のビルド (初回のみ)

### 1.1 クローン

```bash
cd /home/select766/shogi/train-nnue
git clone -b tanuki-dr4-learner-20240526 https://github.com/select766/tanuki- tanuki-learner
```

### 1.2 Linux向けの修正

`tanuki-learner/source/tanuki_kifu_shuffler.cpp` に以下の修正を適用:

- `#include <direct.h>` → `#include <sys/stat.h>` と `#include <climits>` に置換 (Windows専用ヘッダの除去)
- `_mkdir(...)` → `mkdir(..., 0755)` に置換
- `_MAX_PATH` → `PATH_MAX` に置換
- `_fseeki64` → `fseeko` に置換
- `_ftelli64` → `ftello` に置換

### 1.3 evallearn ビルド

```bash
cd /home/select766/shogi/train-nnue/tanuki-learner/source
make evallearn BLAS=
```

- `BLAS=` はOpenBLASが不要な場合にリンクをスキップするオプション
- ビルド成功: `tanuki-learner/source/YaneuraOu-by-gcc` が生成される

### 1.4 評価関数ファイルの配置

qsearch実行に評価関数が必要:

```bash
mkdir -p tanuki-learner/source/eval
cp YaneuraOu/source/eval/nn.bin tanuki-learner/source/eval/nn.bin
```

## 2. 入力データの準備

**重要**: `KifuReader` はディレクトリ内の全ファイルをバイナリとして読み込むため、
`.bin` 以外のファイル (README.md 等) が含まれるとセグフォルトする。
シンボリックリンクで `.bin` ファイルのみのディレクトリを作成する:

```bash
mkdir -p /home/select766/shogi/train-nnue/test_input_noreadme
ln -s /home/select766/shogi/train-nnue/subset_tanuki-.nnue-pytorch-2024-07-30.1/*.bin \
  /home/select766/shogi/train-nnue/test_input_noreadme/
```

## 3. shuffle_kifu の実行

### 3.1 出力ディレクトリのクリア

```bash
rm -rf /home/select766/shogi/train-nnue/dataset_qsearch_shuffled/*
```

### 3.2 実行

```bash
cd /home/select766/shogi/train-nnue/tanuki-learner/source
bash run_shuffle_noreadme.sh 2>&1
```

`run_shuffle_noreadme.sh` はFIFO経由でコマンドを送信し、`shuffled.bin` の生成を検出する。
8スレッドで全データ (~9.7GB, ~259M records) の処理に約20分。

処理の流れ:
1. 入力データを読み込み、各レコードにqsearchを適用
2. 256個の一時ファイル (`shuffled.000.bin` ~ `shuffled.255.bin`) に分配
3. 各一時ファイルをシャッフルし、最終的に `shuffled.bin` に結合
4. 一時ファイルを削除

## 4. train/val 分割

**注意 (data leakage)**: 同一の `.bin` ファイル内には1つの対局から生成された複数の局面が含まれている。
レコード単位で train/val を分割すると、同じ対局の局面が train と val の両方に含まれ、
正解のリーク（data leakage）が発生する。
本格的な学習では**train/val の分割はファイル単位で行う**必要がある。

```bash
# 例: 33ファイル中、最後の1ファイルをval用、残り32ファイルをtrain用とする
mkdir -p input_train input_val
ls subset_tanuki-.nnue-pytorch-2024-07-30.1/*.bin | head -32 | xargs -I{} ln -s {} input_train/
ls subset_tanuki-.nnue-pytorch-2024-07-30.1/*.bin | tail -1  | xargs -I{} ln -s {} input_val/
# それぞれに対して shuffle_kifu を実行
```

今回の検証では簡易的にレコード単位で分割した:

```bash
cd /home/select766/shogi/train-nnue
python3 shuffle_dataset.py dataset_qsearch_shuffled dataset_qsearch_split
```

これにより `dataset_qsearch_split/train.bin` (98%) と `dataset_qsearch_split/val.bin` (2%) が生成される。

## 5. NNUE学習

**重要 (メモリ消費の回避)**: 学習の標準出力をClaude Code等のプロセスが直接受け取ると、
ログの蓄積により数十GBのメモリを消費しPCがクラッシュする場合がある。
**必ず出力をファイルにリダイレクトする**こと。

```bash
cd /home/select766/shogi/train-nnue/nnue-pytorch
source .venv/bin/activate

python train.py \
  --features "HalfKP" \
  --batch-size 16384 \
  --max_epochs 1000000 \
  --enable_progress_bar False \
  --default_root_dir logs/qsearch_run \
  --threads 8 \
  --lr 0.5 0.05 \
  --num-workers 1 \
  --lambda 1.0 0.5 \
  --label-smoothing-eps 0.001 \
  --accelerator gpu \
  --devices 1 \
  --score-scaling 361 \
  --min-newbob-scale 1e-5 \
  --epoch-size 1000000 \
  --num-epochs-to-adjust-lr 500 \
  --momentum 0.9 \
  --network-save-period 1000 \
  --resume-from-model "" \
  /home/select766/shogi/train-nnue/dataset_qsearch_split/train.bin \
  /home/select766/shogi/train-nnue/dataset_qsearch_split/val.bin \
  > /tmp/train_nnue.log 2>&1
```

学習の進捗を確認する場合:

```bash
tail -f /tmp/train_nnue.log
```

200エポック程度 (~10分) で検証に十分なモデルが得られる。
`final.ckpt` が出力ディレクトリに生成されたら学習完了。

## 6. モデル変換

```bash
cd /home/select766/shogi/train-nnue/nnue-pytorch
source .venv/bin/activate

python serialize.py --features "HalfKP" \
  logs/qsearch_run/lightning_logs/version_0/200.ckpt \
  logs/qsearch_run/lightning_logs/version_0/nn.nnue
```

やねうら王に配置:

```bash
cp logs/qsearch_run/lightning_logs/version_0/nn.nnue \
  /home/select766/shogi/train-nnue/YaneuraOu/source/eval/nn.bin
```

## 7. 動作検証

### USI通信の注意事項

**重要**: やねうら王はUSIプロトコルで `isready` を送信後、`readyok` が返るまで
評価関数のロードやハッシュテーブルの初期化を行っている。
`readyok` を受信する前に `position` や `go` コマンドを送ると、
**エンジンが正しく動作しない** (`depth 1 nodes 0 score cp 0` のような異常な結果を返す)。

`echo -e` でパイプに一括送信する方法ではこの問題が発生するため、
応答を逐次待つスクリプト (`run_yaneuraou.py`) を使用する。

### 検証スクリプトの実行

```bash
cd /home/select766/shogi/train-nnue/YaneuraOu/source
python3 /home/select766/shogi/train-nnue/run_yaneuraou.py ./YaneuraOu-by-gcc
```

`run_yaneuraou.py` は以下を自動実行する:
1. `usi` → `usiok` 待ち
2. `isready` → `readyok` 待ち
3. 初期局面で `go byoyomi 1000` → `bestmove` 待ち
4. `2g2f` 後の局面で `go byoyomi 1000` → `bestmove` 待ち

### 期待される結果

| テスト | 期待される手 | 合格例 |
|--------|-------------|--------|
| 初期局面 1秒思考 | `2g2f` or `7g7f` | `bestmove 2g2f` (depth 17, cp 22) |
| `2g2f`後 1秒思考 | `3c3d` or `8c8d` | `bestmove 8c8d` (depth 17, cp -21) |

両方パスすれば、qsearch適用シャッフルを含む学習パイプライン全体が正常に動作している。

## トラブルシューティング

### セグフォルト (SIGSEGV) で `shuffle_kifu` がクラッシュする

**原因**: `KifuReader` は `std::filesystem::directory_iterator` で入力ディレクトリ内の全ファイルを列挙し、
バイナリの棋譜ファイルとして読み込む。`README.md` などの非バイナリファイルが含まれると、
不正なデータが局面として解釈され、`qsearch` 実行時にセグフォルトが発生する。

**対策**: `.bin` ファイルのみのディレクトリを用意する (上記セクション2参照)。

### YaneuraOu が `depth 1 nodes 0 score cp 0` を返す

**原因**: `isready` の応答 (`readyok`) を待たずに `position` や `go` コマンドを送信すると、
エンジンの初期化が完了しておらず、探索が正しく行われない。

**対策**: `run_yaneuraou.py` を使用し、`readyok` を受信してから次のコマンドを送る。
手動でインタラクティブに実行する場合は、`readyok` が表示されてから次のコマンドを入力する。

### 学習中にPCがクラッシュする (メモリ不足)

**原因**: Claude Code等のプロセスが学習の標準出力を直接受け取ると、
出力行のバッファが蓄積し、数十GBのメモリを消費する場合がある。
200エポックでログが数万行に達するため、長時間の学習で特に顕著。

**対策**: 学習コマンドの出力を必ずファイルにリダイレクトする (`> /tmp/train_nnue.log 2>&1`)。
進捗確認は `tail -f /tmp/train_nnue.log` で別途行う。

## ディレクトリ構成

```
train-nnue/
├── run_yaneuraou.py                 # USI動作検証スクリプト (応答待ち対応)
├── shuffle_dataset.py               # train/val分割スクリプト
├── tanuki-learner/                  # tanuki- フォーク (shuffle_kifu用)
│   └── source/
│       ├── YaneuraOu-by-gcc         # evallearn ビルド済みバイナリ
│       ├── eval/nn.bin              # qsearch用評価関数
│       ├── run_shuffle_noreadme.sh  # シャッフル実行スクリプト
│       └── tanuki_kifu_shuffler.cpp # Linux向け修正済み
├── test_input_noreadme/             # .bin のみのシンボリックリンクディレクトリ
├── dataset_qsearch_shuffled/        # shuffle_kifu 出力 (shuffled.bin)
├── dataset_qsearch_split/           # train/val 分割後
│   ├── train.bin
│   └── val.bin
├── YaneuraOu/                       # やねうら王 v9.01git
│   └── source/
│       ├── YaneuraOu-by-gcc         # ビルド済みバイナリ
│       └── eval/nn.bin              # 検証用評価関数
├── nnue-pytorch/                    # 学習フレームワーク
│   └── logs/qsearch_run/           # 学習結果
└── subset_tanuki-.nnue-pytorch-2024-07-30.1/  # 元データセット (.bin + README.md)
```
