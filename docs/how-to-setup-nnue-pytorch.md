# nnue-pytorch 環境構築手順 (Ubuntu 24.04)

## 前提条件

- Ubuntu 24.04
- NVIDIA GPU (RTX 4070で確認)
- git, cmake, g++ がインストール済み
- uv がインストール済み (`pip install uv` または公式インストール方法)

## 1. リポジトリのクローン

```bash
cd /home/select766/shogi/train-nnue
git clone -b shogi.2025-04-12.halfkp_512x2-8-96 https://github.com/nodchip/nnue-pytorch
cd nnue-pytorch
```

使用ブランチは `shogi.2025-04-12.halfkp_512x2-8-96` (HalfKP対応の比較的新しいブランチ)。

## 2. Python仮想環境の構築

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### パッケージのインストール

```bash
# PyTorch (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他の依存パッケージ
uv pip install "pytorch-lightning==1.9.5" "python-chess==0.31.4" cshogi matplotlib tensorboard tensorboardX
```

確認済みバージョン:
- Python 3.11.14
- PyTorch 2.5.1+cu121
- pytorch-lightning 1.9.5
- python-chess 0.31.4
- cshogi 0.9.7

GPU認識確認:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. C++データローダーのビルド

### Linux向けの修正

`lib/nnue_training_data_stream.h` を編集し、Windows固有のヘッダ・関数を除去する:

1. `#include <ppl.h>` の行を削除
2. `BinSfenInputStream::fill` メソッド内の `concurrency::parallel_for` を通常の `for` ループに置換:

```cpp
// 変更前:
concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
    {
        vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
    });

// 変更後:
for (size_t i = 0; i < n; ++i)
{
    vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
}
```

### ビルド

```bash
cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config RelWithDebInfo --target install
```

`libtraining_data_loader.so` がリポジトリ直下に生成される。

## 4. モデル設定の変更 (標準NNUE)

標準NNUE形式 (halfkp_256x2-32-32) を使用する場合、`model.py` の先頭を編集:

```python
# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32
```

## 5. データセットの準備

### データセット形式

データセットは YaneuraOu PackedSfenValue 形式 (1レコード40バイト) の `.bin` ファイル。

### シャッフルと train/val 分割

以下のPythonスクリプト (`shuffle_dataset.py`) でシャッフルと分割を行う:

```bash
python shuffle_dataset.py <入力ディレクトリ> <出力ディレクトリ>
# 例:
python shuffle_dataset.py subset_tanuki-.nnue-pytorch-2024-07-30.1 dataset_shuffled
```

出力:
- `dataset_shuffled/train.bin` (98%)
- `dataset_shuffled/val.bin` (2%)

## 6. 学習の実行

```bash
cd nnue-pytorch
source .venv/bin/activate

python train.py \
  --features "HalfKP" \
  --batch-size 16384 \
  --max_epochs 1000000 \
  --enable_progress_bar False \
  --default_root_dir logs/run1 \
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
  /home/select766/shogi/train-nnue/dataset_shuffled/train.bin \
  /home/select766/shogi/train-nnue/dataset_shuffled/val.bin
```

学習経過は TensorBoard で確認可能:

```bash
tensorboard --logdir logs/run1
```

## 7. モデル変換 (YaneuraOu用)

学習済みチェックポイントを NNUE 形式に変換:

```bash
python serialize.py --features "HalfKP" \
  logs/run1/lightning_logs/version_0/<epoch>.ckpt \
  logs/run1/lightning_logs/version_0/nn.nnue
```

標準NNUE (256x2-32-32) の場合、出力ファイルサイズは 64,217,072 バイト。

YaneuraOu で使用する場合、`nn.nnue` を `nn.bin` にリネームして `eval/` ディレクトリに配置する。

## 8. 学習結果の検証

学習した評価関数が常識的な手を指せるか、やねうら王で検証する。
200エポック程度（約10分）の学習で以下の検証をパスする。

### 準備

学習済みチェックポイントをNNUE形式に変換し、やねうら王の `eval/` に配置する:

```bash
# 変換
python serialize.py --features "HalfKP" \
  logs/verify_run/lightning_logs/version_0/200.ckpt \
  logs/verify_run/lightning_logs/version_0/nn.nnue

# やねうら王に配置
cp logs/verify_run/lightning_logs/version_0/nn.nnue /path/to/YaneuraOu/source/eval/nn.bin
```

### 検証1: 初期局面

初期局面で1秒思考し、初手が `2g2f` (角道を開ける) または `7g7f` (飛車先を突く) であることを確認する。

```
position startpos
go byoyomi 1000
```

期待される出力例: `bestmove 7g7f`

### 検証2: 2g2f 後の局面

`2g2f` が指された局面で1秒思考し、応手が `3c3d` または `8c8d` であることを確認する。

```
position startpos moves 2g2f
go byoyomi 1000
```

期待される出力例: `bestmove 8c8d`

### 検証結果の例

| テスト | 期待される手 | 実際の出力 | 結果 |
|--------|-------------|-----------|------|
| 初期局面 1秒思考 | `2g2f` or `7g7f` | `7g7f` (depth 16, cp 23) | PASS |
| `2g2f`後 1秒思考 | `3c3d` or `8c8d` | `8c8d` (depth 17, cp -7) | PASS |

両方パスすれば、学習環境に欠陥がないことが確認できる。

## ディレクトリ構成

```
train-nnue/
├── setup.md                          # セットアップ指示書
├── how-to-setup-nnue-pytorch.md      # 本ドキュメント
├── shuffle_dataset.py                # データセットシャッフルスクリプト
├── subset_tanuki-.nnue-pytorch-2024-07-30.1/  # 元データセット
├── dataset_shuffled/                 # シャッフル済みデータセット
│   ├── train.bin
│   └── val.bin
└── nnue-pytorch/                     # nnue-pytorch リポジトリ
    ├── .venv/                        # Python仮想環境
    ├── libtraining_data_loader.so    # ビルド済みデータローダー
    ├── model.py                      # モデル定義 (L1=256, L2=32, L3=32 に変更済み)
    ├── train.py                      # 学習スクリプト
    ├── serialize.py                  # NNUE変換スクリプト
    └── logs/                         # 学習ログ・チェックポイント
```
