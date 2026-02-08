# AGENTS.md - train-nnue リポジトリガイド

## プロジェクト概要

将棋NNUEモデル (HalfKP 256x2-32-32) の学習環境。
教師データのqsearchシャッフル、学習 (PyTorch Lightning)、モデル変換、やねうら王での動作検証までのパイプラインを管理する。

## 重要な注意事項

### 標準出力の扱い
学習の標準出力をClaude Code等のプロセスが直接受け取ると、ログの蓄積により数十GBのメモリを消費しPCがクラッシュする。
学習コマンドの実行時は**必ず出力をファイルにリダイレクトする** (`> /tmp/xxx.log 2>&1`)。
進捗確認は `tail -f` で別途行う。

### サブモジュール構成
3つのgitサブモジュールがある。nnue-pytorchの中身を編集する場合は、サブモジュール内で先にコミットし、その後親リポジトリでサブモジュール参照を更新してコミットする。

| サブモジュール | ブランチ | 用途 |
|--------------|---------|------|
| `nnue-pytorch/` | `shogi-linux-halfkp256` | 学習フレームワーク (model.py, train.py等) |
| `YaneuraOu/` | (デフォルト) | やねうら王 v9.01git (検証用、変更なし) |
| `tanuki-learner/` | `tanuki-dr4-learner-linux` | shuffle_kifu (qsearch適用シャッフル用) |

### nnue-pytorch の Python 環境
nnue-pytorch 内のスクリプト (`train.py`, `serialize.py` 等) は `nnue-pytorch/.venv/` のvenvで実行する。
`cd nnue-pytorch && source .venv/bin/activate` を忘れずに。

### 大容量データのパス
- 元データ (読み取り専用): `/home/select766/exthdd/dataset/kifu/tanuki-.nnue-pytorch-2024-07-30.1/` (~300GB, 1016個の .bin)
- 加工データ出力先: `/home/select766/exthdd/dev/train-nnue/split_v1/`
- 検証用サブセット: `dataset_qsearch_split/` (リポジトリ直下、train.bin 9.5GB + val.bin 199MB)

## ディレクトリ構成

```
train-nnue/
├── nnue-pytorch/                    # [サブモジュール] 学習フレームワーク
│   ├── model.py                     #   NNUEモデル定義 (HalfKP 256x2-32-32)
│   ├── train.py                     #   学習エントリポイント
│   ├── serialize.py                 #   .ckpt → .nnue 変換
│   ├── nnue_dataset.py              #   C++データローダー
│   ├── features.py / halfkp.py      #   特徴量定義
│   └── .venv/                       #   Python仮想環境
├── YaneuraOu/                       # [サブモジュール] やねうら王 (変更なし)
├── tanuki-learner/                  # [サブモジュール] shuffle_kifu用
│
├── split_and_shuffle.py             # データ分割 (ファイル単位、seed=42)
├── run_shuffle_splits.sh            # 全splitのqsearchシャッフル実行
├── run_shuffle.sh                   # 単一ディレクトリのqsearchシャッフル
├── run_train_halfkp.sh              # 学習実行 (中断・再開対応)
├── shuffle_dataset.py               # 簡易train/val分割 (レコード単位、検証用)
├── run_yaneuraou.py                 # やねうら王USI動作検証
│
├── bin/                             # ビルド成果物・実行環境 (gitignore)
│   ├── YaneuraOu-by-gcc             #   やねうら王バイナリ
│   ├── eval/nn.bin                  #   検証用NNUEモデル
│   └── shuffle/                     #   tanuki-learner実行環境
│       ├── tanuki-learner           #   shuffle_kifuバイナリ
│       └── eval/nn.bin              #   qsearch用既成モデル (学習対象とは別物)
│
├── logs/                            # 学習ログ・チェックポイント (gitignore)
├── dataset_qsearch_split/           # 検証用サブセット (gitignore)
│
├── docs/                            # ドキュメント
│   ├── how-to-train.md              #   大容量データ学習パイプラインの手順
│   ├── how-to-qsearch-shuffle.md    #   qsearchシャッフルの手順 (サブセットでの検証含む)
│   ├── how-to-setup-nnue-pytorch.md #   nnue-pytorch環境構築手順
│   ├── how-to-build-yaneuraou.md    #   やねうら王ビルド手順
│   ├── train-plan.md                #   学習計画・ハイパーパラメータ決定
│   ├── shuffle-model.md             #   シャッフルモデルの手順
│   ├── setup.md                     #   セットアップ手順
│   └── verify-nnue-training.md      #   学習検証メモ
│
└── AGENTS.md                        # リポジトリガイド (本ファイル)
```

## 主要なコマンド

### 学習 (大容量データ)
```bash
# 1. データ分割
python3 split_and_shuffle.py

# 2. qsearchシャッフル (~10時間)
bash run_shuffle_splits.sh 2>&1 | tee /tmp/shuffle_splits.log

# 3. 学習開始 (再開も同じコマンド)
bash run_train_halfkp.sh

# 4. 進捗確認
tail -f /tmp/train_nnue_halfkp.log
```

### モデル変換・検証
```bash
cd nnue-pytorch && source .venv/bin/activate

# .ckpt → .nnue
python serialize.py --features "HalfKP" ../logs/halfkp_v1/checkpoints/XXXX.ckpt ../logs/halfkp_v1/nn.nnue

# やねうら王に配置して検証
cp ../logs/halfkp_v1/nn.nnue ../bin/eval/nn.bin
cd .. && python3 run_yaneuraou.py
```

### 検証用サブセットでの学習 (動作確認用)
```bash
cd nnue-pytorch && source .venv/bin/activate
python train.py --features "HalfKP" --batch-size 16384 --max_epochs 200 \
  --enable_progress_bar False --default_root_dir logs/test_run \
  --threads 8 --lr 0.5 --accelerator gpu --devices 1 \
  --epoch-size 100000 --network-save-period 50 \
  ../dataset_qsearch_split/train.bin ../dataset_qsearch_split/val.bin \
  > /tmp/train_test.log 2>&1
```

## モデルアーキテクチャ

HalfKP 256x2-32-32: 入力(125388) → L1(256) → L2(32) → L3(32) → 出力(1)

学習率はnewbobスケジューリング (`--lr 0.5 0.05` で2段階)。
`newbob_scale` が `min_newbob_scale` を下回ると次のLRステージに移行し、全ステージ完了で学習終了。

## train.py の再開オプション

| オプション | 用途 | 復元される状態 |
|-----------|------|--------------|
| `--resume-from-checkpoint <.ckpt>` | 中断からの再開 | 重み + オプティマイザ + エポック + カスタム状態 |
| `--resume-from-model <.ckpt>` | Fine-tuning | 重みのみ (LR等は初期化) |

チェックポイントは `<default_root_dir>/checkpoints/` に保存される。

## shuffle_kifu の注意

- `shuffle_kifu` は入力ディレクトリ内の**全ファイル**をバイナリとして読み込む。`.bin` 以外のファイル (README.md等) が含まれるとセグフォルトする。シンボリックリンクディレクトリで `.bin` のみを渡すこと。
- `bin/shuffle/eval/nn.bin` はqsearch実行に必要な既成モデル。学習対象のモデルとは別物。
