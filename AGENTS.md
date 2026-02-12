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

### Python 環境 (2つの仮想環境)
- `.venv/` (リポジトリルート): データ処理スクリプト用。`uv sync` で管理。`uv run python -m train_nnue.xxx` で実行。
- `nnue-pytorch/.venv/`: 学習用 (PyTorch, python-chess 等)。nnue-pytorch 独自管理。`cd nnue-pytorch && source .venv/bin/activate` で使用。

2つの環境は依存ライブラリが異なるため分離している。

### 大容量データのパス
- 元データ (読み取り専用): `/home/select766/exthdd/dataset/kifu/tanuki-.nnue-pytorch-2024-07-30.1/` (~300GB, 1016個の .bin)
- 加工データ出力先: `/home/select766/exthdd/dev/train-nnue/split_v1/`
- 検証用サブセット: `dataset_qsearch_split/` (リポジトリ直下、train.bin 9.5GB + val.bin 199MB)

## ディレクトリ構成

```
train-nnue/
├── pyproject.toml                   # uv プロジェクト定義 (numpy 依存)
├── uv.lock                          # 依存ロックファイル
├── .python-version                  # Python バージョン (3.11)
│
├── src/train_nnue/                  # Python パッケージ (uv run で実行)
│   ├── split_and_shuffle.py         #   データ分割 (ファイル単位、seed=42)
│   ├── shuffle_dataset.py           #   簡易train/val分割 (レコード単位、検証用)
│   └── run_yaneuraou.py             #   やねうら王USI動作検証
│
├── scripts/                         # Bash スクリプト
│   ├── run_shuffle.sh               #   単一ディレクトリのqsearchシャッフル
│   ├── run_shuffle_splits.sh        #   全splitのqsearchシャッフル実行
│   └── run_train_halfkp.sh          #   学習実行 (中断・再開対応)
│
├── nnue-pytorch/                    # [サブモジュール] 学習フレームワーク
│   ├── model.py                     #   NNUEモデル定義 (HalfKP 256x2-32-32)
│   ├── train.py                     #   学習エントリポイント
│   ├── serialize.py                 #   .ckpt → .nnue 変換
│   ├── nnue_dataset.py              #   C++データローダー
│   ├── features.py / halfkp.py      #   特徴量定義
│   └── .venv/                       #   Python仮想環境 (学習用、独自管理)
├── YaneuraOu/                       # [サブモジュール] やねうら王 (変更なし)
├── tanuki-learner/                  # [サブモジュール] shuffle_kifu用
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
uv run python -m train_nnue.split_and_shuffle

# 2. qsearchシャッフル (~10時間)
bash scripts/run_shuffle_splits.sh 2>&1 | tee /tmp/shuffle_splits.log

# 3. 学習開始 (再開も同じコマンド)
bash scripts/run_train_halfkp.sh

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
cd .. && uv run python -m train_nnue.run_yaneuraou
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

## 学習パフォーマンスの知見

### num_workers (C++データローダーのワーカースレッド数)

`train.py` の `--num-workers` はC++データローダー (`nnue_dataset.py` 経由) のワーカースレッド数を制御する。
RTX 4070 + 16コアCPU環境で、50エポック (epoch-size=1000000, batch-size=16384) のベンチマーク結果:

| num_workers | 実行時間 | 平均GPU使用率 |
|-------------|----------|---------------|
| 1 (デフォルト) | 267s | 62.6% |
| 8 | 232s | 72.2% |
| 16 | 234s | 71.6% |
| 64 | 235s | 71.6% |

- `num_workers=1` → `8` で明確な改善 (実行時間 -13%、GPU使用率 +10pt)
- `num_workers=8` 以上は頭打ち。データローダーではなくGPU側の計算 (モデルが小さくバッチ処理が軽い) が律速と推測
- **推奨値: `--num-workers 8`**

ベンチマーク詳細ログ: `logs/benchmark_num_workers/`

## shuffle_kifu の注意

- `shuffle_kifu` は入力ディレクトリ内の**全ファイル**をバイナリとして読み込む。`.bin` 以外のファイル (README.md等) が含まれるとセグフォルトする。シンボリックリンクディレクトリで `.bin` のみを渡すこと。
- `bin/shuffle/eval/nn.bin` はqsearch実行に必要な既成モデル。学習対象のモデルとは別物。
