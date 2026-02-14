# 大容量データセットでのNNUE学習手順

## 概要

300GBの教師データ (tanuki-.nnue-pytorch-2024-07-30.1) を用いてHalfKP NNUEモデルを学習するための手順。

パイプライン全体の流れ:

1. データの分割 (ファイル単位、data leakage回避)
2. 各splitにqsearch適用シャッフル
3. 学習 (中断・再開対応)
4. モデル変換・動作検証

## 前提条件

- `how-to-qsearch-shuffle.md` の手順1 (tanuki-learnerビルド) が完了済み
- `how-to-setup-nnue-pytorch.md` に従って nnue-pytorch 環境が構築済み
- 元データセットが以下に配置済み (読み取り専用):
  `./dataset/tanuki-.nnue-pytorch-2024-07-30.1/` (1016個の `.bin` ファイル)

## 1. データ分割

1016個の `.bin` ファイルをファイル単位で6つのsplitに分割する。
同一 `.bin` ファイル内には1つの対局から生成された複数の局面が含まれるため、
レコード単位ではなくファイル単位で分割することでdata leakageを回避する。

### 1.1 分割の実行

```bash
cd /home/select766/shogi/train-nnue
uv run python -m train_nnue.split_and_shuffle
```

**処理内容** (`src/train_nnue/split_and_shuffle.py`):
- ソースディレクトリの `.bin` ファイル1016個をリストアップ・ソート
- seed=42でリストをシャッフル
- ファイルをsplitに割り当て: train=916, val1=20, val2=20, val3=20, val4=20, test=20
- 各split用のシンボリックリンクディレクトリを作成
- `split_manifest.json` に割り当て情報を保存

**出力先**: `./dataset/split_v1/`

```
split_v1/
  input_train/    → .binファイルへのシンボリックリンク (916個)
  input_val1/     → シンボリックリンク (20個)
  input_val2/     → シンボリックリンク (20個)
  input_val3/     → シンボリックリンク (20個)
  input_val4/     → シンボリックリンク (20個)
  input_test/     → シンボリックリンク (20個)
  split_manifest.json
```

### 1.2 分割結果の検証

```bash
# ファイル数の確認
python3 -c "
import json
m = json.load(open('./dataset/split_v1/split_manifest.json'))
for k, v in m.items():
    print(f'{k}: {len(v)} files')
total = sum(len(v) for v in m.values())
print(f'total: {total} files')
"
```

期待される出力:
```
train: 916 files
val1: 20 files
val2: 20 files
val3: 20 files
val4: 20 files
test: 20 files
total: 1016 files
```

## 2. qsearch適用シャッフル

各splitのシンボリックリンクディレクトリに対して `shuffle_kifu` (qsearch適用+シャッフル) を実行する。

### 2.1 実行

```bash
cd /home/select766/shogi/train-nnue
bash scripts/run_shuffle_splits.sh 2>&1 | tee /tmp/shuffle_splits.log
```

**処理内容** (`scripts/run_shuffle_splits.sh`):
- 6つのsplit (train, val1-4, test) を順に処理
- 各splitに対して `scripts/run_shuffle.sh` (8スレッド) を実行
- 出力の `shuffled.bin` を `train.bin`, `val1.bin` 等にリネーム
- 既に出力ファイルが存在するsplitはスキップ (中断後の再実行に対応)

**推定処理時間**: 約10時間 (8スレッド、300GB)

**出力**:
```
split_v1/
  train.bin     → ~269GB (qsearch適用+シャッフル済)
  val1.bin      → ~6GB
  val2.bin      → ~6GB
  val3.bin      → ~6GB
  val4.bin      → ~6GB
  test.bin      → ~6GB
```

### 2.2 検証

```bash
ls -lh ./dataset/split_v1/*.bin
```

各 `.bin` ファイルが生成されていることを確認する。

## 3. 学習

### 3.1 学習スクリプトの実行

```bash
cd /home/select766/shogi/train-nnue
bash scripts/run_train_halfkp.sh
```

**処理内容** (`scripts/run_train_halfkp.sh`):
- `logs/halfkp_v1/checkpoints/` から最新の `.ckpt` を探す
- 見つかれば `--resume-from-checkpoint` で再開、なければ新規開始
- 出力は `/tmp/train_nnue_halfkp.log` にリダイレクト

**ハイパーパラメータ**:

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| `--features` | HalfKP | 特徴量 |
| `--batch-size` | 16384 | GPU最適化 |
| `--lr` | 0.5 0.05 | 2段階LR (newbobで切替) |
| `--lambda` | 1.0 0.5 | 2段階lambda |
| `--momentum` | 0.9 | SGDモメンタム |
| `--score-scaling` | 361 | 評価値スケーリング |
| `--epoch-size` | 1000000 | 1エポックあたりの局面数 |
| `--num-epochs-to-adjust-lr` | 500 | LR調整間隔 |
| `--network-save-period` | 500 | チェックポイント保存間隔 (約40分ごと) |
| `--label-smoothing-eps` | 0.001 | ラベル平滑化 |

### 3.2 進捗の確認

```bash
tail -f /tmp/train_nnue_halfkp.log
```

LR調整のログ例:
```
self.current_epoch=500, latest_loss=0.283 < self.best_loss=10000000000.0, accepted, self.newbob_scale=1.0
self.current_epoch=1000, latest_loss=0.281 < self.best_loss=0.283, accepted, self.newbob_scale=1.0
```

`newbob_scale` が `min_newbob_scale` (1e-5) を下回ると、次のLRステージに移行するか学習が終了する。

### 3.3 中断と再開

**中断**: Ctrl-C で学習を停止する。

**再開**: 同じスクリプトを再実行する。

```bash
bash scripts/run_train_halfkp.sh
```

スクリプトが `logs/halfkp_v1/checkpoints/` 内の最新 `.ckpt` を自動検出し、
`--resume-from-checkpoint` で全状態を復元して学習を再開する。

復元される状態:
- モデルの重み
- オプティマイザの状態 (モメンタム等)
- エポック番号・グローバルステップ
- `newbob_scale` (LRスケール)
- `best_loss` (ベスト検証ロス)
- `parameter_index` (LRステージのインデックス)
- warmup関連の状態

### 3.4 チェックポイントのディレクトリ構成

```
logs/halfkp_v1/
  checkpoints/
    500.ckpt
    1000.ckpt
    ...
  lightning_logs/
    version_0/     → TensorBoardログ
    version_1/     → 再開時に新バージョンが作られる
```

チェックポイントは `checkpoints/` に保存されるため、
TensorBoardのバージョン番号 (`version_N`) に依存せず安定した場所に保存される。

TensorBoard可視化:

```bash
(cd nnue-pytorch; uv run tensorboard --logdir ../logs)
```

## 4. モデル変換と動作検証

### 4.1 .nnue ファイルへの変換

```bash
cd /home/select766/shogi/train-nnue/nnue-pytorch
source .venv/bin/activate

# 最新チェックポイントを変換
python serialize.py --features "HalfKP" \
  ../logs/halfkp_v1/checkpoints/XXXX.ckpt \
  ../logs/halfkp_v1/nn.nnue
```

`XXXX.ckpt` は対象のチェックポイントファイル名に置き換える。

### 4.2 やねうら王への配置

```bash
cp /home/select766/shogi/train-nnue/logs/halfkp_v1/nn.nnue \
   /home/select766/shogi/train-nnue/bin/eval/nn.bin
```

### 4.3 動作検証

```bash
cd /home/select766/shogi/train-nnue
uv run python -m train_nnue.run_yaneuraou
```

詳細は `how-to-qsearch-shuffle.md` のセクション7を参照。

## 実装の技術的詳細

### 中断・再開の仕組み

既存の `--resume-from-model` はモデル重みのみの復元 (fine-tuning用) だったため、
完全な学習再開のために以下を実装した。

**model.py の変更**:
- `on_save_checkpoint()`: チェックポイントに `newbob_scale`, `best_loss`, `parameter_index`,
  `warmup_start_global_step`, `latest_loss_sum`, `latest_loss_count` を保存
- `on_load_checkpoint()`: チェックポイントから上記の状態を復元

**train.py の変更**:
- `--resume-from-checkpoint` 引数を追加: PyTorch Lightning の `trainer.fit(ckpt_path=...)` で
  重み・オプティマイザ・エポック番号・カスタム状態をすべて復元
- チェックポイント保存先を `default_root_dir/checkpoints/` に固定:
  TensorBoardの `version_N` に依存しないため、再起動時にチェックポイントが見つからなくなる問題を解消

### --resume-from-model と --resume-from-checkpoint の違い

| | `--resume-from-model` | `--resume-from-checkpoint` |
|---|---|---|
| 用途 | Fine-tuning | 中断からの再開 |
| モデル重み | 復元する | 復元する |
| オプティマイザ状態 | 復元しない (新規作成) | 復元する |
| エポック番号 | 0からリセット | 保存時の値から継続 |
| newbob_scale / best_loss | 復元しない (初期値) | 復元する |
| LRステージ (parameter_index) | 復元しない (0) | 復元する |

## ディレクトリ構成

```
train-nnue/
├── src/train_nnue/
│   └── split_and_shuffle.py      # データ分割スクリプト (ファイル単位)
├── scripts/
│   ├── run_shuffle_splits.sh     # 全split qsearchシャッフル実行
│   ├── run_train_halfkp.sh       # 学習実行 (中断・再開対応)
│   └── run_shuffle.sh            # 単一ディレクトリの qsearchシャッフル
├── logs/
│   └── halfkp_v1/               # 学習ログ・チェックポイント
│       ├── checkpoints/         # .ckpt ファイル (安定したパス)
│       └── lightning_logs/      # TensorBoardログ
├── nnue-pytorch/                # 学習フレームワーク
│   ├── model.py                 # チェックポイントフック追加済み
│   └── train.py                 # --resume-from-checkpoint 追加済み
└── bin/
    ├── eval/nn.bin              # 検証用モデル
    └── shuffle/                 # tanuki-learner 実行環境

└── dataset/ # 大容量データ (シンボリックリンクで外部ディスクへポイント)
      └── split_v1/
          ├── split_manifest.json      # 分割情報
          ├── input_train/             # シンボリックリンク (916個)
          ├── input_val1/ ~ input_test/
          ├── train.bin                # qsearch適用+シャッフル済み (~269GB)
          ├── val1.bin ~ val4.bin      # 各~6GB
          └── test.bin                 # ~6GB
```
