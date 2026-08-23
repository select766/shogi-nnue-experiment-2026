# DNNとNNUEが異なる局面を読む学習（実装仕様）

## 目的

Expert Blending 学習で、DNNとNNUEが同一局面を読む従来方式から、

- DNN: ルート寄り局面
- NNUE: 探索後局面（qsearch適用）

を読む方式へ切り替える。  
これにより、DNNが読む文脈とNNUEが実戦で評価する局面の乖離を減らす。

## 実装の全体像

### 1. シャッフル後データ形式（paired）

1レコード80バイトのペア形式。

```
[DNN用 PackedSfenValue 40B | NNUE用 PackedSfenValue 40B]
```

- 前半40B（DNN側）: qsearch未適用
- 後半40B（NNUE側）: qsearch適用済み

`PackedSfenValue` は従来どおり40B（sfen 32B + score/move/gamePly/game_result等）。

### 2. paired shuffle 側（tanuki-learner）

実装ファイル:

- `tanuki-learner/source/tanuki_kifu_shuffler.cpp`

追加USIオプション:

- `PairedShuffle` (bool)
- `MaxOutputSamples` (int)
- `OffsetAlpha` (float)

数理モデル（背景と記号定義）:

- 生データを対局ごとの時系列局面として `S(x,y)` と書く
  - `x`: 対局ID
  - `y`: その対局内の局面インデックス（手数方向に増加）
- 1サンプルは `(S(x,y-p), S(x,y))` のペアで構成する
  - DNN入力局面: `S(x,y-p)`（ルート側、qsearch未適用）
  - NNUE教師局面: `S(x,y)`（探索側、qsearch適用）
- `p >= 0` は「同一対局内で、NNUE側局面から何手戻るか」のオフセット
  - `p = 0` なら同一局面
  - `p > 0` なら過去局面

オフセットサンプリングの数学的仕様:

- 理想モデルは
  - `P(p=k) ∝ exp(-alpha * k)` (`k=0,1,2,...`)
- 実装ではこれを離散分布として幾何分布に落としている
  - `r = exp(-alpha)`
  - `P(p=k) = (1-r) * r^k`
  - `k` は `std::geometric_distribution` で生成
- 同一対局制約のため、実際に使う `p` は
  - `p = min(k, y - y_start(x))`
  - ここで `y_start(x)` は対局 `x` の先頭局面インデックス

挙動:

- `PairedShuffle=true` で80Bレコード出力
- NNUE側（後半40B）は従来同様 `ApplyQSearch=true` でqsearch適用
- DNN側（前半40B）は同一対局内の過去局面を使う
- オフセット `p` は上記モデル（指数減衰に対応する幾何分布）でサンプル（`alpha=OffsetAlpha`）
- `MaxOutputSamples>0` なら出力レコード数を上限で打ち切る

### 3. 学習データローダー側（Python）

実装ファイル:

- `src/train_nnue/expert_blending_dataset.py`

追加クラス/機能:

- `PairedExpertBlendingDataset`
- `extract_paired_nnue_bin()`

挙動:

- DNN特徴量は80Bレコード前半40Bから生成
- NNUE特徴量は80Bレコード後半40Bから生成
- NNUE側は既存C++ローダー（`SparseBatchProvider`）を流用するため、
  起動時に後半40Bだけを抽出した一時キャッシュを作成して使用
  - 既定: `<paired_bin名>.nnue40.bin`
  - 学習スクリプトから `--paired-nnue-cache-dir` 指定可

### 4. 学習スクリプト側

実装ファイル:

- `src/train_nnue/train_expert_blending.py`

追加引数:

- `--paired`
- `--paired-nnue-cache-dir`

起動スクリプト:

- `scripts/run_train_expert_blending_8experts_v3.sh`

このスクリプトは `dataset/split_v1_paired/train.bin` / `val1.bin` を使って
`--paired` で学習を開始する。

## コマンド（実運用）

### 0. 前提

- `bin/shuffle/tanuki-learner` が最新ビルド済み
- `bin/shuffle/eval/nn.bin` が存在

必要なら再ビルド:

```bash
cd tanuki-learner/source
make evallearn BLAS= > /tmp/tanuki_evallearn_build.log 2>&1
cd /home/select766/shogi/train-nnue
cp tanuki-learner/source/YaneuraOu-by-gcc bin/shuffle/tanuki-learner
```

### 1. paired shuffle（単一split）

```bash
bash scripts/run_paired_shuffle.sh <input_dir> <output_dir> [threads] [max_output_samples] [offset_alpha]
```

例:

```bash
bash scripts/run_paired_shuffle.sh \
  dataset/split_v1/input_train \
  dataset/split_v1_paired/output_train \
  8 \
  480000000 \
  0.216
```

- `max_output_samples=480000000` は約38.4GB（80B/record）
- `offset_alpha=0.216` は32手程度で重みが十分小さくなる設定

出力は `<output_dir>/shuffled.bin`。

### 2. splitごとの配置

`dataset/split_v1` と同じ分割を使うため、各splitで `shuffled.bin` を作ってリネームする。

例（trainとval1）:

```bash
mkdir -p dataset/split_v1_paired

bash scripts/run_paired_shuffle.sh dataset/split_v1/input_train dataset/split_v1_paired/output_train 8 480000000 0.216
mv dataset/split_v1_paired/output_train/shuffled.bin dataset/split_v1_paired/train.bin
rm -rf dataset/split_v1_paired/output_train

bash scripts/run_paired_shuffle.sh dataset/split_v1/input_val1 dataset/split_v1_paired/output_val1 8 10000000 0.216
mv dataset/split_v1_paired/output_val1/shuffled.bin dataset/split_v1_paired/val1.bin
rm -rf dataset/split_v1_paired/output_val1
```

### 3. paired学習の実行

```bash
bash scripts/run_train_expert_blending_8experts_v3.sh
```

ログ:

- 学習ログ: `/tmp/train_expert_blending_8experts_v3_paired.log`
- 監視: `tail -f /tmp/train_expert_blending_8experts_v3_paired.log`

### 4. スモークテスト（10MB程度）

```bash
# 131072 records * 80B = 10,485,760 bytes
bash scripts/run_paired_shuffle.sh tmp/paired_smoke_input tmp/paired_smoke_output 8 131072 0.216
ls -lh tmp/paired_smoke_output/shuffled.bin
```

## 注意事項

- `shuffle_kifu` 入力ディレクトリには `.bin` 以外を置かないこと
- 学習コマンドの標準出力は必ずファイルへリダイレクトすること
- `dataset/` と `logs/` が外部ディスクへのシンボリックリンク環境では、
  実行権限に注意すること
