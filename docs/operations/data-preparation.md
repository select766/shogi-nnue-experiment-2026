# データ準備

## 現在の保持データ

| パス | 役割 | 状態 |
|---|---|---|
| `dataset/tanuki-.nnue-pytorch-2024-07-30.1/` | 再生成元の40B PackedSfenValue | 保持、読み取り専用扱い |
| `dataset/split_v1_paired_uniform_50/train/` | 主学習データ (`dnn.bin`, `nnue.bin`) | 現行 |
| `dataset/split_v1_paired_uniform_50/val1/` | 主validationデータ | 現行 |

旧`split_v1`、旧paired、`uniform_50_old`は2026-08-23に削除した。`scripts/archive/`や
`docs/archive/`の旧パスを現行入力として使わない。

## レコード形式

- `dnn.bin`: ルート局面。DNN gateの入力。
- `nnue.bin`: 同じサンプルの0--50手先を一様に選びqsearchした局面。NNUE lossの入力。
- どちらも1レコード40Bで、レコード数と順序が一致する必要がある。

確認例:

```bash
scripts/project_python.sh scripts/split_paired_bin.py --help
stat -c '%s %n' dataset/split_v1_paired_uniform_50/val1/{dnn.bin,nnue.bin}
```

## 新しいpairedデータの生成

入力ディレクトリには`.bin`以外を置かない。`shuffle_kifu`は全ファイルをバイナリとして読むため、
READMEなどがあるとクラッシュする。

```bash
make -C tanuki-learner/source evallearn BLAS=NONE -j8
cp tanuki-learner/source/YaneuraOu-by-gcc bin/shuffle/tanuki-learner

bash scripts/run_paired_shuffle.sh \
  /absolute/path/to/bin-only-input \
  dataset/new_paired/train \
  8 0 50 \
  > /tmp/paired_shuffle.log 2>&1
```

引数は`input_dir output_dir threads max_output_samples offset_uniform_max`。処理後、ラッパーが
`shuffled.bin`を`dnn.bin`と`nnue.bin`へ分離する。大規模な再生成を始める前に小規模入力で
件数とペア対応を検証する。
