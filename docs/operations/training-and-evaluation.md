# 学習と評価

## 前提

```bash
bash scripts/check_environment.sh --require-gpu
```

主学習データは`dataset/split_v1_paired_uniform_50/{train,val1}`、初期NNUEは
`logs/halfkp_v1/checkpoints/83000.ckpt`である。学習・評価コマンドはsandbox外で実行する。

## Expert Blending学習

現行の唯一の入口は`scripts/train_expert_blending.sh`。次は8 experts、DNN backboneの
新規runを開始する完全な例である。

実行前に`--run-name`の後、`--`の前へ`--dry-run`を追加すると、学習を開始せず
解決済みコマンドを表示できる。

```bash
bash scripts/train_expert_blending.sh \
  --run-name expert_blending_trial \
  -- \
  --feature-set HalfKP \
  --n-experts 8 \
  --adapter-hidden 128 \
  --adapter-noise-scale 0.0 \
  --batch-size 256 \
  --train-shuffle-buffer-size 64 \
  --epoch-size 1000000 \
  --lr-nnue 0.01 \
  --lr-adapter 0.1 \
  --lambda 1.0 \
  --label-smoothing-eps 0.001 \
  --score-scaling 361 \
  --num-batches-warmup 10000 \
  --newbob-decay 0.5 \
  --num-epochs-to-adjust-lr 20 \
  --min-newbob-scale 1e-5 \
  --momentum 0.9 \
  --network-save-period 10 \
  --max-epochs 1000000 \
  --gpus 1 \
  --seed 42
```

標準出力と標準エラーは`/tmp/train_nnue_expert_blending_trial.log`へ保存される。同じ
`--run-name`を再実行すると`logs/<run-name>/checkpoints/`の最新checkpointから完全再開する。

```bash
tail -f /tmp/train_nnue_expert_blending_trial.log
```

新しい実験は新しいrun名を使う。`--load-weights-only`によるfine-tuneは、`--`以降へ指定する。

## やねうら王形式へ変換

```bash
bash scripts/export_expert_blending.sh \
  logs/expert_blending_trial/checkpoints/100.ckpt \
  tmp/expert_blending_release \
  8
```

`tmp/expert_blending_release/{backbone.onnx,head.bin,head.json}`が生成される。

## 最善手一致率

ベースライン:

```bash
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_halfkp_v1.json \
  data/accuracy_eval/test.jsonl \
  results/accuracy_eval_halfkp_current.json
```

Expert Blendingは先に`tmp/expert_blending_release`へ変換してから実行する。

```bash
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_expert_blending_current.json \
  data/accuracy_eval/test.jsonl \
  results/accuracy_eval_expert_blending_current.json
```

ログは`/tmp/eval_accuracy_<output-name>.log`、詳細結果は指定したJSONへ保存される。設定の仕様と
データ抽出方法は[accuracy-evaluation.md](accuracy-evaluation.md)を参照する。

## validation loss診断

```bash
scripts/gpu_python.sh -u -m train_nnue.check_loss_per_gameply \
  --expert-blending-checkpoint logs/expert_blending_trial/checkpoints/100.ckpt \
  --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \
  --val-dir dataset/split_v1_paired_uniform_50/val1 \
  --feature-set HalfKP \
  --max-positions 1000000 \
  --output tmp/expert_blending_trial/loss_per_gameply.png \
  > /tmp/check_loss_per_gameply.log 2>&1
```

`check_loss_per_expert`など他のGPU評価も同様に`scripts/gpu_python.sh`を使い、必ず
`/tmp`へリダイレクトする。

## TensorBoard

```bash
scripts/nnue_python.sh -m tensorboard.main --logdir logs --port 6006
```

これは長時間プロセスなので、Codexが起動する場合は出力をファイルへ保存する。
