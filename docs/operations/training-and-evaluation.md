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

標準NNUE、Expert Blendingとも評価はCPUで実行する。Expert Blendingのgateはやねうら王へ
組み込んだCPU版ONNX Runtimeが担当し、Python推論サーバーやCUDAは使わない。ただし
やねうら王を起動する評価なので、運用規則に従ってsandbox外で実行する。

ベースライン:

```bash
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_halfkp_v1.json \
  data/accuracy_eval_10k/test.jsonl \
  results/accuracy_eval_10k_halfkp_v1.json
```

Expert Blendingは先に`tmp/expert_blending_release`へ変換してから実行する。

```bash
bash scripts/eval_accuracy.sh \
  configs/accuracy_eval_expert_blending_current.json \
  data/accuracy_eval_10k/test.jsonl \
  results/accuracy_eval_10k_expert_blending_current.json
```

ログは`/tmp/eval_accuracy_<output-name>.log`、詳細結果は指定したJSONへ保存される。設定の仕様と
データ抽出方法は[accuracy-evaluation.md](accuracy-evaluation.md)を参照する。testはvalidationで
選抜済みの少数候補だけに使い、両結果の対応あり比較も同文書の手順で必ず実行する。

checkpoint 180を評価した2026-08-23の結果は
[固定10,000局面評価](../research/results/accuracy-eval-10k-current/README.md)に保存している。

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

## Gate診断

局面別の疎性、全体利用均衡、expert単独評価値の相関・分散、oracle routing改善上限を
validation 10,000局面でまとめて保存する。

```bash
bash scripts/diagnose_gate.sh \
  logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt \
  results/gate_diagnostics_8experts_lambda05_180.json
```

詳細な指標定義とJSON仕様は[Expert分析](expert-analysis.md)を参照する。

## Gate正則化の短期比較

checkpoint 510を同一初期値として、`lambda_sparse={0,1e-4,1e-3,1e-2}`と
`lambda_balance={1e-3,1e-2}`の8条件を各3 epoch学習する。

```bash
bash scripts/run_gate_regularization_sweep.sh
```

各runのTensorBoard値は次のコマンドでJSONへ集約できる。短期比較を通過した候補だけ追加学習し、
独立10,000局面のgate診断と最善手一致率へ進める。

```bash
scripts/nnue_python.sh -m train_nnue.summarize_gate_regularization \
  --pattern 'gate_reg_short3_*_from510' \
  --output results/gate_regularization_short3_from510.json
```

softmaxで基準を通らなかった場合の1.5-entmax短期比較:

```bash
bash scripts/run_entmax15_sweep.sh

scripts/nnue_python.sh -m train_nnue.summarize_gate_regularization \
  --pattern 'gate_entmax15_short3_*_from510' \
  --output results/gate_entmax15_short3_from510.json
```

`--gate-transform entmax15`はcheckpointのhyperparametersへ保存され、gate診断、可視化、
weight分析、やねうら王用ONNX exportで自動復元される。旧checkpointは`softmax`として扱う。

## Router蒸留

checkpoint 510のexpertsを固定し、pairedデータと同じ順序のteacher cacheを作ってadapterだけを
短期学習する。cache生成と学習はGPU必須で、各スクリプトが`gpu_python.sh`と`/tmp`ログを使う。

```bash
# expert単独loss（hard/soft teacher用）
bash scripts/generate_router_teacher_caches.sh

# 単一expert oracleの純粋蒸留とtask-only対照
bash scripts/run_router_distillation_sweep.sh

# task lossと勾配scaleを合わせたsoft teacher補助損失
bash scripts/run_router_distillation_combined_sweep.sh

# 実際のblend lossを局面ごとに最適化したgate teacher
bash scripts/generate_optimized_gate_teacher_caches.sh
bash scripts/run_optimized_gate_distillation_sweep.sh
```

`*.npy` cacheは`tmp/router_teacher_cache/checkpoint510/`へ置く。単一expert版は
`(positions, 8)`のloss、最適化gate版は`(positions, 16)`で先頭8列がexpert loss、後半8列が
teacher gateである。`--val-start-position`はbatch sizeの倍数を指定し、cacheとpaired binaryを
同じ位置から読む。結果と判断は
[router蒸留実験](../research/results/router-distillation-20260823/README.md)に記録した。

## TensorBoard

```bash
scripts/nnue_python.sh -m tensorboard.main --logdir logs --port 6006
```

これは長時間プロセスなので、Codexが起動する場合は出力をファイルへ保存する。
