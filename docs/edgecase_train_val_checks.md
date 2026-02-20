# train/val 挙動エッジケース検証

実施日: 2026-02-18

## 実験条件

短時間実行のため、`max-epochs=1`, `batch-size=256`, `epoch-size=4096` (または 20480) を使用。
共通で paired データセットを利用。

## 実行ケースと結果

1. `lr=0`, `train=val=val1.bin`, `adapter_noise_scale=0.0`
- logdir: `/tmp/edge_lr0_same_noise0`
- 指標: `train_loss=0.033164032`, `val_loss=0.033164032`, `delta=+0.000000000`

2. `lr=0`, `train=val=val1.bin`, `adapter_noise_scale=1.0`
- logdir: `/tmp/edge_lr0_same_noise1`
- 指標: `train_loss=0.033164024`, `val_loss=0.033164032`, `delta=+0.000000007`

3. `lr=0`, `train=train.bin`, `val=val1.bin`, `adapter_noise_scale=0.0`
- logdir: `/tmp/edge_lr0_diff_noise0`
- 指標: `train_loss=0.034900568`, `val_loss=0.033164032`, `delta=-0.001736537`

4. (stepログ確認用) `lr=0`, `train=val=val1.bin`, `adapter_noise_scale=0.0`, `epoch-size=20480`
- logdir: `/tmp/edge_lr0_same_noise0_s80`
- 指標: `train_loss=0.032558817`, `val_loss=0.032555599`, `delta=-0.000003219`

5. (stepログ確認用) `lr=0`, `train=val=val1.bin`, `adapter_noise_scale=1.0`, `epoch-size=20480`
- logdir: `/tmp/edge_lr0_same_noise1_s80`
- 指標: `train_loss=0.032558810`, `val_loss=0.032555599`, `delta=-0.000003211`

## 追加切り分け (単一バッチ)

`adapter_noise_scale=1.0` でも差が出にくい理由を確認するため、単一バッチで train/eval を直接評価。

- experts を初期状態(全expert同一)のまま:
  - `noise=0.0` でも `noise=1.0` でも損失は同値
- expert 0 の重みを人工的にずらす:
  - `train(seed42)=0.036608398`
  - `train(seed43)=0.036551386`
  - `eval(seed42)=0.035635442`
  - `eval(seed43)=0.035635442`

## 解釈

- `lr=0` かつ `train=val` で `train_loss ≒ val_loss` が成立しており、
  train/val パスに重大な不整合は見えない。
- `train != val` では `lr=0` でも損失差が出るため、データ分布差は実在。
- `adapter_noise_scale` が表面上効きにくいのは、初期状態で全expert重みが同一で、
  ゲーティング重みが変わっても blended NNUE 出力が不変になるため。

