# Expert Blending 合成速度の計測ログ

やねうら王 (`YaneuraOu-expert-blending`) において、expert blending モデルを使った
評価関数合成の速度を改善していく際の、変更点と測定結果をここに追記していく。

## 計測方法

- スクリプト: `scripts/benchmark_expert_blending_speed.sh`
  - 内部で `python -m train_nnue.benchmark_blending_speed` を呼ぶ。
- やり方: USI エンジンに `setoption ... DNNServerCmd` を渡して isready で
  Python DNN サーバーを起動し、`go nodes <N>` を繰り返して `bestmove` までの
  wall-clock 時間を測る。`N` を小さく (デフォルト 1000) することで、
  探索コストを実質無視できるようにし、合成パイプライン
  (Python 推論 + IPC + `UpdateWeightsFromBuffer`) の代理指標とする。
- DNN サーバー側のログ (`infer=...s blend=...s`) をパースして内訳も併せて記録する。

ホスト: AMD Ryzen 7 5700X (16 thread) / Linux 6.8.0-110-generic / x86_64。
特記なき限り `Threads=1`、`nodes=1000`、`iters=20` (うち先頭 2 回を warmup として除外)。

数値は `go nodes 1000` 1 回あたりのミリ秒。`measure` 区間 (warmup 後) の値を使う。

## 改善ログ

### baseline — `9bcdaed` (2026-04-25)

- 対象 ckpt: `logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt`
- backbone: `tmp/dlshogi-model/model_resnet10_swish-072`
- n_experts: 8
- 変更点: なし (測定基盤を導入した時点の値)

| 指標                       | mean | median | min  | max  | stdev |
| -------------------------- | ---- | ------ | ---- | ---- | ----- |
| wall-clock per `go` (ms)   | 751.8 | 751.9 | 740.9 | 768.3 | 7.9 |
| Python `infer` (ms)        | 6.2   | 6.0   | 5.0   | 8.0   | 0.7 |
| Python `blend+pack` (ms)   | 400.2 | 401.5 | 392.0 | 418.0 | 6.1 |

その他:
- isready (Python サーバー起動 + ready 受信) 所要: **3.10 s**
- 重みペイロードサイズ: **64,216,868 bytes (~64 MB)**
- wall-clock − (infer + blend) ≈ **345 ms**。これが IPC (~64MB pipe write/read) +
  C++ 側の `UpdateWeightsFromBuffer` + 1000 ノード探索 + その他オーバーヘッド。

支配的なのは **blend+pack (~400ms)** と **C++ 側の重み取り込み + IPC (~345ms)** の 2 つ。
infer (gate weights 推論) は ~6ms で誤差レベル。

## 追記テンプレート

新しい改善を入れたら、以下の体裁で追記する:

```
### <一行サマリ> — `<short-sha>` (YYYY-MM-DD)

- 対象 ckpt: ...
- 変更点: 何をどう変えたか (コミット範囲、修正ファイル、アプローチ)

| 指標                       | mean | median | min  | max  | stdev |
| -------------------------- | ---- | ------ | ---- | ---- | ----- |
| wall-clock per `go` (ms)   |      |        |      |      |       |
| Python `infer` (ms)        |      |        |      |      |       |
| Python `blend+pack` (ms)   |      |        |      |      |       |

- baseline からの改善: -X ms (-Y%)
- メモ: ...
```
