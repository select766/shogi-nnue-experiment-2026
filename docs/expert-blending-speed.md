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

### iter1: blend_expert_weights を matmul (GEMV) に置換 (2026-04-25)

- 対象 ckpt: 同上
- 変更点: `src/train_nnue/blend_and_export.py` の `blend_expert_weights` で
  `(param * w).sum(dim=0)` ナイーブ broadcast を `torch.matmul(gate, param.reshape(E, -1))` に変更。
  - 元の式は input_weight (E=8, L1=256, F=125388) で約 1 GB の中間テンソルを物理確保していた。
  - matmul に落とすと BLAS GEMV が使われ、中間メモリ不要 + マルチスレッド GEMV で高速化。
- 数値的影響: 量子化後 int16 で `0/32,099,328` 不一致 (純粋に等価)。

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 668.0 | 668.0  | 659.4 | 678.9 | 5.0   |
| Python `infer` (ms)        | 6.7   | 7.0    | 6.0   | 7.0   | 0.5   |
| Python `blend+pack` (ms)   | 310.6 | 310.5  | 307.0 | 319.0 | 2.7   |

- baseline からの改善: **wall -83.8 ms (-11.1%)**, **blend -89.6 ms (-22.4%)**
- 残り内訳 (推定): blend+pack 311 ms / IPC + UpdateWeightsFromBuffer + 探索 ~351 ms
- 100 ms 目標まで: あと **-568 ms**。次は IPC ペイロードを大幅削減 or
  blend+pack をさらに削るために input_weight を事前量子化する案。

### iter2: C++ `UpdateWeightsFromBuffer` の FT を memcpy 化 (2026-04-26)

- 対象 ckpt: 同上
- 変更点:
  - `YaneuraOu/source/eval/nnue/nnue_feature_transformer.h` に
    `LoadParametersFromBuffer(const char*)` を追加 (FT bias / weight を memcpy で
    一括コピー)。`kBufferReadBytes` 定数で読み取りバイト数を提供。
  - `YaneuraOu/source/eval/nnue/evaluate_nnue.cpp::UpdateWeightsFromBuffer` で
    FT 部分は memcpy 経路、FC 部分のみ従来の istream 経路を使うように変更。
  - 動機: `read_little_endian<int16_t>` は per-element `stream.read()` +
    バイト単位アンパックなので、FT の 32M+ 要素 (64MB) ループが支配的だった。
    on-wire は LE 平坦化で x86_64 のメモリレイアウトと一致するため
    memcpy で安全。FC 層はスクランブル順序があるので memcpy 化していない
    (ただし合計 < 17KB なので影響なし)。

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 429.1 | 429.3  | 424.9 | 433.8 | 2.4   |
| Python `infer` (ms)        | 6.3   | 6.0    | 5.0   | 7.0   | 0.6   |
| Python `blend+pack` (ms)   | 311.7 | 312.0  | 306.0 | 317.0 | 2.8   |

- iter1 からの改善: **wall -238.9 ms (-35.8%)**。
  blend は不変 (Python 側未変更) だが、C++ 側が ~351 ms → ~111 ms に短縮。
- baseline からの累計: **wall -322.7 ms (-42.9%)**。
- 残り内訳 (推定): blend+pack 312 ms / C++ + IPC + 探索 ~111 ms
- 100 ms 目標まで: あと **-329 ms**。Python blend+pack 312 ms が支配的になったので
  次はこちらを削る。候補:
  - input_weight を起動時に量子化 + 転置済みでキャッシュし、
    blend を `(8, num_features, L1)` 形状で実行 → `transpose` + `mul` + `round` を回避。
  - もしくはアーキテクチャを変えて Python 側は gate (32B) のみ送信し、
    blend を C++ で行う (64MB IPC を一気に解消)。

### iter3: FT input_weight を起動時に転置 + FT_SCALE 倍してキャッシュ (2026-04-26)

- 対象 ckpt: 同上
- 変更点:
  - `src/train_nnue/blend_and_export.py` に `FastBlendingPacker` クラスを追加。
    起動時に input_weight を `(E, num_features, L1)` (= C++ 期待のメモリ順) に
    transpose し、`FT_SCALE = 127` を掛けた状態でキャッシュする。
    input_bias / base 重みも同様にスケール済みで保持する。
  - `src/train_nnue/dnn_inference_server.py` を `packer.blend_and_pack(gate)`
    経路に切り替え (factorized 特徴量使用時のみ従来 `blend_expert_weights`
    + `quantize_and_pack` にフォールバック)。
  - 毎 go の作業は: matmul → reshape (転置不要) → round → int16 → tobytes。
    `transpose(0,1).contiguous()` (~64MB shuffle) と `× FT_SCALE`、加えて
    1 GB 中間バッファを伴う元の broadcast (iter1 で除去済み) を完全に回避。
- 数値的影響: 量子化後 int16 で `7 / 32,099,584` 不一致 (max abs diff = 1)。
  原因は浮動小数点演算順序の違いによる丸め境界跨ぎ (× FT_SCALE を行う
  タイミングが matmul の前後で異なる)。実用上問題ないと判断。

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 370.5 | 369.4  | 359.3 | 384.2 | 7.9   |
| Python `infer` (ms)        | 7.1   | 7.0    | 6.0   | 8.0   | 0.8   |
| Python `blend+pack` (ms)   | 250.2 | 250.0  | 243.0 | 260.0 | 5.2   |

- iter2 からの改善: **wall -58.6 ms (-13.7%)**, **blend -61.5 ms (-19.7%)**。
- baseline からの累計: **wall -381.3 ms (-50.7%)**。
- 残り内訳 (推定): blend+pack 250 ms / C++ + IPC + 探索 ~120 ms
- 100 ms 目標まで: あと **-270 ms**。Python blend+pack 250 ms 内訳の予想は
  matmul ~50 ms / `round.to(int16)` ~80 ms / `tobytes` + IPC コピー ~50 ms / 残 70 ms。
  次は (a) numpy で `np.rint`/`astype` を使って int16 変換を高速化、
  (b) matmul の `out=` 引数で割当を削減、
  (c) もしくは C++ 側に blend を寄せて 64 MB IPC を撤廃する大改修、
  あたりが有力。

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
