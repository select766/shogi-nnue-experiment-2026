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
- 100 ms 目標まで: あと **-270 ms**。Python blend+pack 250 ms 内訳の実測は
  matmul 98 ms / `round.to(int16)` 82 ms / `tobytes` 63 ms / FC ~0 ms。
  matmul はメモリ帯域 (64 MB×8 expert read) で律速、スレッドを増やしても 17 % 程度。
  攻めるべきは round/cast の Python オーバーヘッドと、`tobytes` で生じる
  64 MB の bytes() 割当。

### iter4: matmul `out=` + numpy round + memoryview 直書き (2026-04-26)

- 対象 ckpt: 同上
- 変更点:
  - `FastBlendingPacker.__init__` で FT weight 用の事前確保バッファ
    `(F, L1)` 形状の float32 / int16 (numpy view 込み) を持たせ、
    `payload_size` を起動時に計算。
  - 新メソッド `write_to_stream(gate, stream)` を追加:
    - matmul は `torch.matmul(g.unsqueeze(0), ft_w_flat, out=...)` でバッファ再利用
    - round / int16 化は `np.rint(out=)` + `np.copyto(casting='unsafe')` で in-place
    - 64 MB の int16 結果は `memoryview(...).cast('B')` を `stream.write` に
      直接渡し、bytes() 変換 (=64 MB のもう 1 回コピー) を撤廃
  - `dnn_inference_server.py` を `packer.write_to_stream(gate, sys.stdout.buffer)`
    パスに切り替え (size は `packer.payload_size` で事前確定)。
  - 旧 `blend_and_pack(gate)` は `BytesIO` 経由で互換維持。
- 数値的影響: iter3 と同じ 7 / 32 M 不一致 (max abs diff = 1)。

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 176.8 | 176.0  | 174.1 | 186.0 | 2.9   |
| Python `infer` (ms)        | 6.8   | 6.0    | 5.0   | 10.0  | 1.7   |
| Python `blend+pack` (ms)*  | 105.6 | 105.0  | 103.0 | 114.0 | 2.4   |

注 *: iter4 から `blend+pack` の計測区間が「stdout flush 完了まで」になり、
64 MB の pipe write による blocking 待ち (≈ C++ 側の read 完了) を含む。
これまでの iter1-3 では pipe write の前で計測終了していたため、
直接比較は不可だが、wall は確実に短縮されている。

- iter3 からの改善: **wall -193.7 ms (-52.3%)**。
- baseline からの累計: **wall -575.0 ms (-76.5%)**。
- 残り内訳 (推定): Python (blend + IPC write 待ち) 106 ms / C++ + 探索 71 ms
- 100 ms 目標まで: あと **-77 ms**。残るのは大きく 2 系統:
  - Python の matmul 98 ms (1 GB 読込メモリ帯域がボトルネック)
  - 64 MB IPC pipe (write/read)
  どちらも実体は同じ「合成済み 64 MB 重みを Python→C++ に流すコスト」。
  これを根本的に削るにはアーキテクチャ変更で、Python は gate (32 B) のみを
  送り、blend を C++ 側に持っていく必要がある。

### iter5: 共有メモリ (mmap) IPC で 64 MB pipe 転送を撤廃 (2026-04-26)

- 対象 ckpt: 同上
- 変更点:
  - `src/train_nnue/dnn_inference_server.py`:
    - 起動時に `/tmp/expert_blending_shm_<pid>.bin` を作成 + mmap。
    - ready 行を `"ready <path> <size>\n"` に拡張。旧 `"ready"` も C++ 側で
      互換的にサポート (mmap なしフォールバック)。
    - 毎 go: `packer.write_to_shm(gate)` で mmap に直接書き、pipe には
      gate (32 B) + size (4 B) のみ送る。
  - `src/train_nnue/blend_and_export.py::FastBlendingPacker`:
    - `attach_shm(buf)` / `write_to_shm(gate)` を追加。numpy view 経由で
      mmap に直接書き込む。FT bias / FT weight / FC を所定オフセットへ。
  - `YaneuraOu/source/eval/nnue/dnn_bridge.cpp`:
    - ready 行に shm path/size がある場合、`open` + `mmap(PROT_READ, MAP_SHARED)`。
    - `request_weights` で payload 用の pipe read を撤廃し、mmap から
      memcpy で `buffer` に取り出す (既存 API 維持)。
- 数値的影響: なし (wire format は同一)。

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 163.2 | 163.0  | 157.5 | 169.8 | 2.9   |
| Python `infer` (ms)        | 6.3   | 6.0    | 5.0   | 8.0   | 0.8   |
| Python `blend` (ms)        | 60.1  | 59.5   | 57.0  | 67.0  | 2.4   |

注: `blend` が 106 → 60 ms に大きく減ったのは、pipe write の blocking
(64 MB を C++ 側が pipe から読み終えるまでの待ち) が消えたため。実際の
Python 計算量は iter4 と同じ ~58 ms。一方、C++ 側に memcpy 64 MB
(mmap → vector<char>) が新たに発生して ~5 ms 程度コストが乗るので、
wall の改善は -14 ms にとどまる。

- iter4 からの改善: **wall -13.6 ms (-7.7%)**。
- baseline からの累計: **wall -588.6 ms (-78.3%)**。
- 残り内訳 (推定): Python 60 ms / C++ memcpy + 探索 + 残 ~103 ms。
- 100 ms 目標まで: あと **-63 ms**。
- 次のターゲット:
  - C++ 側 `request_weights` で `vector<char>` への memcpy を撤廃し、
    `UpdateWeightsFromBuffer` に mmap ポインタを直渡し (-5 ms 程度)。
  - 本命は依然「blend を C++ 側に寄せて Python は gate のみ送る」アーキ変更。
    これにより Python 60 ms をほぼ撤廃でき、wall を 90-100 ms 圏に持ち込める。

### iter6: Python/IPC 撤廃 + onnxruntime + C++ 内 ブレンド (2026-04-26)

これまでは Python サブプロセスを起動し、pipe / mmap を経由して合成済み 64 MB
重みをやねうら王へ送り込んでいた。本 iter ではその構造そのものを廃止し、
**やねうら王本体だけで完結**する形に再設計した。

#### アーキテクチャの変更

- やねうら王に **onnxruntime (CPU prebuilt 1.19.2)** を同梱
  (`YaneuraOu/extra/onnxruntime/`)。`fetch_onnxruntime.sh` でダウンロード。
- やねうら王に **dlshogi cppshogi** を vendoring (`YaneuraOu/source/eval/nnue/
  dlshogi_cppshogi/`)。`dlshogi_features::make_features_from_sfen()` で
  `Position` から ONNX backbone への入力 (features1 / features2) を作る。
- 学習済みモデル → やねうら王ロード形式の変換ツール
  `src/train_nnue/export_for_yaneuraou.py` を新規追加。
  - `backbone.onnx` : `DNNBackbone` + `DNNAdapter` を 1 つの ONNX に
    エクスポート (gate (8,) softmax 出力)。
  - `head.bin`      : 128B 固定ヘッダ + E 個の expert を **事前量子化**
    (FT bias / FT weight int16, FC bias int32, FC weight int8 padded) で
    連続書き出し。`(F, L1)` 順 (やねうら王 memcpy 順)。residual モードでは
    末尾に `base_*` 1 セット。
  - `head.json`     : 人間可読のメタ情報 (C++ は読まない)。
- `dnn_bridge.{h,cpp}` を撤去し `expert_blending_loader.{h,cpp}` に置換。
  - `setoption ExpertBlendingDir <path>` でディレクトリを指定。
  - 各 go の前に C++ で:
      ① `Position::sfen()` → cppshogi で features1/2
      ② onnxruntime で gate (8,) を推論
      ③ head.bin の 8 expert × int16 重みを gate でブレンドし、
         `UpdateWeightsFromBuffer` 互換のバイト列を組み立てる
      ④ 既存の `Eval::NNUE::UpdateWeightsFromBuffer` に渡す
- USI option `DNNServerCmd` は削除。代わりに `ExpertBlendingDir`。

#### ブレンドカーネルの実装メモ

- FT weight (32M elem) のブレンドは中間 float バッファ (256MB の write/read)
  を使うとメモリ帯域でボトルネック化したため、**1024 要素チャンク** に分割
  して L1 cache に局所化する実装に変更。これで blend 120ms → **71ms**。
- `lrintf` + `clamp` で int16 へ書き戻す。8 expert × float の積和は
  -O3 + AVX2 で SIMD 化される (clang 14)。
- fixed-point (int16×int16→int32 累積) も試したが、AVX2 で
  vpmaddwd を活かすには gather/permute が必要で、scalar 実装では float 版に
  劣った (165ms)。将来 SIMD intrinsic で書き直せばさらに削れる余地あり。

#### 計測結果

| 指標                       | mean  | median | min   | max   | stdev |
| -------------------------- | ----- | ------ | ----- | ----- | ----- |
| wall-clock per `go` (ms)   | 144.1 | 144.0  | 143.5 | 146.1 | 0.7   |

内訳 (`EXPERT_BLENDING_VERBOSE=1` で `info string ExpertBlending timing(ms)` を確認):
- feat (cppshogi 局面エンコード): **0.01 ms** (誤差レベル)
- onnx (gate 推論, 1 thread CPU):  **9.5 ms**
- blend (FT/FC 合成 + UpdateWeightsFromBuffer 用バッファ組み立て): **71 ms**
- 残り (UpdateWeightsFromBuffer の memcpy + 1000 ノード探索 + USI 往復): **~63 ms**

#### 比較サマリ

| iter | wall (ms) | Python？ | IPC 形式             | 主な変更                          |
| ---- | ---------:|:--------:| -------------------- | --------------------------------- |
| 0    | 751.8     | yes      | pipe (~64MB)         | baseline                          |
| 1    | 668.0     | yes      | pipe                 | blend を matmul に                |
| 2    | 429.1     | yes      | pipe                 | C++ FT を memcpy 化               |
| 3    | 370.5     | yes      | pipe                 | FT input_weight を起動時に転置    |
| 4    | 176.8     | yes      | pipe (block on write)| numpy in-place + memoryview write |
| 5    | 163.2     | yes      | mmap shm             | 64MB pipe 転送を撤廃              |
| **6** | **144.1** | **no**  | **同一プロセス**     | **onnxruntime + C++ ブレンド**    |

- iter5 比 **wall -19.1 ms (-11.7%)**、isready の起動コストも 3.10 s → 1.01 s
  (Python/torch のコールドスタートが消えたため)。
- baseline (iter0) からの累計: **wall -607.7 ms (-80.8%)**。
- アーキ的副次効果:
  - 配布物が 1 ファイル + DLL になり、Python 環境のデプロイが不要。
  - GPU を要求しなくなる (CPU のみで完結)。Windows portable build も視野。
- 残るボトルネック:
  - blend 71 ms (memory-bound. SIMD intrinsic で更に半減可能)
  - onnx 9.5 ms (1 thread CPU。-march/quantization で改善余地)

#### ファイル/オプション変更まとめ

新規:
- `src/train_nnue/export_for_yaneuraou.py`
- `YaneuraOu/source/eval/nnue/dlshogi_cppshogi/` (cppshogi vendoring + bridge)
- `YaneuraOu/source/eval/nnue/expert_blending_loader.{h,cpp}`
- `YaneuraOu/extra/onnxruntime/` (prebuilt 配置場所 + fetch_onnxruntime.sh)

削除:
- `YaneuraOu/source/eval/nnue/dnn_bridge.{h,cpp}`
- USI option `DNNServerCmd`

旧 Python スクリプトは互換用に残置:
- `src/train_nnue/dnn_inference_server.py` (もはや engine からは呼ばれないが、
  Python だけで `blend_and_pack` を試す等の検証用)。

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
