# Expert Blending 実装計画

model-idea-refined.md のロードマップ1, 2を実現するための詳細実装計画。

## 前提情報

### dlshogi アーキテクチャ

| 項目 | 値 |
|------|-----|
| GitHub | https://github.com/TadaoYamaoka/DeepLearningShogi |
| ネットワーク | ResNet 192 filters × 10 blocks (Swish) |
| 入力 | x1: FEATURES1_NUM(62)ch × 9×9 (盤面・利き), x2: FEATURES2_NUM(57)ch × 9×9 (持ち駒等) |
| Policy出力 | 2187 (27×9×9) |
| Value出力 | スカラー (sigmoid, [0, 1]) |
| 活性化関数 | Swish |
| フレームワーク | PyTorch |
| 重みフォーマット | NPZ (dlshogi.serializers) |
| 使用モデル | `model_resnet10_swish-072` (7.3Mパラメータ) |
| backbone出力 | (batch, 192, 9, 9) |

※ AobaZeroからの方針転換: AobaZeroは362ch入力の再現が困難 (詳細: `docs/aobazero-issue.md`)。
dlshogiは入力特徴量エンコーダがC++拡張 (cppshogi) として提供されるため、手動実装不要。

### 既存 nnue-pytorch 環境

| 項目 | 値 |
|------|-----|
| NNUEアーキテクチャ | HalfKP 256×2-32-32 (入力125,388 → L1=256 → L2=32 → L3=32 → 出力1) |
| 学習フレームワーク | PyTorch Lightning |
| 教師データ形式 | packed SFEN .bin (40バイト/局面) |
| 学習済みチェックポイント | `logs/halfkp_v1/checkpoints/83000.ckpt` |
| データローダー | C++ (libtraining_data_loader.so) によるスパースバッチ |

### AobaZero (廃止)

AobaZero (362ch入力) は入力特徴量の再現が困難なため廃止。詳細は `docs/aobazero-issue.md` を参照。
関連ファイル (`aobazero_features.py`, `aobazero_model.py`, `convert_aobazero_weights.py`, `verify_aobazero.py`) は参考として残置。

---

## ロードマップ 1: dlshogiモデルのbackbone利用

### 方針

dlshogi (DeepLearningShogi) の学習済みモデルをbackboneとして利用する。

- dlshogiのResNet (192 filters × 10 blocks, Swish) を事前学習済み特徴抽出器として使用
- 入力特徴量エンコーダはdlshogi付属のC++拡張 (cppshogi) をそのまま利用
- モデル定義もdlshogiライブラリに含まれるため、再実装不要

### ステップ 1-1: dlshogi環境構築・モデル読み込み

- [x] DeepLearningShogi リポジトリのクローン (`dlshogi-source/`)
- [x] ソースからのインストール (`pip install -e dlshogi-source` in `nnue-pytorch/.venv`)
- [x] モデル読み込み確認 (`model_resnet10_swish-072`, 7.3Mパラメータ)
- [x] cppshogi C++拡張による特徴量エンコード動作確認

**特徴量エンコード方法:** cshogi Board → HCP (HuffmanCodedPos) → HCPE構造体 → `hcpe_decode_with_value()` で features1, features2 に変換

### ステップ 1-2: 精度検証

- [x] 検証スクリプト作成 (`src/train_nnue/verify_dlshogi.py`)
- [x] `data/accuracy_eval/test.jsonl` (1000局面) で評価

**結果:**

| 指標 | 値 | 基準 | 判定 |
|------|-----|------|------|
| 指し手一致率 | 49.3% | > 30% | PASS |
| 勝敗一致率 | 72.6% | > 70% | PASS |

**完了条件:** PyTorchで `policy, value = model(x1, x2)` が正常に動作し、精度基準を満たすこと。→ 達成済み

---

## ロードマップ 2: 教師あり学習

### ステップ 2-1: Expert Blendingモデルの実装

- [x] DNN_adapter の実装
  - 入力: backbone出力 feat (shape: batch × 192 × 9 × 9)
  - Global Average Pooling → 全結合層2層 → N_EXPERTS次元出力
  - softmax + noise (学習時のみ)
- [x] NNUE Expert の実装
  - 既存のNNUE重みを N_EXPERTS 個複製して初期化
  - 重み付き平均の計算ロジック
  - NNUE関数 (state_kp, averaged_weight) → value
- [x] ExpertBlending モデル全体の実装
  - DNN_backbone (frozen) + DNN_adapter (trainable) + NNUE_weights (trainable)
  - forward: state_tensor, state_kp → value

**実装場所:** `src/train_nnue/expert_blending_model.py` (新規)

**設計上の検討事項:**

| パラメータ | 初期値案 | 備考 |
|-----------|---------|------|
| N_EXPERTS | 4 | まず少数から開始。8, 16も実験予定 |
| adapter隠れ層サイズ | 128 | 192 → 128 → N_EXPERTS |
| noise分散 | 0.1 | 学習初期は大きめ、アニーリング可能に |
| backbone固定 | True | DNN_backboneの重みは更新しない |

### ステップ 2-2: データローダーの拡張

- [x] 既存の packed SFEN データから、NNUE用特徴量 (state_kp) と DNN用特徴量 (x1, x2) を同時に生成するデータローダーを実装
  - NNUE特徴量: HCP → cshogi Board → SFEN → `make_sparse_batch_from_fens()` で sparse COO tensor 生成
  - DNN特徴量: HCP → HCPE構造体 → `cppshogi.hcpe_decode_with_value()` でバッチデコード
  - 不正なHCPレコード (Huffmanデコード失敗) は自動スキップ
- [x] バッチ内で両方の特徴量が対応することを保証 (correspondence test で全局面一致を確認)
- [x] パフォーマンスのボトルネックを測定 (CPU で ~470 positions/sec @ batch_size=256)

**実装場所:** `src/train_nnue/expert_blending_dataset.py`

**出力タプル:** `(x1, x2, us, them, white, black, outcome, score, ply)` — `ExpertBlendingModel.forward(x1, x2, us, them, w_in, b_in)` に対応

### ステップ 2-3: 学習ループの実装

- [x] PyTorch Lightningベースの学習モジュールを実装
  - 損失関数: 既存NNUEと同じ (teacher_loss + outcome_loss のλブレンド)
  - 勾配はDNN_adapterとNNUE_weightsにのみ流す (backbone frozen)
- [x] 学習ハイパーパラメータの決定
  - NNUE_weightsの学習率: 既存学習と同程度 (初期LR 0.5 → decay)
  - DNN_adapterの学習率: 別途設定可能に (初期はNNUEと同じで試行)
  - 2つのパラメータグループ (adapter / NNUE experts) で個別LR設定
- [x] チェックポイント保存・再開の実装
  - `CheckpointEveryNEpochs` コールバックによる定期保存
  - `--resume-from-checkpoint` による学習再開
  - NewBob state (scale, best_loss等) もチェックポイントに保存
- [x] TensorBoardログの実装 (loss, expert重み分布等)
  - train_loss, val_loss, lr をログ
  - 各expertの平均重み、expert重みエントロピーを100 stepごとにログ
- [x] sparse tensor の dense 変換 (既知の問題の解決)
  - `NNUEExperts.forward()` 冒頭で `w_in.to_dense()` を呼び出す方式 (案A) を採用
- [x] HalfKP → HalfKP^ のパディング対応
  - `load_nnue_experts()` でチェックポイントの重みサイズが異なる場合にゼロパディング

**実装場所:** `src/train_nnue/train_expert_blending.py` (新規)

### ステップ 2-4: 実験・評価

- [x] 小規模データ (subset) での動作確認
  - dataset_qsearch_split/ の train.bin, val.bin を使用
  - **結果:** 3エポック学習で train_loss=0.045, val_loss=0.043 (初期loss 0.046 → 収束確認)
- [x] ベースラインとの比較
  - ベースライン: 単一NNUE (83000.ckpt) の validation loss = **0.0462**
  - Expert Blending (3エポック): val_loss = **0.0427** (ベースライン比 7.6%改善)
  - **検証スクリプト:** `src/train_nnue/eval_baseline_loss.py`
- [x] Expert重みの分析
  - **分析スクリプト:** `src/train_nnue/analyze_expert_weights.py`
  - 4 experts での結果 (10,000局面):
    - Expert 0: mean=0.303, Expert 1: mean=0.300, Expert 2: mean=0.145, Expert 3: mean=0.252
    - エントロピー: mean=0.281 (最大1.386)
    - **結論:** 全expertが使用され、退化は回避 (修正前は1つに集中していた)
- [x] 大規模データでの学習準備
  - **学習スクリプト:** `scripts/run_train_expert_blending_8experts.sh` (8 experts, 長時間学習用)
  - データ: split_v1/ (269GB train.bin)
  - ハイパーパラメータ: HalfKP学習と同等 (max_epochs=1M, network_save_period=500)

**完了条件:** Expert Blendingモデルの学習が完了し、ベースライン(単一NNUE)とのvalidation loss比較ができること。→ 達成済み

**重要なバグ修正:**
- **PackedSfen形式変換バグ:** .binファイルはYaneuraOu PackedSfen形式だが、dlshogiはApery HCP形式を期待。`cshogi.Board.set_psfen()` → `to_hcp()` で変換することで "incorrect Huffman code" エラーを解消。
- **OOMバグ:** `NNUEExperts.forward()` の einsum による重みブレンドが~30GB中間テンソルを生成。`Σ gate_k * F.linear(...)` の線形分解で、重み先行ブレンドと等価なままOOM回避。

---

## ディレクトリ構成 (新規追加分)

```
train-nnue/
├── dlshogi-source/                        # DeepLearningShogi リポジトリ (gitignore対象)
├── tmp/dlshogi-model/                     # dlshogi学習済みモデル (gitignore対象)
│   └── model_resnet10_swish-072           # NPZ形式の重み
│
├── src/train_nnue/
│   ├── verify_dlshogi.py                  # dlshogiモデル精度検証
│   ├── expert_blending_model.py           # Expert Blendingモデル
│   ├── expert_blending_dataset.py         # 複合データローダー (PackedSfen→HCP変換含む)
│   ├── train_expert_blending.py           # 学習スクリプト
│   ├── eval_baseline_loss.py              # ベースラインNNUE validation loss計算
│   └── analyze_expert_weights.py          # Expert重み分布の分析・可視化
│
├── scripts/
│   ├── run_train_expert_blending.sh       # 小規模データ学習スクリプト
│   ├── run_train_expert_blending_large.sh # 大規模データ学習スクリプト (4 experts)
│   └── run_train_expert_blending_8experts.sh  # 大規模データ学習スクリプト (8 experts)
│
└── logs/
    ├── expert_blending_v1/                # 小規模データ学習ログ
    └── expert_blending_8experts_v1/       # 大規模データ学習ログ (8 experts)
```

---

## 既知の問題 (全て解決済み)

### 1. NNUEExperts の OOM (解決済み)

**現象:** `torch.einsum('bn,nof->bof', gate_weights, self.input_weight)` で入力層の重み加重平均を計算すると、中間テンソル (batch, 256, 125388) ≈ 30GB が生成され OOM。

**原因:** 4 experts × (256×125388) 重みの加重平均を batch ごとに生成する einsum 実装。

**解決:** 「重みを先に合成してから推論」の意味論を維持しつつ、入力層は線形性を使って `Σ gate_k * F.linear(x, W_k, b_k)` で等価計算する実装に変更。これにより巨大な `(batch, 256, 125388)` 中間テンソルを作らずOOMを回避。

### 2. PackedSfen と HCP の形式不一致 (解決済み)

**現象:** "incorrect Huffman code" エラーが大量に発生。初期 loss が ~3.06 でベースライン (0.046) と大きく乖離。

**原因:** .bin ファイルは YaneuraOu の PackedSfen 形式 (32 bytes) だが、dlshogi の `hcpe_decode_with_value()` は Apery の HuffmanCodedPos (HCP) 形式 (32 bytes) を期待。両者は異なる Huffman コーディングテーブルを使用しており、直接読み込むと不正な局面として解釈される。

**解決:** `cshogi.Board.set_psfen(PackedSfen)` → `to_hcp(HCP)` で形式変換してから dlshogi 特徴量生成を行う。エラー件数 0、初期 loss もベースラインと一致 (0.046) することを確認。

### 3. NNUE 特徴量の不一致 (解決済み)

**現象:** Expert Blending の NNUE 特徴量とベースライン SparseBatchDataset の特徴量が一致しない (同一局面で非ゼロ要素が 17/38 しか一致しない)。

**原因:** `make_sparse_batch_from_fens()` (SFEN 文字列経由) と C++ の `SparseBatchProvider` (packed SFEN バイナリ直読み) で異なる特徴量生成パスを使用。C++ 側で `pos->set(sfen_string)` と `pos->set_from_packed_sfen()` が異なる実装になっており、HalfKP 特徴量が不一致。

**解決:** `ExpertBlendingDataset` を書き直し、NNUE 特徴量も `SparseBatchProvider` (C++ reader) 経由で生成するよう変更。ベースラインと完全一致を確認。

---

## リスクと対策

| リスク | 影響度 | 対策 | 状態 |
|--------|--------|------|------|
| NNUE experts × N の重みが大きくGPUメモリ不足 | 中 | N_EXPERTS=4から開始。重み先行ブレンドと等価な線形分解実装でOOM回避 | 解決済み |
| 教師データからDNN特徴量生成が遅い | 中 | cppshogi C++拡張でバッチ処理 (~470 pos/sec @ CPU) | 対応済み |
| Expert重みが退化 (1つのexpertに集中) | 低 | noise正則化 (学習時に logits に Gaussian noise 追加) | 監視中 (4 experts で退化なし確認) |
| PackedSfen/HCP 形式の混在 | 高 | set_psfen() → to_hcp() で変換 | 解決済み |

---

## 進捗トラッカー

| ステップ | 内容 | 状態 | 備考 |
|---------|------|------|------|
| 1-1 | dlshogi環境構築・モデル読み込み | DONE | `dlshogi-source/` クローン、pip install、7.3Mパラメータ読み込み確認 |
| 1-2 | 精度検証 | DONE | 指し手一致率49.3%、勝敗一致率72.6% (`verify_dlshogi.py`) |
| 2-1 | Expert Blendingモデル実装 | DONE | `expert_blending_model.py`: DNNBackbone, DNNAdapter, NNUEExperts, ExpertBlendingModel |
| 2-2 | データローダー拡張 | DONE | `expert_blending_dataset.py`: PackedSfen→HCP変換、SparseBatchProvider統合 |
| 2-3 | 学習ループ実装 | DONE | `train_expert_blending.py`: PL Module, 損失関数, NewBob LR, チェックポイント, TBログ |
| 2-4 | 実験・評価 | DONE | 小規模学習(val_loss 0.043)、ベースライン比較(0.0462→0.0427)、expert重み分析、8experts学習スクリプト |

---

## 追記: 実装現状と動作検証 (2026-02-16)

### 実装現状

Expert Blending の対局時 DNN サーバ起動まわりで、以下の修正を反映済み。

1. `dnn_inference_server.py` の `features` import を起動ディレクトリ非依存化
- 変更: `src/train_nnue/dnn_inference_server.py`
- 内容: `NNUE_PYTORCH_DIR` → `<repo>/nnue-pytorch` → `cwd` の順で `features.py` を探索
- 効果: `ModuleNotFoundError: No module named 'features'` を解消

2. 対局スクリプトの DNN 起動コマンドを安定化
- 変更: `scripts/run_expert_blending_match.sh`
- 内容:
  - DNN サーバ起動を `nnue-pytorch/.venv/bin/python` の絶対パスに固定
  - checkpoint / backbone / eval dir を絶対パス化
  - 必須ファイル存在チェックを追加
- 効果: 実行ディレクトリ差異による失敗を回避

3. やねうら王側で DNNBridge 初期化を実際の `isready()` 経路へ実装
- 変更: `YaneuraOu/source/engine/yaneuraou-engine/yaneuraou-search.cpp`
- 内容: `YaneuraOuEngine::isready()` 内で `DNNBridge::init()` を実行
- 背景: `Engine::isready()` 側のみの実装では標準探索エンジン経路で呼ばれず、外部プロセスが起動しなかった

### 動作検証コマンド

1. `dnn_inference_server` 単体起動確認 (手動)
```bash
PYTHONPATH=/home/select766/shogi/train-nnue/src:$PYTHONPATH \
/home/select766/shogi/train-nnue/nnue-pytorch/.venv/bin/python \
-m train_nnue.dnn_inference_server \
  --checkpoint /home/select766/shogi/train-nnue/logs/expert_blending_8experts_v2/checkpoints/160.ckpt \
  --backbone-weights /home/select766/shogi/train-nnue/tmp/dlshogi-model/model_resnet10_swish-072 \
  --features HalfKP \
  --n-experts 8
```
- 期待結果: `Model loaded ...` の後に `ready` が出力される

2. やねうら王から外部プロセス起動されることの確認 (USI 直叩き)
```bash
ROOT=/home/select766/shogi/train-nnue
{
  printf 'usi\n'
  sleep 0.2
  printf 'setoption name EvalDir value %s\n' "$ROOT/bin/eval"
  printf 'setoption name DNNServerCmd value PYTHONPATH=%s/src:$PYTHONPATH %s/nnue-pytorch/.venv/bin/python -m train_nnue.dnn_inference_server --checkpoint %s/logs/expert_blending_8experts_v2/checkpoints/160.ckpt --backbone-weights %s/tmp/dlshogi-model/model_resnet10_swish-072 --features HalfKP --n-experts 8 --log /tmp/dnn_server_from_yane.log\n' "$ROOT" "$ROOT" "$ROOT" "$ROOT"
  printf 'isready\n'
  sleep 2
  printf 'quit\n'
} | ./bin/YaneuraOu-expert-blending
```
- 期待結果:
  - 標準出力に `info string DNNBridge: starting process: ...` が出る
  - `/tmp/dnn_server_from_yane.log` に `Loading model...` が出る

3. `ps` で DNN サーバ子プロセス確認
```bash
ps -ef | rg "dnn_inference_server|YaneuraOu-expert-blending|run_match" -n
```
- 期待結果: `YaneuraOu-expert-blending` の子として `sh -c ... dnn_inference_server` と Python プロセスが見える

4. マッチスクリプト経由の実行確認
```bash
./scripts/run_expert_blending_match.sh \
  ./logs/expert_blending_8experts_v2/checkpoints/160.ckpt \
  ./tmp/dlshogi-model/model_resnet10_swish-072 \
  bin/eval \
  8 1 3000
```
- 期待結果:
  - 起動後に対局が進行し、最終的に `Final Results` が出る
  - 別端末 `ps` で DNN サーバ子プロセスを確認可能

### 注意事項

- `YaneuraOu/source/` を編集した場合は、`bin/YaneuraOu-expert-blending` を必ず再ビルド・再配置すること。  
  (ソース更新のみでは実行バイナリに反映されない)
