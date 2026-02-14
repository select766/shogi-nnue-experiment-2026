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

- [ ] PyTorch Lightningベースの学習モジュールを実装
  - 損失関数: 既存NNUEと同じ (teacher_loss + outcome_loss のλブレンド)
  - 勾配はDNN_adapterとNNUE_weightsにのみ流す (backbone frozen)
- [ ] 学習ハイパーパラメータの決定
  - NNUE_weightsの学習率: 既存学習と同程度 (初期LR 0.5 → decay)
  - DNN_adapterの学習率: 別途設定可能に (初期はNNUEと同じで試行)
- [ ] チェックポイント保存・再開の実装
- [ ] TensorBoardログの実装 (loss, expert重み分布等)

**実装場所:** `src/train_nnue/train_expert_blending.py` (新規)

### ステップ 2-4: 実験・評価

- [ ] 小規模データ (subset) での動作確認
  - dataset_qsearch_split/ の train.bin, val.bin を使用
  - 学習が収束すること、lossが下がることを確認
- [ ] Expert重みの分析
  - 局面に応じてexpert重みがどう変化するか可視化
  - 全expertが均等に使われているか確認
- [ ] ベースラインとの比較
  - ベースライン: 単一NNUE (83000.ckpt)
  - 比較指標: validation loss
- [ ] 大規模データでの学習 (split_v1データ使用)

**完了条件:** Expert Blendingモデルの学習が完了し、ベースライン(単一NNUE)とのvalidation loss比較ができること。

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
│   ├── expert_blending_dataset.py         # 複合データローダー
│   └── train_expert_blending.py           # 学習スクリプト
│
├── scripts/
│   └── run_train_expert_blending.sh       # 学習実行スクリプト
│
├── configs/
│   └── expert_blending_default.json       # ハイパーパラメータ設定
│
└── logs/expert_blending_v1/               # 学習ログ (gitignore対象)
```

---

## 既知の問題

### NNUEExperts の sparse tensor 非対応 (ステップ 2-3 で要対応)

**現象:** `ExpertBlendingModel.forward()` に実データを流すと `bmm_sparse: Tensor 'mat2' must be dense` エラーが発生する。

**原因:** データローダー (`expert_blending_dataset.py`) が返す `white`, `black` は `torch.sparse_coo_tensor` (shape `[B, 127017]`)。一方 `NNUEExperts.forward()` の168-169行目で `torch.bmm(avg_input_w, w_in.unsqueeze(-1))` を呼んでおり、`bmm` は sparse tensor を mat2 に受け付けない。

**対処方針 (ステップ 2-3 で実装):**

1. **案A: モデル側で dense 変換** — `NNUEExperts.forward()` の冒頭で `w_in = w_in.to_dense()` を呼ぶ。最もシンプルだが、`[B, 127017]` の dense tensor を作るためメモリ効率が悪い。
2. **案B: sparse mm を使う** — `torch.bmm` の代わりに `torch.sparse.mm` を使い、einsum の結果と sparse 入力の行列積を計算する。ただし batch 次元の扱いに工夫が必要。
3. **案C: データローダー側で dense 化** — データローダーが dense tensor を返すように変更。案Aと本質的に同じだが責務の切り分けが異なる。

**影響範囲:** `expert_blending_model.py` の `NNUEExperts.forward()` のみ。データローダー側は変更不要 (sparse/dense どちらにも対応可能な設計)。

---

## リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| NNUE experts × N の重みが大きくGPUメモリ不足 | 中 | N_EXPERTS=4から開始。重みの差分のみ保持する省メモリ実装も検討 |
| 教師データからDNN特徴量生成が遅い | 中 | cppshogi C++拡張でバッチ処理。事前計算+キャッシュも検討 |
| Expert重みが退化 (1つのexpertに集中) | 低 | noise正則化、entropy正則化、load balancing loss |

---

## 進捗トラッカー

| ステップ | 内容 | 状態 | 備考 |
|---------|------|------|------|
| 1-1 | dlshogi環境構築・モデル読み込み | DONE | `dlshogi-source/` クローン、pip install、7.3Mパラメータ読み込み確認 |
| 1-2 | 精度検証 | DONE | 指し手一致率49.3%、勝敗一致率72.6% (`verify_dlshogi.py`) |
| 2-1 | Expert Blendingモデル実装 | DONE | `expert_blending_model.py`: DNNBackbone, DNNAdapter, NNUEExperts, ExpertBlendingModel |
| 2-2 | データローダー拡張 | DONE | `expert_blending_dataset.py` 実装。下記「既知の問題」参照 |
| 2-3 | 学習ループ実装 | TODO | |
| 2-4 | 実験・評価 | TODO | |
