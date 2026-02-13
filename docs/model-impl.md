# Expert Blending 実装計画

model-idea-refined.md のロードマップ1, 2を実現するための詳細実装計画。

## 前提情報

### AobaZero アーキテクチャ (調査結果)

| 項目 | 値 |
|------|-----|
| GitHub | https://github.com/kobanium/aobazero |
| ネットワーク | ResNet 256 filters × 20 residual blocks (w449以降) |
| 入力 | 362チャンネル × 9×9 (過去7手の履歴を含む) |
| Policy出力 | 2187 (27×9×9) ※v28以降 |
| Value出力 | スカラー (tanh, [-1, +1]) |
| 活性化関数 | Swish (v28以降、それ以前はReLU) |
| 推論フレームワーク | C++ + Caffe (cuDNN) |
| 重みフォーマット | プレーンテキスト (.txt), xz圧縮配布 (~230MB/ファイル) |
| 最終重み | w4591 (配布版はw1650) |
| 重みダウンロード | http://www.yss-aya.com/aobazero/ |

### 既存 nnue-pytorch 環境

| 項目 | 値 |
|------|-----|
| NNUEアーキテクチャ | HalfKP 256×2-32-32 (入力125,388 → L1=256 → L2=32 → L3=32 → 出力1) |
| 学習フレームワーク | PyTorch Lightning |
| 教師データ形式 | packed SFEN .bin (40バイト/局面) |
| 学習済みチェックポイント | `logs/halfkp_v1/checkpoints/83000.ckpt` |
| データローダー | C++ (libtraining_data_loader.so) によるスパースバッチ |

### 参考: python-dlshogi

| 項目 | 値 |
|------|-----|
| GitHub | https://github.com/TadaoYamaoka/DeepLearningShogi |
| 入力 | 104チャンネル × 9×9 |
| Policy出力 | 2187 (27×9×9) |
| フレームワーク | PyTorch |

python-dlshogiは入力チャンネル数がAobaZeroと異なる(104 vs 362)。特にAobaZeroは過去7手の履歴を入力に含む。Expert Blendingでは局面の特徴量抽出がbackboneの目的であり、過去の手履歴は必須ではない可能性がある。入力特徴量の設計方針は要検討。

---

## ロードマップ 1: AobaZeroとnnue-pytorchの統合

### 方針の選択肢

AobaZeroのbackbone (feat抽出部) をPyTorchで利用する方法として以下を検討する。

**案A: AobaZeroの重みをそのまま変換**
- AobaZeroのネットワーク構造をPyTorchで再実装し、テキスト形式の重みファイルを読み込む
- 利点: 大規模自己対局で学習済みの強力な特徴量をそのまま利用できる
- 課題: 362チャンネル入力の再現が必要（過去7手の履歴情報の生成ロジックが複雑）

**案B: python-dlshogiベースのモデルを利用**
- python-dlshogiのアーキテクチャ (104ch入力) を採用し、別途学習済み重みを用意する
- 利点: 入力特徴量の生成が比較的シンプル
- 課題: 学習済み重みを自前で用意する必要がある (python-dlshogiの公開重みがあれば利用可能)

**案C: AobaZero重みを変換しつつ、入力を簡素化**
- AobaZeroの20ブロックResNetをPyTorchで再実装するが、入力チャンネルを削減 (現局面のみ、履歴なし)
- backboneの最初の畳み込み層だけランダム初期化し、残りはAobaZeroの重みで初期化
- 利点: backboneの事前学習の恩恵を部分的に受けられる
- 課題: 入力層の不一致による性能劣化の可能性

→ **推奨: 案Aを第一目標とし、362ch入力の再現が困難な場合は案Cにフォールバック。**

### ステップ 1-1: AobaZero環境構築とテストデータ作成

- [ ] AobaZeroリポジトリのクローンとビルド (Linux)
- [ ] 重みファイル (w1650.txt.xz 等) のダウンロード
- [ ] AobaZeroでの推論実行確認
- [ ] テスト用局面セット (10~50局面) を用意
- [ ] 各局面に対して以下を保存:
  - 入力テンソル (362 × 9 × 9)
  - backbone出力 (feat)
  - policy出力
  - value出力
- [ ] テストデータをnpy等のフォーマットで保存

**作業場所:** 新ディレクトリ `aobazero/` (リポジトリルート)

**注意事項:**
- AobaZeroのビルドにはCaffe依存がある。Caffeのセットアップが困難な場合、ソースコードからネットワーク構造のみを読み取り、重みテキストから直接変換する方法も検討する
- AobaZeroのソースコード `learn/yss_dcnn.cpp` に入力特徴量エンコーディングの定義がある

### ステップ 1-2: 入力特徴量エンコーダの実装

- [ ] AobaZeroソースから入力特徴量の仕様を正確に把握
  - 362チャンネルの内訳を文書化
  - 盤面→テンソル変換ロジックの理解
- [ ] Pythonで入力特徴量エンコーダを実装
  - cshogiライブラリを使って局面を読み込み、362ch × 9×9 テンソルを生成
  - 過去7手の履歴が必要な場合、その生成方法を決定
- [ ] ステップ1-1のテストデータと一致することを検証

**実装場所:** `src/train_nnue/aobazero_features.py` (新規)

**依存:** cshogi (既存の .venv で利用可能)

### ステップ 1-3: PyTorchモデル定義

- [ ] AobaZeroのResNet構造をPyTorchで再実装
  - 256 filters × 20 blocks
  - Swish活性化 (v28以降)
  - Batch Normalization
  - Policy head (→ 2187)
  - Value head (→ 1)
- [ ] backbone部分 (policy/value headの直前まで) を分離可能な設計にする

**実装場所:** `src/train_nnue/aobazero_model.py` (新規)

### ステップ 1-4: 重み変換

- [ ] AobaZeroのテキスト重みフォーマットを解析
  - Leela Zero系のフォーマット: 各層のconv重み, BN重み(mean, var, gamma, beta)が行ごとに格納
  - 層の順序を確認
- [ ] テキストファイルからPyTorch state_dict への変換スクリプトを実装
- [ ] 変換後のモデルで推論し、ステップ1-1のテストデータ (policy, value) と一致することを検証
  - 数値誤差の許容範囲: float32精度で相対誤差1e-5以下

**実装場所:** `src/train_nnue/convert_aobazero_weights.py` (新規)

### ステップ 1-5: 統合テスト

- [ ] 変換済みPyTorchモデルで複数局面のpolicy/valueを計算
- [ ] AobaZeroオリジナルとの出力一致を確認
- [ ] backboneの出力 (feat) の形状と値の確認
- [ ] GPU推論の動作確認

**完了条件:** PyTorchで `feat = DNN_backbone(state_tensor)` が実行でき、AobaZeroオリジナルと同等の出力が得られること。

---

## ロードマップ 2: 教師あり学習

### ステップ 2-1: Expert Blendingモデルの実装

- [ ] DNN_adapter の実装
  - 入力: backbone出力 feat (shape: batch × feature_channel × 9 × 9)
  - Global Average Pooling → 全結合層2層 → N_EXPERTS次元出力
  - softmax + noise (学習時のみ)
- [ ] NNUE Expert の実装
  - 既存のNNUE重みを N_EXPERTS 個複製して初期化
  - 重み付き平均の計算ロジック
  - NNUE関数 (state_kp, averaged_weight) → value
- [ ] ExpertBlending モデル全体の実装
  - DNN_backbone (frozen) + DNN_adapter (trainable) + NNUE_weights (trainable)
  - forward: state_tensor, state_kp → value

**実装場所:** `src/train_nnue/expert_blending_model.py` (新規)

**設計上の検討事項:**

| パラメータ | 初期値案 | 備考 |
|-----------|---------|------|
| N_EXPERTS | 4 | まず少数から開始。8, 16も実験予定 |
| adapter隠れ層サイズ | 128 | feat → 128 → N_EXPERTS |
| noise分散 | 0.1 | 学習初期は大きめ、アニーリング可能に |
| backbone固定 | True | DNN_backboneの重みは更新しない |

### ステップ 2-2: データローダーの拡張

- [ ] 既存の packed SFEN データから、NNUE用特徴量 (state_kp) と DNN用特徴量 (state_tensor) を同時に生成するデータローダーを実装
  - 既存のC++データローダー (state_kp生成) を活用
  - state_tensorの生成はPython側で行う (ステップ1-2のエンコーダ使用)
- [ ] バッチ内で両方の特徴量が対応することを保証
- [ ] パフォーマンスのボトルネックを測定
  - state_tensorの生成がボトルネックになる場合、C++側への移植やキャッシュを検討

**実装場所:** `src/train_nnue/expert_blending_dataset.py` (新規) または nnue-pytorch のデータローダー拡張

**課題:**
- 過去7手の履歴情報: packed SFEN には1局面の情報しか含まれない。AobaZeroの362ch入力を完全に再現するには棋譜レベルの情報が必要。
  - 対策1: 履歴なしの入力に簡素化 (案C的アプローチ)
  - 対策2: 教師データ内の前後局面から履歴を再構成 (複雑だが可能)
  - **推奨: まず対策1で実装し、性能を確認してから対策2を検討**

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
├── aobazero/                              # AobaZero関連 (gitignore対象)
│   ├── repo/                              # AobaZeroリポジトリクローン
│   ├── weights/                           # ダウンロードした重みファイル
│   └── test_data/                         # テスト用入出力データ (.npy)
│
├── src/train_nnue/
│   ├── aobazero_features.py               # 入力特徴量エンコーダ
│   ├── aobazero_model.py                  # PyTorch ResNetモデル定義
│   ├── convert_aobazero_weights.py        # 重み変換スクリプト
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

## リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| AobaZeroのビルドが困難 (Caffe依存) | 高 | 重みテキストから直接変換。テストデータはpython-dlshogi等で代用検討 |
| 362ch入力の再現が困難 (過去7手履歴) | 中 | 現局面のみの入力に簡素化 (案C) |
| NNUE experts × N の重みが大きくGPUメモリ不足 | 中 | N_EXPERTS=4から開始。重みの差分のみ保持する省メモリ実装も検討 |
| 教師データからstate_tensor生成が遅い | 中 | C++データローダー拡張、事前計算+キャッシュ |
| Expert重みが退化 (1つのexpertに集中) | 低 | noise正則化、entropy正則化、load balancing loss |
| AobaZeroオリジナルと数値が一致しない | 低 | BN層のeps等の差異を慎重に確認。許容誤差を設定して検証 |

---

## 進捗トラッカー

| ステップ | 内容 | 状態 | 備考 |
|---------|------|------|------|
| 1-1 | AobaZero環境構築・テストデータ作成 | TODO | |
| 1-2 | 入力特徴量エンコーダ | TODO | |
| 1-3 | PyTorchモデル定義 | TODO | |
| 1-4 | 重み変換 | TODO | |
| 1-5 | 統合テスト | TODO | |
| 2-1 | Expert Blendingモデル実装 | TODO | |
| 2-2 | データローダー拡張 | TODO | |
| 2-3 | 学習ループ実装 | TODO | |
| 2-4 | 実験・評価 | TODO | |
