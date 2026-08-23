# Blending Head NNUE バックボーン実装計画

## Context

現在、Expert Blendingモデルのバックボーンは `DNNBackbone`（dlshogi ResNet10、frozen）+ `DNNAdapter`（FC 2層→gate weights）で構成されている。これは処理負荷が高く学習に時間がかかるため、バックボーンにもNNUE構造を使う軽量な代替を導入する。

## 重要な制約: データセットの局面ペア構造

dnn.bin と nnue.bin は**異なる局面**を含んでいる（ペアだが同一ではない）。

- dnn.bin の局面 → backbone（DNN or NNUE）への入力
- nnue.bin の局面 → NNUEExperts への入力（教師あり学習のターゲット含む）

NNUEバックボーン使用時は、dnn.bin の局面を NNUE 特徴量に変換してバックボーンに流す必要がある。nnue.bin の局面をバックボーンに流してはならない。

## 設計方針

`NNUEBackbone` クラスを新設し、`DNNBackbone` + `DNNAdapter` の役割を1つで担う。

- 標準NNUEの構造（input→256→512→32→32→**1**）の最終層を、**n_experts出力**の層に置き換える
- 最終層以外は学習済みNNUE重みで初期化、最終層はランダム初期化
- 全層を学習可能（frozenにしない）
- 出力にsoftmaxを適用してgate weightsとする

データフロー:
```
[NNUE backbone] us_bb, them_bb, w_in_bb, b_in_bb → gate_weights (batch, n_experts)
                 ↑ dnn.bin由来のNNUE特徴量
[NNUEExperts]   gate_weights, us, them, w_in, b_in → value (batch, 1)
                 ↑ nnue.bin由来のNNUE特徴量（従来通り）
```

## 変更ファイルと内容

### 1. `src/train_nnue/expert_blending_dataset.py`

**`ExpertBlendingDataset` / `_ExpertBlendingIterator` の修正:**
- `backbone_type` パラメータを追加（デフォルト `"dnn"`）
- `backbone_type="nnue"` の場合:
  - dnn.bin を読む 2つ目の `SparseBatchProvider` を作成し、dnn.bin の局面から NNUE 特徴量を生成
  - 出力タプルに backbone 用 NNUE 特徴量 `(us_bb, them_bb, white_bb, black_bb)` を追加
  - DNN dense features (`x1`, `x2`) の生成はスキップ（HCP変換不要 → 高速化）
- `backbone_type="dnn"` の場合: 従来通り（x1, x2 を生成、backbone用NNUE特徴量なし）

**出力タプルの変更:**
- DNN: `(x1, x2, us, them, white, black, outcome, score, ply)` — 変更なし
- NNUE: `(us_bb, them_bb, white_bb, black_bb, us, them, white, black, outcome, score, ply)`
  - x1, x2 の代わりに us_bb, them_bb, white_bb, black_bb を先頭に置く

**`create_data_loaders` の修正:**
- `backbone_type` パラメータを追加し、`ExpertBlendingDataset` に渡す

### 2. `src/train_nnue/expert_blending_model.py`

**新規クラス `NNUEBackbone`:**
- NNUE と同じ構造だが最終層が `Linear(L3, n_experts)` + softmax
- `noise_scale` パラメータ対応（DNNAdapterと同様）
- `forward(us, them, w_in, b_in, training)` → `(batch, n_experts)`
- 内部構造:
  - `self.input = nn.Linear(num_features, L1)` — L1=256
  - `self.l1 = nn.Linear(2*L1, L2)` — L2=32
  - `self.l2 = nn.Linear(L2, L3)` — L3=32
  - `self.output = nn.Linear(L3, n_experts)` — 最終層のみ形状変更
  - 活性化: clipped ReLU (clamp 0.0〜1.0)

**新規関数 `load_nnue_backbone`:**
- NNUEチェックポイントから重みを読み込み
- input/l1/l2 層は学習済み重みで初期化
- output 層は形状が異なる（1→n_experts）ためデフォルトの初期化のまま

**`ExpertBlendingModel` の修正:**
- `backbone_type` を保持（`"dnn"` or `"nnue"`）
- `forward` のシグネチャを変更: バッチタプルをそのまま受け取るか、backbone_type に応じて引数を使い分ける
  - DNN: `backbone(x1, x2)` → `adapter(feat)` → gate_weights（従来通り）
  - NNUE: `backbone(us_bb, them_bb, w_in_bb, b_in_bb)` → gate_weights

**`create_expert_blending_model` の修正:**
- `backbone_type` 引数を追加（デフォルト `"dnn"`）
- `"nnue"` の場合は `backbone_weights_path` を不要とし、`nnue_ckpt_path` からbackboneも初期化
- `"dnn"` の場合は従来通り

### 3. `src/train_nnue/train_expert_blending.py`

**コマンドライン引数の追加:**
- `--backbone-type`: `"dnn"` (default) or `"nnue"`

**既存引数の調整:**
- `--backbone-weights`: `--backbone-type nnue` のとき不要にする（required=False化、DNN時にバリデーション）

**`ExpertBlendingLightningModule` の修正:**
- `_compute_loss`: バッチのunpack方法を backbone_type に応じて分岐
- `configure_optimizers`: NNUE backboneの場合、backboneのパラメータも学習対象に追加（lr_adapterの学習率を使用）
- `_log_expert_weights`: backbone_type に応じてgate_weightsの取得方法を分岐
- `forward`: シグネチャを backbone_type に合わせて調整

**`create_data_loaders` 呼び出しの修正:**
- `backbone_type` を渡す

## 検証方法

テストデータ `tmp/headtest_data/{train,val}` を使い、既存スクリプトと同じハイパーパラメータ（`scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0.sh` ベース）で確認する。

NNUEチェックポイント: `logs/halfkp_v1/checkpoints/83000.ckpt`
DNNバックボーン重み: `tmp/dlshogi-model/model_resnet10_swish-072`

```bash
cd nnue-pytorch && source .venv/bin/activate

# NNUEバックボーンで学習実行（数エポック）
PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.train_expert_blending \
    --train ../tmp/headtest_data/train \
    --val ../tmp/headtest_data/val \
    --backbone-type nnue \
    --nnue-checkpoint ../logs/halfkp_v1/checkpoints/83000.ckpt \
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
    --max-epochs 5 \
    --gpus 1 \
    --seed 42 \
    --default-root-dir ../tmp/test_nnue_backbone

# DNNバックボーンで学習実行（既存動作の回帰確認）
PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.train_expert_blending \
    --train ../tmp/headtest_data/train \
    --val ../tmp/headtest_data/val \
    --backbone-type dnn \
    --backbone-weights ../tmp/dlshogi-model/model_resnet10_swish-072 \
    --nnue-checkpoint ../logs/halfkp_v1/checkpoints/83000.ckpt \
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
    --max-epochs 5 \
    --gpus 1 \
    --seed 42 \
    --default-root-dir ../tmp/test_dnn_backbone
```

確認項目:
- NNUEバックボーンで学習が開始しlossが減少すること
- DNNバックボーンの既存動作が壊れていないこと
- NNUEバックボーンの全パラメータが学習可能であること（requires_grad=True）
