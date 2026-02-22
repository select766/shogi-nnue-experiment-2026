# dnn_inference_server.py に NNUEBackbone 対応を追加する実装計画

## Context

`blending-head-nnue-impl-plan.md` に基づき、`NNUEBackbone` を用いた Expert Blending モデルの学習が実装済み。しかし推論サーバー (`dnn_inference_server.py`) は `DNNBackbone` + `DNNAdapter` のみ対応しており、`NNUEBackbone` で学習したチェックポイントを推論に使えない。本変更で `backbone_type="nnue"` に対応する。

## 変更対象ファイル

`src/train_nnue/dnn_inference_server.py` のみ

## 変更内容

### 1. import 追加

- `NNUEBackbone` を `expert_blending_model` から追加 import
- `nnue_dataset.py` の `make_sparse_batch_from_fens` をインポート（SFEN → NNUE スパース特徴量変換に使用）

```python
from train_nnue.expert_blending_model import (
    DNNBackbone, DNNAdapter, NNUEBackbone, NNUEExperts,
    ExpertBlendingModel, load_backbone,
)
```

`make_sparse_batch_from_fens` のインポートは `nnue-pytorch` ディレクトリの `nnue_dataset.py` から行う。`load_nnue_feature_set` と同様に `sys.path` 操作後にインポートするか、feature_set 解決時にまとめてインポートする。

### 2. `load_model_from_checkpoint` の修正

`backbone_type` パラメータを追加し、分岐:

**`backbone_type="nnue"` の場合:**
- `backbone_weights_path` は不要（None を許容）
- チェックポイント state_dict から `model.backbone.*` キーを取り出して `NNUEBackbone` を復元
  - `num_features` は `state_dict['model.backbone.input.weight'].shape[1]` から推定
  - `n_experts` は `state_dict['model.backbone.output.weight'].shape[0]` から推定
- `adapter = None`（NNUE backbone は adapter 不要）
- 戻り値: `(backbone, None, nnue_experts)`

**`backbone_type="dnn"` の場合:**
- 従来通り（変更なし）
- 戻り値: `(backbone, adapter, nnue_experts)`

### 3. `infer_gate_weights` の修正（または新規関数追加）

`backbone_type="nnue"` 用の推論パスを追加:

```python
def infer_gate_weights_nnue(backbone, board, feature_set, device='cpu'):
    sfen = board.sfen()
    batch = make_sparse_batch_from_fens(feature_set, [sfen], [0], [0], [0])
    us, them, white, black, outcome, score, ply = batch.get_tensors(device)
    with torch.no_grad():
        gate_weights = backbone(us, them, white, black, training=False)
    return gate_weights[0]  # (n_experts,)
```

- `make_sparse_batch_from_fens` で SFEN 1件からスパースバッチを生成
- `scores=[0], plies=[0], results=[0]` はダミー（特徴量生成に不使用）
- `SparseBatch.get_tensors()` が返す `(us, them, white, black, ...)` をそのまま `NNUEBackbone.forward` に渡す

### 4. `main()` の修正

**引数の変更:**
- `--backbone-type` 追加: choices=`["dnn", "nnue"]`, default=`"dnn"`
- `--backbone-weights` を `required=False` に変更（`backbone_type="dnn"` 時にのみ必須、バリデーションで確認）

**モデルロードの分岐:**
- `backbone_type` を `load_model_from_checkpoint` に渡す
- NNUE backbone の場合、`adapter` は None なので `.eval()` 呼び出しをスキップ

**メインループの分岐:**
- `backbone_type="nnue"`: `infer_gate_weights_nnue(backbone, board, feature_set, device)` を呼ぶ
- `backbone_type="dnn"`: 従来通り `infer_gate_weights(backbone, adapter, board, device)` を呼ぶ

**`make_sparse_batch_from_fens` のインポート:**
- `load_nnue_feature_set` と同じパス解決ロジックを使い、`nnue_dataset` モジュールからインポート
- `backbone_type="nnue"` のときのみ必要

### 5. プロトコルへの影響

変更なし。出力フォーマット（gate_weights float32 x 8 + size uint32 + weight_bytes）は backbone_type に依存しない。

## 使用例

```bash
# NNUEバックボーンモデルで推論サーバーを起動
PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.dnn_inference_server \
    --checkpoint <expert_blending_nnue_backbone.ckpt> \
    --backbone-type nnue \
    --features HalfKP \
    --n-experts 8

# DNNバックボーンモデル（従来通り）
PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.dnn_inference_server \
    --checkpoint <expert_blending_dnn_backbone.ckpt> \
    --backbone-type dnn \
    --backbone-weights <dlshogi_model.npz> \
    --features HalfKP \
    --n-experts 8
```

## 検証方法

1. NNUEバックボーンで学習済みのチェックポイントを用意
2. `--backbone-type nnue` で推論サーバーを起動し、`ready` が出力されることを確認
3. SFEN を stdin に送り、gate_weights とバイナリ重みが正常に返ることを確認
4. `--backbone-type dnn` で従来通り動作することを確認（回帰テスト）
