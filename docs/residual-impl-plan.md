# 残差モデル (Residual blending mode) の実装計画

## Context

現行の Expert Blending 学習 (`scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0.sh`) では、
`N_EXPERTS` 個の NNUE 重み `C_k` をベースライン `B` で初期化し、gate_weights `w_k` で加重平均した合成重み
`W = Σ C_k * w_k` を学習している。本タスクでは、これを残差形式に拡張する:

- 初期化: `C_k ← 0` (ゼロ初期化)
- 推論/学習: `W = B + Σ C_k * w_k`
- `B` は学習対象外 (ベースライン NNUE 重みを凍結)

従来モード (weighted) との切り替えはオプションで行い、既存機能は保持する。
評価は `docs/check-loss-per-gameply-dnn-backbone-v4/README.md` と同等の `check_loss_per_gameply` 出力が得られるようにする。

## 方針

- `NNUEExperts` に **blend_mode** を導入 ("weighted" / "residual")。
- residual モード時、ベースライン `B` を **凍結 Parameter** (`requires_grad=False`) として保持し、state_dict に含める。
  → チェックポイントが自己完結し、評価時に baseline を別途渡す必要がない。
- 学習の forward は `F.linear(x, B, b_B) + Σ_k g_k * F.linear(x, C_k, b_{C,k})` に変更。
  softmax により `Σ g_k = 1` が保たれるので、等価式 `W_total = B + Σ g_k * C_k` が成立する。
- backbone_type (dnn / nnue) と直交したオプションとして実装する。

## 実装内容

### 1. `src/train_nnue/expert_blending_model.py`

- `NNUEExperts.__init__(n_experts, num_features, blend_mode="weighted")` に引数追加。
  - `blend_mode="residual"` の場合、`base_input_weight / base_input_bias / base_l1_weight / base_l1_bias / base_l2_weight / base_l2_bias / base_output_weight / base_output_bias` を `nn.Parameter(..., requires_grad=False)` として登録。shape は非 n_experts 軸 (例: `(L1, num_features)`)。
  - `blend_mode="weighted"` では base パラメータは登録しない (従来と完全一致)。
- `NNUEExperts.forward`:
  - `_blended_linear(x, weight, bias, gate_weights, base_weight=None, base_bias=None)` に拡張。
  - base 系が非 None なら `F.linear(x, base_weight, base_bias)` を加算。
  - 各層呼び出しで `self.base_input_weight` 等を渡す (residual 時のみ)。
- `load_nnue_experts(ckpt_path, n_experts, feature_set, blend_mode="weighted")`:
  - NNUE チェックポイントから state を取り出した後、
    - weighted: 既存通り各 expert に複製。
    - residual: 各 expert 重みを**ゼロで据え置き**、`base_*` パラメータに NNUE 重みをコピー。HalfKP→HalfKP^ パディング処理は従来と同じ方針を base_input_weight にも適用。
- `create_expert_blending_model(...)` に `blend_mode="weighted"` を追加し、`NNUEExperts` と `load_nnue_experts` に伝搬。

### 2. `src/train_nnue/train_expert_blending.py`

- `--blend-mode {weighted,residual}` (default `weighted`) を追加し、`create_expert_blending_model` に渡す。
- optimizer は `nnue_experts.parameters()` を使っているが、base_* は `requires_grad=False` なので optimizer に入っても勾配が流れない (念のため `filter(lambda p: p.requires_grad, ...)` で明示的に除外)。
- その他は既存ロジックそのまま。

### 3. `src/train_nnue/check_loss_per_gameply.py`

- `load_expert_blending_model` 内で `state_dict` に `model.nnue_experts.base_input_weight` が含まれるかで blend_mode を判定。
- `NNUEExperts` の生成時に `blend_mode` を渡してから `load_state_dict` する (base_* キーを受け取れるようにするため)。
- それ以外 (ExpertBlendingModel の組立て、per-record loss 計算、bin 集計、プロット、テーブル出力) は変更不要。
- → 既存の実行コマンド (`README.md` 記載) がそのまま residual checkpoint にも適用できる。

### 4. `src/train_nnue/blend_and_export.py`

- `blend_expert_weights(nnue_experts, gate_weights)`:
  - 現行は `nnue_experts.<param>` を加重平均。
  - residual 判定 (`hasattr(nnue_experts, 'base_input_weight')` または `nnue_experts.blend_mode == "residual"`) して、**加重平均結果に base_* を加算**。
- やねうら王に渡す量子化バイナリは、この blended 辞書を使うので追加修正なし。
- CLI (`__main__`): checkpoint の state_dict に base_* が含まれるかで residual を判定し、`NNUEExperts(n_experts, num_features, blend_mode=...)` を生成して `load_state_dict`。

### 5. `src/train_nnue/check_loss_per_expert.py`

- 各 expert を単独 NNUE として評価する際、residual の場合は `base + expert_k` を実効 NNUE 重みとして組み立てる。
- チェックポイントに base_* があるか判定し、`expert_state` 構築時に base を加算する分岐を追加。
- 通常モードでは従来どおり。

### 6. 実行スクリプト (新規)

- `scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0_residual.sh` を新規作成 (既存スクリプトのコピーに `--blend-mode residual` を追加し、`LOGDIR` / `LOG_FILE` を別名に)。
- `scripts/run_train_expert_blending_8experts_v4.sh` (内部ランナー) は既存のまま (TRAIN_ARGS でそのまま通る)。

### 7. 評価ドキュメント

- `docs/check-loss-per-gameply-dnn-backbone-v4-residual/README.md` を追加予定地として明記 (書式は v4 と同等: 実行コマンド, delta 別テーブル, nnue_ply 別テーブル)。
  - 本実装タスクでは雛形のみ準備してもよいが、実測値のプロット/テーブルは学習完了後に記入。

## 変更/追加ファイル

- [M] `src/train_nnue/expert_blending_model.py`
- [M] `src/train_nnue/train_expert_blending.py`
- [M] `src/train_nnue/blend_and_export.py`
- [M] `src/train_nnue/check_loss_per_gameply.py`
- [M] `src/train_nnue/check_loss_per_expert.py`
- [A] `scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0_residual.sh`
- [A] `docs/check-loss-per-gameply-dnn-backbone-v4-residual/README.md` (雛形)

## 再利用する既存コンポーネント

- `expert_blending_model.py:_extract_nnue_state_dict` — base パラメータへのロードに再利用。
- `expert_blending_model.py:load_backbone / load_nnue_backbone` — backbone 構築ロジックは不変。
- `train_expert_blending.py:CheckpointEveryNEpochs` — 保存ロジック不変。
- `expert_blending_dataset.py:create_data_loaders` — データローダも不変。
- `check_loss_per_gameply.py:compute_per_record_loss / _bin_and_aggregate / _plot_loss_chart` — 評価ロジック不変。
- `blend_and_export.py:coalesce_ft_weights / quantize_and_pack / write_nnue_file` — 量子化ロジック不変。

## 等価性の確認ポイント

- `blend_mode="weighted"` (デフォルト) のときに、state_dict のキー、forward 出力、ロス、チェックポイント互換性が従来と完全一致すること。
  - → `base_*` パラメータは登録されず、`_blended_linear` にも base_* は渡されない分岐で保つ。
- `blend_mode="residual"` 初期状態 (ステップ 0) では `C_k=0` のため、`W = B + 0 = B` であり、モデル出力はベースライン NNUE と一致することを確認する。

## 検証手順

1. **単体動作 (weighted 既存回帰)**: 
   `bash scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0.sh` を数エポックだけ試走し、logs 生成と val_loss 進行を確認。state_dict のキー集合が従来と一致することを確認。
2. **単体動作 (residual 新規)**: 
   `bash scripts/run_train_expert_blending_8experts_v4_paired_uniform50_noise0_residual.sh` を数エポック試走。
   - ステップ 0 時点の val_loss がベースライン NNUE (83000.ckpt) と近似一致すること。
   - TensorBoard で train_loss が下がること。
3. **評価 (residual)**: 学習後 checkpoint に対して
   ```bash
   PYTHONPATH=../src:$PYTHONPATH python -u -m train_nnue.check_loss_per_gameply \
       --expert-blending-checkpoint <residual_checkpoint> \
       --nnue-checkpoint /home/select766/shogi/modelarchive/train-tanuki/83000.ckpt \
       --val-dir /home/select766/shogi/train-nnue/dataset/split_v1_paired_uniform_50/val1 \
       --feature-set HalfKP \
       --max-positions 1000000 \
       --output docs/check-loss-per-gameply-dnn-backbone-v4-residual/loss_per_gameply.png
   ```
   delta/nnue_ply 別プロットと集計テーブルが v4 と同じ形式で出力されることを確認。
4. **エクスポート**: `python -m train_nnue.blend_and_export --checkpoint <residual_ckpt> --features HalfKP --output <nnue>` で residual 合成重みを量子化出力し、やねうら王でロード可能であること。
5. **expert 単独評価**: `python -m train_nnue.check_loss_per_expert` が residual モードでは `base + expert_k` の合成 NNUE として評価することを確認。
