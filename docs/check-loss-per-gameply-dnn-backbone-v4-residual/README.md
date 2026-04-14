# game_plyの差によるlossの違いの検証 (DNNバックボーン, v4 residual)

Residual blending mode の Expert Blending とベースライン単一NNUEの validation loss を
`delta = nnue_ply - dnn_ply` および `nnue_ply` で bin 分割して比較する。

実測のプロットとテーブルは residual 学習後の checkpoint で更新する。

## 実行コマンド

```bash
cd nnue-pytorch && source .venv/bin/activate
PYTHONPATH=../src:$PYTHONPATH python -u -m train_nnue.check_loss_per_gameply \
    --expert-blending-checkpoint /home/select766/shogi/train-nnue/logs/expert_blending_8experts_v4_paired_uniform50_noise0_residual/checkpoints/<EPOCH>.ckpt \
    --nnue-checkpoint /home/select766/shogi/modelarchive/train-tanuki/83000.ckpt \
    --val-dir /home/select766/shogi/train-nnue/dataset/split_v1_paired_uniform_50/val1 \
    --feature-set HalfKP \
    --max-positions 1000000 \
    --output /home/select766/shogi/train-nnue/docs/check-loss-per-gameply-dnn-backbone-v4-residual/loss_per_gameply.png
```

## 結果

### delta (nnue_ply - dnn_ply) 別

![delta別loss](loss_per_gameply_delta.png)

| delta | EB_loss | BL_loss | count |
|------:|--------:|--------:|------:|
| TBD | TBD | TBD | TBD |

### nnue_ply (絶対手数) 別

![nnue_ply別loss](loss_per_gameply_nnue_ply.png)

| nnue_ply | EB_loss | BL_loss | count |
|---------:|--------:|--------:|------:|
| TBD | TBD | TBD | TBD |
