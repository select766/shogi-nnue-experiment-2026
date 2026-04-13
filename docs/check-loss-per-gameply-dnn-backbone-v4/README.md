# game_plyの差によるlossの違いの検証 (DNNバックボーン, v4 checkpoint 400)

Expert Blending (DNNバックボーン, checkpoint 400) とベースライン単一NNUEの validation loss を
`delta = nnue_ply - dnn_ply` および `nnue_ply` でbin分割して比較した。

エンジン設定は `configs/accuracy_eval_expert_blending_uniform_50_v4.json` に対応。

## 実行コマンド

```bash
cd nnue-pytorch && source .venv/bin/activate
PYTHONPATH=../src:$PYTHONPATH python -u -m train_nnue.check_loss_per_gameply \
    --expert-blending-checkpoint /home/select766/shogi/train-nnue/logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/400.ckpt \
    --nnue-checkpoint /home/select766/shogi/modelarchive/train-tanuki/83000.ckpt \
    --val-dir /home/select766/shogi/train-nnue/dataset/split_v1_paired_uniform_50/val1 \
    --feature-set HalfKP \
    --max-positions 1000000 \
    --output /home/select766/shogi/train-nnue/docs/check-loss-per-gameply-dnn-backbone-v4/loss_per_gameply.png
```

- データ: `dataset/split_v1_paired_uniform_50/val1` の先頭100万レコード (40B/record ペア形式)
- Expert Blending (DNNバックボーン): `logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/400.ckpt`
- ベースラインNNUE: `modelarchive/train-tanuki/83000.ckpt`

## 結果

### delta (nnue_ply - dnn_ply) 別

![delta別loss](loss_per_gameply_delta.png)

| delta | EB_loss | BL_loss | count |
|------:|--------:|--------:|------:|
| 0.5 | 0.001793 | 0.002035 | 8172 |
| 1.5 | 0.016366 | 0.018547 | 48334 |
| 2.5 | 0.019053 | 0.021518 | 40240 |
| 3.5 | 0.021470 | 0.024350 | 36180 |
| 4.5 | 0.022514 | 0.025063 | 33769 |
| 5.5 | 0.025525 | 0.027907 | 31271 |
| 6.5 | 0.026275 | 0.028596 | 29826 |
| 7.5 | 0.027303 | 0.029401 | 28300 |
| 8.5 | 0.029514 | 0.031527 | 27342 |
| 9.5 | 0.029147 | 0.031011 | 26009 |
| 10.5 | 0.029895 | 0.031775 | 25415 |
| 11.5 | 0.029824 | 0.031715 | 24391 |
| 12.5 | 0.032611 | 0.034400 | 24053 |
| 13.5 | 0.033523 | 0.035276 | 23002 |
| 14.5 | 0.033664 | 0.035159 | 22538 |
| 15.5 | 0.033265 | 0.034722 | 21914 |
| 20.5 | 0.038810 | 0.039966 | 19580 |
| 25.5 | 0.043707 | 0.044674 | 17174 |
| 30.5 | 0.045333 | 0.046292 | 16159 |
| 35.5 | 0.047325 | 0.048387 | 14898 |
| 40.5 | 0.046944 | 0.048128 | 13777 |
| 45.5 | 0.047406 | 0.048296 | 12647 |
| 50.5 | 0.049063 | 0.049800 | 11922 |

- 全delta帯でExpert Blending (DNNバックボーン) がベースラインより低いloss
- NNUEバックボーン版と比較して改善幅が大きい（例: delta=2.5で約0.0025 vs 約0.001）
- delta増大に伴うloss増加傾向は他のバックボーンと同様

### nnue_ply (絶対手数) 別

![nnue_ply別loss](loss_per_gameply_nnue_ply.png)

| nnue_ply | EB_loss | BL_loss | count |
|---------:|--------:|--------:|------:|
| 7 | 0.000480 | 0.000556 | 14028 |
| 17 | 0.001267 | 0.001415 | 39585 |
| 27 | 0.003674 | 0.003943 | 64871 |
| 37 | 0.009563 | 0.010059 | 81556 |
| 47 | 0.019526 | 0.020381 | 81415 |
| 57 | 0.029697 | 0.030928 | 81378 |
| 67 | 0.038288 | 0.039767 | 81490 |
| 77 | 0.044960 | 0.046814 | 80431 |
| 87 | 0.050430 | 0.052418 | 78999 |
| 97 | 0.052447 | 0.054712 | 75942 |
| 107 | 0.054059 | 0.056210 | 69571 |
| 117 | 0.053659 | 0.055827 | 61200 |
| 127 | 0.052303 | 0.054125 | 50152 |
| 137 | 0.049084 | 0.051454 | 39236 |
| 147 | 0.047224 | 0.048879 | 29447 |

- 全ply帯でExpert Blending (DNNバックボーン) がベースラインより低いloss
- 改善幅はply 90~120 (中盤) で最大（約0.002の差）
- NNUEバックボーン版（改善幅 最大約0.0015）より改善幅が大きい
