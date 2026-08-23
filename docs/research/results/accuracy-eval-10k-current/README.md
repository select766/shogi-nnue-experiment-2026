# 固定10,000局面の最善手一致率評価 (2026-08-23)

## 結論

checkpoint 180のExpert Blendingは、固定test 10,000局面・`Threads=1`・100万ノードで
ベースラインHalfKPを0.45ポイント上回った。ただし対応ありの正確McNemar検定は
`p=0.314924`で、優位性は確認できない。

| モデル | 一致数 | 正解率 | Wilson 95%区間 |
|---|---:|---:|---:|
| HalfKP checkpoint 83000 | 6,287/10,000 | 62.87% | 61.92--63.81% |
| Expert Blending checkpoint 180 | 6,332/10,000 | 63.32% | 62.37--64.26% |

対応あり集計:

| 両方正解 | baselineのみ | Expertのみ | 両方不正解 | 差 | 正確McNemar p |
|---:|---:|---:|---:|---:|---:|
| 5,351 | 936 | 981 | 2,732 | +0.45 pt | 0.314924 |

したがって、旧1,000局面で観測した-1.1ポイントは固定10,000局面では再現しなかったが、
現モデルがベースラインより強いとも結論できない。

## 評価条件

- dataset: `data/accuracy_eval_10k/test.jsonl`
- search: `Threads=1`, 1,000,000 nodes、4 engine workers
- baseline: `logs/halfkp_v1/checkpoints/83000.ckpt`から変換済みの`bin/eval/nn.bin`
- Expert Blending: `logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt`
- Expert Blending推論: やねうら王プロセス内のCPU版ONNX Runtime。Python推論サーバーとCUDAは不使用
- release: `tmp/expert_blending_release/{backbone.onnx,head.bin,head.json}`

## 層別結果

教師評価値絶対値による層別は探索的な結果であり、多重比較補正はしていない。

| 教師評価値絶対値 | 局面数 | baseline | Expert | 差 | McNemar p |
|---|---:|---:|---:|---:|---:|
| 0--500 | 6,330 | 60.79% | 60.88% | +0.09 pt | 0.8878 |
| 501--2000 | 2,987 | 67.22% | 67.83% | +0.60 pt | 0.4577 |
| 2001以上 | 683 | 63.10% | 66.18% | +3.07 pt | 0.0871 |

入玉局面は44件だけで、baseline 26正解、Expert 21正解だった。対応差は-11.36ポイント、
`p=0.0625`だが、件数が小さいためこの結果だけで一般化しない。HCPEには実手数がないため
`game_ply`層別は利用不可である。

## 成果物

- `results/accuracy_eval_10k_halfkp_v1.json`
- `results/accuracy_eval_10k_expert_blending_current.json`
- `results/accuracy_comparison_10k_expert_blending_current.json`
- logs: `/tmp/eval_accuracy_accuracy_eval_10k_halfkp_v1.log`,
  `/tmp/eval_accuracy_accuracy_eval_10k_expert_blending_current.log`
