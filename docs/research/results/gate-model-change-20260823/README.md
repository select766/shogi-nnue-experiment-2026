# Gateモデル変更実験 (2026-08-23)

## 結論

checkpoint 510のsoftmax gateに対する温度変更とentropy正則化は、validation lossを維持したまま
実効expert数を2--4へ下げられなかった。1.5-entmaxへ変更して6 epoch fine-tuneした候補は、
独立10,000 validation positionsでsoftmaxと同等のlossを維持しつつ、実効expert数を
6.13から3.92へ下げた。死んだexpertはない。

## 共通条件

- 初期値: `logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt`
- DNN backbone、8 experts、weighted blend、uniform-50 paired data
- validation: `dataset/split_v1_paired_uniform_50/val1`先頭10,000 paired positions
- 短期学習: 100,000 positions/epoch、batch 256、同一seed 42
- fine-tune LR: NNUE `0.001`、adapter `0.01`

## 温度変更

再学習なしでgate確率を温度変換し、変換後の重みでNNUEを再合成した。

| 温度 | loss | T=1との差 | 実効expert | 最大重み |
|---:|---:|---:|---:|---:|
| 1.00 | 0.033790 | - | 6.13 | 0.306 |
| 0.75 | 0.033786 | -0.000004 | 5.39 | 0.365 |
| 0.50 | 0.034200 | +0.000410 | 4.23 | 0.470 |
| 0.25 | 0.036586 | +0.002796 | 2.50 | 0.677 |

`T=0.75`の差は実質ゼロで疎性が不足し、目標に近い`T=0.5`ではlossが悪化したため、
温度だけのfine-tuneは実施しなかった。

## Entropy + balance

`L = L_task + lambda_sparse H(p) + lambda_balance KL(mean(p) || Uniform)`を実装し、
8条件を各3 epoch比較した。

| lambda_sparse | lambda_balance | 最終loss | 実効expert | 最大重み | 判定 |
|---:|---:|---:|---:|---:|---|
| 0 | 1e-3 | 0.033852 | 6.22 | 0.298 | 過密 |
| 0 | 1e-2 | 0.033958 | 6.43 | 0.278 | 過密・loss悪化 |
| 1e-4 | 1e-3 | 0.033852 | 6.20 | 0.300 | 過密 |
| 1e-4 | 1e-2 | 0.033958 | 6.42 | 0.278 | 過密・loss悪化 |
| 1e-3 | 1e-3 | 0.033857 | 6.03 | 0.315 | 過密 |
| 1e-3 | 1e-2 | 0.033969 | 6.30 | 0.286 | 過密・loss悪化 |
| 1e-2 | 1e-3 | 0.036099 | 1.04 | 0.995 | collapse |
| 1e-2 | 1e-2 | 0.034212 | 5.02 | 0.417 | 目標外・loss悪化 |

強いentropy係数は弱いbalanceでは単一expertへcollapseし、強いbalanceではcollapseを防ぐ代わりに
疎性不足となった。選抜基準を通る条件はなかった。

## 1.5-entmax

softmaxを厳密なゼロを生成できる1.5-entmaxへ置換した。再学習なしでは実効expert 3.42、
loss 0.034640だった。balanceを3条件で各3 epoch比較した結果は次のとおり。

| lambda_balance | 最終loss | 実効expert | 最大重み | 周辺KL |
|---:|---:|---:|---:|---:|
| 0 | 0.033872 | 3.86 | 0.440 | 0.247 |
| 1e-3 | 0.033909 | 4.06 | 0.419 | 0.163 |
| 1e-2 | 0.034150 | 4.37 | 0.390 | 0.044 |

balanceなしをさらに3 epoch学習した最終候補の独立診断結果:

| 指標 | softmax checkpoint 510 | entmax候補 |
|---|---:|---:|
| blended loss | 0.0337900 | 0.0337878 |
| entropy | 1.7969 | 1.3321 |
| 実効expert | 6.13 | 3.92 |
| 最大重み | 0.306 | 0.435 |
| top-2 mass | 0.519 | 0.694 |
| dead experts | なし | なし |

entmax候補のargmax利用率は
`[0.49, 29.29, 6.05, 2.50, 49.41, 1.61, 3.56, 7.09]%`で、利用偏りは残る。
expert単独出力相関は平均0.9685だった。gate top-1とoracleの一致率は19.59%、周辺分布からの
偶然期待14.20%に対するliftは1.38である。

候補checkpoint:
`logs/gate_entmax15_continue3_b0_from_short3/checkpoints/2.ckpt`

## 固定10,000局面の最善手一致率

entmax候補だけをCPU版ONNX Runtimeを組み込んだやねうら王へexportし、固定test、
`Threads=1`、100万nodes、4 workersで評価した。

| モデル | 一致数 | 正解率 | Wilson 95%区間 | HalfKP差 |
|---|---:|---:|---:|---:|
| HalfKP checkpoint 83000 | 6,287 | 62.87% | 61.92--63.81% | - |
| dense checkpoint 180 | 6,332 | 63.32% | 62.37--64.26% | +0.45 pt |
| entmax候補 | 6,304 | 63.04% | 62.09--63.98% | +0.17 pt |

HalfKPとの対応あり比較は、両方正解5,342、HalfKPのみ945、entmaxのみ962、両方不正解
2,751で、正確McNemar検定は`p=0.714084`だった。dense checkpoint 180との比較は、denseのみ
925、entmaxのみ897、差-0.28ポイント、`p=0.527044`だった。

したがって、entmaxはvalidation lossを維持してgateを明確に疎化できたが、最善手一致率の
改善には転換されなかった。HalfKPとの差は正方向だが統計的な優位性はなく、既存dense候補も
上回っていない。hard top-kへ進む根拠にはせず、次はoracleとの不一致を直接扱うrouter蒸留や
ranking lossを優先する。

## 成果物

- `results/gate_temperature_sweep_8experts_v4_510.json`
- `results/gate_regularization_short3_from510.json`
- `results/gate_entmax15_short3_from510.json`
- `results/gate_diagnostics_entmax15_continue3_epoch2.json`
- `results/accuracy_eval_10k_expert_blending_entmax15_candidate.json`
- `results/accuracy_comparison_10k_entmax15_vs_halfkp.json`
- `results/accuracy_comparison_10k_entmax15_vs_dense180.json`
