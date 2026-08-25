# Expert多様性ボトルネック 結果 (2026-08-25)

## 結論

`H-EXPERT-DIVERSITY`は事前規則により**棄却**した。現M0 expertの平均を固定してexpert偏差だけを2倍へ
拡大すると、出力相関平均は`0.9667`から`0.8925`へ下がり、実探索leaf group lossの悪化も
`+0.000975`で許容上限`+0.002`内だった。しかし同一64 root・100万nodesのsearch utilityでは、
有効教師率、oracle gain、候補手多様度のすべてが悪化した。

この結果は「どのような多様化も無効」という意味ではない。否定されたのは、現在学習済みexpert間差の方向を
放射状に広げる介入である。rootから予測可能な役割を持つtask-aligned specializationは
`H-TASK-ALIGNED-EXPERTS`として分離する。

## 介入と静的診断

計画は[Expert多様性ボトルネック](../../experiments/expert-diversity-20260825.md)を参照する。各expert
パラメータを`W'_k = mean(W) + alpha * (W_k - mean(W))`で変換し、adapterとgateを固定した。

| 条件 | expert相関平均 | paired blended loss | M0との差 | expert値分散平均 |
|---|---:|---:|---:|---:|
| M0 | 0.96667 | 0.035612 | - | 92,639 |
| alpha 1.5 | 0.93305 | 0.036322 | +0.000710 | 176,488 |
| alpha 2.0 | **0.89246** | 0.037338 | +0.001726 | 273,169 |

変換対象8テンソルのexpert平均最大誤差はalpha 1.5で`4.77e-7`、alpha 2.0で`3.58e-7`だった。
gateは完全に同じである。

実探索leaf `tmp/proxy_gap_v2/valA` 9,984 rootの対応比較:

| 条件 | group loss | M0との差 (root bootstrap 95% CI) | 改善root |
|---|---:|---:|---:|
| M0 | 0.016906 | - | - |
| alpha 1.5 | 0.017221 | +0.000315 `[+0.000232,+0.000388]` | 31.45% |
| alpha 2.0 | 0.017880 | +0.000975 `[+0.000802,+0.001125]` | 25.16% |

alpha 2.0だけが相関0.90以下を満たし、loss悪化も事前上限内なのでsearch utilityへ進めた。

## 100万nodes search utility

H-SU-HORIZONの結果に従い、候補探索と独立制限手utilityをともに1,000,000 nodesへ揃えた。
rootは前実験と重ならない`tmp/search_utility_v1/val/roots.bin`の128--191番、計64件である。

| 条件 | 有効教師率 | oracle gain平均 | oracle gain p90 | 候補手多様度平均 |
|---|---:|---:|---:|---:|
| M0 | 10/64 (15.63%) | 16.09cp | 35.5cp | 1.656 |
| alpha 2.0 | 5/64 (7.81%) | 7.55cp | 1.4cp | 1.578 |

事前の支持条件はutility指標の相対25%以上改善かつ多様度10%以上改善だったが、実際は有効教師率
`-50.0%`、p90 `-96.1%`、多様度`-4.7%`だった。相関低下は必要十分条件ではなく、expert差の方向と
各expert単体の評価品質が重要である。alpha 2.0では単体expert lossも悪化しており、単なる距離拡大は
有用な選択肢を作らない。

## 成果物

- 変換: `scripts/scale_expert_deviations.py`
- checkpoint: `tmp/expert_diversity_alpha{15,20}.ckpt`
- gate診断: `results/expert_diversity_{m0,alpha15,alpha20}_diagnostics.json`
- group loss: `results/expert_diversity_group_loss.json`
- utility集計: `results/expert_diversity_utility_{m0,alpha20}.json`
- raw utility details: `tmp/expert_diversity_utility_{m0,alpha20}/details.jsonl`
