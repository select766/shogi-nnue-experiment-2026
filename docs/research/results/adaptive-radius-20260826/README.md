# Root別search utility候補半径 結果 (2026-08-26)

## 結論

`H-SU-ADAPTIVE-RADIUS`は**棄却**した。root特徴から選ぶ深さ1の半径policyは、held-out 96 rootで
固定半径4より有効教師率を`+5.21`ポイント改善したが、候補手多様度を`-0.3125`悪化させた。
多様度差のbootstrap 95%区間は`[-0.5000, -0.1354]`であり、「有効教師率と候補手多様度をともに
改善する」という事前条件を満たさない。

この判定は、候補数9、positive-axis、半径`0.5, 1, 2, 4`、低コストroot/gate特徴、深さ3以下の
決定木という範囲に対するものである。

## 条件

- 計画: [Root別search utility候補半径](../../experiments/adaptive-radius-20260826.md)
- root: `tmp/search_utility_v1/val/roots.bin`の192--383番、計192件
- model: M0、8 experts、基準を含むpositive-axis 9候補
- candidate/reference: `Threads=1`、各1,000,000 nodes
- 半径: logit bias `0.5, 1, 2, 4`
- selection/held-out: 先頭96 root / 後半96 root
- policy: game ply、手番、盤上駒率、持駒率、成駒率、合法手率、gate entropy/max/marginから
  深さ1--3を4-fold CVで選択

## 半径別診断

192 root全体の値を示す。

| 半径 | 有効教師率 | oracle gain平均 | 正gain率 | 候補手多様度 |
|---:|---:|---:|---:|---:|
| 0.5 | 19.79% | 90.92cp | 25.00% | 1.755 |
| 1 | **20.31%** | 88.40cp | 23.44% | 1.839 |
| 2 | 19.79% | **241.21cp** | 24.48% | 1.870 |
| 4 | 19.79% | 89.75cp | 23.96% | **2.156** |

selectionでは固定対照に半径4、適応policyに深さ1が選ばれた。木はgate entropy `1.29666`以下で
半径2、それより大きければ半径0.5を選ぶ。

## Held-out結果

| policy | 有効教師率 | oracle gain平均 | 候補手多様度 |
|---|---:|---:|---:|
| 固定半径4 | 16.67% | 153.49cp | **2.135** |
| 適応半径 | **21.88%** | 155.85cp | 1.823 |
| root別oracle | 23.96% | 457.66cp | 1.844 |

適応minus固定の有効教師率差は`+0.0521`、95%区間`[0.0000, 0.1042]`だった。一方、多様度差は
`-0.3125`、95%区間`[-0.5000, -0.1354]`で明瞭な逆方向だった。oracle自体も固定半径4より
多様度が低く、gain最大を教師ラベルにした半径選択では両目的を同時に満たせないことが分かった。

## 成果物

- 収集: `scripts/run_adaptive_radius.sh`
- policy解析: `scripts/analyze_adaptive_radius.py`
- 集計: `results/adaptive_radius_20260826.json`
- raw: `tmp/adaptive_radius_20260826/radius_*/details.jsonl`
- logs: `/tmp/adaptive_radius_*.log`
