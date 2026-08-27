# 候補手disagreement直接目的 結果 (2026-08-27)

## 結論

`H-SU-DIVERSITY-OBJECTIVE`は**棄却**した。selection 96 rootで固定半径4以上の有効教師率と
oracle gainを制約し、候補bestmove種類数を直接最大化したが、未使用held-out 96 rootでは固定半径4より
有効教師率、平均gain、候補手多様度がすべて低下した。多様度差のbootstrap 95%区間も全域負だった。

否定した範囲は、8 expertの各positive-axis方向から半径を1つずつ選ぶ9候補集合である。複数expertを
同時に動かす方向や、rootごとに候補集合自体を変える方式までは否定しない。

## 半径と条件

- 計画: [候補手disagreement多目的選択](../../experiments/disagreement-objective-20260827.md)
- root: `tmp/proxy_gap_v2/valA/roots.bin`の1,000--1,191番、計192件
- selection/held-out: 先頭96件 / 後半96件
- candidate/reference: `Threads=1`、各1,000,000 nodes
- 候補数: 基準gateと8 expert方向、常に計9
- 半径: `0.5, 1, 2, 4`

基準gate logitsを`z`、expert `i`の単位ベクトルを`e_i`とすると、半径`r`の候補は
`softmax(z + r e_i)`である。半径は探索深さや指し手間の距離ではなく、gate logit空間の正方向距離を表す。

## 共通utility再採点

最初の4 runでは、同じ基準手の制限手scoreがrun間で完全一致せず、安全検査で結合解析を停止した。
原因は別process間の探索履歴差と考えられる。selection/held-outの値を見る前に、各rootで4半径の全unique
候補手を共通の固定`nn.bin`で再度1,000,000-node制限手探索し、同じ手には1つの共通utilityを割り当てた。
これは事前計画からの実装上の逸脱だが、異なるrunの候補を同一尺度で比較するために必要だった。

## 選定とheld-out

selectionで選ばれたexpert 0--7の半径は`[4, 2, 4, 4, 1, 4, 2, 4]`だった。

| split | 候補集合 | 有効教師率 | oracle gain平均 | 候補手多様度 |
|---|---|---:|---:|---:|
| selection | 固定半径4 | 11.46% | 307.32cp | 1.979 |
| selection | disagreement選択 | 11.46% | 307.32cp | **2.021** |
| held-out | 固定半径4 | **23.96%** | **22.93cp** | **2.115** |
| held-out | disagreement選択 | 21.88% | 19.27cp | 1.917 |

held-outの選択minus固定差は次のとおりだった。

- 有効教師率: `-0.02083`、bootstrap 95%区間`[-0.05208, 0.00000]`
- oracle gain平均: `-3.65625cp`
- 候補手多様度: `-0.19792`、bootstrap 95%区間`[-0.31250, -0.09375]`

多様度を直接選定目的にしても未使用rootへ転移せず、utility制約もheld-outでは維持されなかった。
次は固定されたexpert別一軸半径ではなく、複数expertを同時に動かす候補方向自体を学ぶ必要がある。

## 成果物

- 収集: `scripts/run_disagreement_objective.sh`
- 共通再採点: `scripts/rescore_disagreement_pool.py`
- 選定・判定: `scripts/analyze_disagreement_objective.py`
- 集計: `results/disagreement_objective_20260827.json`
- raw: `tmp/disagreement_objective_20260827/`
- logs: `/tmp/disagreement_objective_*.log`
