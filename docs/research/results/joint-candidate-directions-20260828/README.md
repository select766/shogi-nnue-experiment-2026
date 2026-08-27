# Joint signed候補方向 結果 (2026-08-28)

## 結論

`H-SU-JOINT-DIRECTIONS`は事前規則により**棄却**した。positive-axis半径4の8方向と、複数expertを
同時に増減するsigned Hadamard 14方向から同じ8方向を選ぶと、selectionでは有効教師率、平均gain、
候補手多様度の3指標が固定axis集合を上回った。しかし未使用held-out 384 rootでは有効教師率差`0`、
平均gain差`-37.625cp`、候補手多様度差`-0.00521`となった。joint方向を加えるだけではselectionの
coverage改善を未知rootへ転移できない。

## 条件

- 計画: [Joint signed候補方向](../../experiments/joint-candidate-directions-20260827.md)
- root: `tmp/proxy_gap_v2/valA/roots.bin`の1,200--1,967番、768件
- split: 前半384 selection / 後半384 held-out
- control: 基準gate + positive-axis半径4の8方向
- pool: control 8方向 + balanced signed Hadamard 14方向、計22方向
- 配備候補数: 基準を含む9候補
- candidate/reference: Threads=1、各1,000,000 nodes
- utility: 全候補手を同じ固定`nn.bin`の制限手探索で共通再採点
- 選定: `C(22,8)=319,770`集合を全列挙。control以上の有効教師率とgainを制約に多様度最大化
- bootstrap: held-out root単位20,000回、固定seed

収集は768 root、各23候補で完了し、source rootは1,200--1,967の連番だった。収集時間は
6,280.71秒で、生poolの候補手種類数は平均2.4388だった。

## 選択集合

選ばれた8方向はaxis 4本とjoint 4本だった。

- axis: `axis_0, axis_2, axis_3, axis_6`
- joint: `hadamard_2_pos, hadamard_3_pos, hadamard_3_neg, hadamard_7_pos`

| selection 384 root | 有効教師率 | 平均gain (cp) | 候補手種類数 |
|---|---:|---:|---:|
| 固定axis | 16.927% | 389.721 | 2.0391 |
| 選択joint | **17.448%** | **391.078** | **2.0859** |

utility制約を保ちながらselection多様度を増やす最適化自体は成功した。

## Held-out

| held-out 384 root | 有効教師率 | 平均gain (cp) | 候補手種類数 |
|---|---:|---:|---:|
| 固定axis | 17.969% | 285.044 | **2.1302** |
| 選択joint | 17.969% | 247.419 | 2.1250 |
| 差 | 0.000pt | **-37.625** | **-0.00521** |

- 有効教師率差 bootstrap 95%区間: `[-0.01823,+0.01823]`
- 多様度差 bootstrap 95%区間: `[-0.05990,+0.04688]`

gain差が負であり、事前の棄却条件を満たした。3点推定がすべて非負ではないため、24時間計画の
fresh root追試分岐には進まない。

cp平均は詰み値を含むため絶対値が大きい。判定は同じroot・同じ共通再採点に対する対応差で行った。

## 成果物

- 候補定義: `configs/joint_candidate_biases_20260827.json`
- 収集: `scripts/collect_search_utility_teachers.py`、`scripts/run_joint_candidate_directions.sh`
- 共通再採点: `scripts/rescore_disagreement_pool.py`
- 全列挙・held-out解析: `scripts/analyze_joint_candidate_directions.py`
- 集計: `results/joint_candidate_directions_20260827.json`
- raw details/cache: `tmp/joint_candidate_directions_20260827/`
- logs: `/tmp/joint_candidate_directions_{collect,rescore,analysis}.log`
