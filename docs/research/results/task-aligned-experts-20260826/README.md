# Root予測可能なexpert専門化 結果 (2026-08-26)

## 結論

`H-TASK-ALIGNED-EXPERTS`は事前規則により**棄却**した。game ply 32手幅の8役割を割り当てた
expert補助lossは、独立valAで担当phase expert loss、8 expert平均loss、blended lossをすべて改善した。
しかし独立64 root・100万nodesのsearch utilityでは、有効教師率とoracle gainは増えた一方、候補手
多様度が`1.828`から`1.766`へ低下したため、支持条件の同時改善を満たさなかった。

否定した範囲は固定32手幅のphase役割とpositive-axis候補である。rootから予測可能な役割による静的な
専門化そのものは成立したが、それだけでは探索時に異なる候補手を作るrouting価値へ転換しなかった。

## 条件

- 計画: [Root予測可能なexpert専門化](../../experiments/task-aligned-experts-20260826.md)
- role `k`: game ply `[32k+1, 32(k+1)]`、role 7は225手以降を含む
- initial: M0
- train: `tmp/proxy_gap_v2/train` 99,840 root x 8 leaf、1 epoch
- validation: valB/valA各19,968 root
- adapter/backbone固定、expertsのみ学習、通常mean group loss weight 1
- role expert補助weight: `0.02, 0.05`、matched controlは0

## Blended loss

差は同条件controlから候補を引いた対応root差であり、負が改善を表す。

| validation | role weight | group loss差 | bootstrap 95%区間 |
|---|---:|---:|---:|
| valB | 0.02 | -0.00000345 | [-0.00000496, -0.00000191] |
| valB | 0.05 | **-0.00001030** | [-0.00001357, -0.00000705] |
| valA | 0.02 | -0.00000618 | [-0.00000758, -0.00000485] |
| valA | 0.05 | **-0.00001440** | [-0.00001742, -0.00001156] |

両候補ともvalBの許容上限`+0.0002`内で、valAでも改善が再現した。以降は改善が大きいweight 0.05を
search utility候補とした。

## 単体expert専門化

valA 19,968 rootで、rootのroleに対応するexpert単体と全expert平均を比較した。

| 条件 | 担当phase expert loss | 8 expert平均loss |
|---|---:|---:|
| M0 | 0.031751 | 0.042567 |
| matched control | 0.031189 | 0.042027 |
| role weight 0.02 | 0.029756 | 0.041479 |
| role weight 0.05 | **0.028355** | **0.040870** |

weight 0.05の担当phase loss差はcontrol比`-0.002834`、95%区間
`[-0.002897, -0.002775]`だった。8 expert平均lossもM0比`-0.001696`、95%区間
`[-0.001748, -0.001648]`で、許容上限`+0.002`を通過した。役割expertだけを改善して他expertを
壊す挙動ではない。

## 100万nodes search utility

未使用`tmp/proxy_gap_v2/valA`先頭64 rootで、M0とrole005の各々について基準gateとexpert軸`+1`の
計9候補を探索した。candidate/referenceはともに1,000,000 nodesである。

| 条件 | 有効教師率 | 正gain率 | oracle gain平均 | 候補手多様度 |
|---|---:|---:|---:|---:|
| M0 | 9/64 (14.06%) | 14.06% | 12.83cp | **1.828** |
| role weight 0.05 | **10/64 (15.63%)** | **17.19%** | **474.58cp** | 1.766 |
| 差 | +1.56pt | +3.13pt | +461.75cp | **-0.0625** |

平均gainの大幅増には少数の大きなgainが寄与するが、事前判定は有効教師率と多様度の双方を要求する。
多様度が逆方向だったため固定bestmoveへ進めず棄却した。

## 成果物

- role cache: `scripts/build_phase_role_cache.py`
- 学習: `scripts/run_task_aligned_experts_stage1.sh`
- valA診断: `scripts/run_task_aligned_experts_stage2.sh`
- utility: `scripts/run_task_aligned_experts_utility.sh`
- 集計: `results/task_aligned_experts_{valB,valA,specialization_valA,utility}.json`
- logs: `/tmp/train_nnue_task_aligned_experts_*.log`、`/tmp/task_aligned_experts_*.log`
