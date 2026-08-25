# Search utility候補幾何 結果 (2026-08-26)

## 結論

`H-SU-GEOMETRY`は事前screening規則により**棄却**した。候補数9、候補探索と独立utility探索を
100万nodesへ揃え、局所`axis-1`に対して`axis-2`と`cyclic-contrast-2`を比較した。axis-2は
oracle gain p90を増やしたが、有効教師率と候補手多様度の事前閾値を満たさなかった。contrastは
3指標とも閾値未満だったため、held-out蒸留へ進める候補はない。

否定した範囲は、全rootへ同じ半径と方向集合を与える候補幾何である。axis-2で一部rootのgainだけは
増えたため、root別に半径を予測・探索する可能性を`H-SU-ADAPTIVE-RADIUS`へ分離する。

## 条件

- 計画: [Search utility候補幾何](../../experiments/search-utility-geometry-20260826.md)
- root: `tmp/search_utility_v1/val/roots.bin`の128--191番、計64件
- model: M0、8 experts、root gateを探索中固定
- candidate/reference: `Threads=1`、各1,000,000 nodes
- utility: 独立固定`nn.bin`による制限手探索のroot手番score
- 各集合: 基準gateを含む9候補

## 結果

| 幾何 | 有効教師率 | oracle gain平均 | oracle gain p90 | 候補手多様度 | 教師L1変化 |
|---|---:|---:|---:|---:|---:|
| axis-1 | 10/64 (15.63%) | 16.09cp | 35.5cp | 1.656 | 0.0299 |
| axis-2 | 12/64 (18.75%) | **21.95cp** | **48.4cp** | **1.719** | 0.0811 |
| cyclic-contrast-2 | 9/64 (14.06%) | 19.89cp | 44.0cp | 1.672 | 0.0733 |

事前通過値は有効教師率19.53125%以上、p90 44.375cp以上、多様度1.821875以上である。axis-2は
p90だけを通過し、有効教師率は0.78125ポイント、多様度は0.103125不足した。contrastはp90も
0.375cp不足した。同じteacherの単純scaleは行わないという規則を維持し、蒸留・固定bestmoveへ進めない。

axis-2では教師gateのL1変化が大きくなったが、候補手の種類は比例して増えなかった。現expertの局所方向では、
gate距離を全root一律に広げると「既に差が出るrootのgain」を増幅する一方、「差が出ないrootを新たに
informativeにする」効果が弱い。

## 成果物

- collector geometry: `scripts/collect_search_utility_teachers.py`
- 集計: `results/search_utility_geometry_{axis2,contrast2}.json`
- raw: `tmp/search_utility_geometry_{axis2,contrast2}/details.jsonl`
- logs: `/tmp/search_utility_geometry_{axis2,contrast2}.log`
