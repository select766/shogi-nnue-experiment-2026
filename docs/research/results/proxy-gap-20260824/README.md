# Proxy gap検証結果 (2026-08-24)

## 結論

proxy gapは確認された。棋譜先局面で学習したrouterと、固定探索器が実際に訪れるqsearch leafで学習した
routerは、同じroot入力から異なる固定blendを学ぶ。教師score生成を固定`nn.bin`のexact qsearchへ統一しても、
実探索leaf学習モデルは棋譜位置学習モデルより実探索validationでA/Bとも約`0.0017`良かった。

逆方向では、実探索leaf学習モデルが棋譜位置validationで約`0.00016`悪化した。したがって一方が全面的に
優れたのではなく、学習leaf分布への特化である。実行時のDNN 1回、rootで決めたblendを探索中固定という
制約はデータ生成・学習・評価の全てで維持した。

## データ

- train: 500,000 root x 8 leaf。元503,838 rootを走査し、8 leafを得られない3,838 rootを除外
- validation: 別棋譜系列20,000 root x 16 leafをA/B各8へ分割
- 探索: 現行YaneuraOu、`Threads=1`、`nodes=1000`、定跡off、`USI_Hash=16`
- 抽出: 通常探索が`depth <= 0`でqsearchへ入る訪問列からAlgorithm R reservoir sampling
- 平均qsearch境界訪問数: 351.58953/root
- 教師: 候補モデルと独立した`bin/eval/nn.bin`によるexact qsearch
- PackedSfenにない`gamePly`はPackedSfenValueのfieldからSFEN末尾へ復元

実探索leafのglobal unique SFEN率はA 99.12%、B 99.09%だった。棋譜proxyはoffsetの復元抽出重複により
A 72.55%、B 72.52%だった。同一root内で実探索leafと棋譜proxy leafが1つ以上一致する割合はA 2.41%、
B 1.99%に留まった。

## 学習条件と選抜

全条件をcheckpoint 510から開始し、expertsとbackboneを固定してadapterだけ学習した。batch 256、
500,000 root/epoch、adapter LR 0.01、momentum 0.9、task-only、`lambda=1`である。

実探索leaf条件は5巡実行し、選抜用B lossは次のように推移した。

| checkpoint | 累積root | B loss |
|---:|---:|---:|
| 0 | 500,000 | 0.0198871903 |
| 1 | 1,000,000 | 0.0199967846 |
| 2 | 1,500,000 | 0.0199430417 |
| 3 | 2,000,000 | 0.0198968463 |
| 4 | 2,500,000 | **0.0198650286** |

最良checkpoint 4をBで選び、Aへ一度だけ確認した。棋譜proxy対照は、元scoreを保つ条件と、全leafを
固定`nn.bin`で再labelする条件の両方を同じ500,000 rootで実行した。いずれもcheckpoint 0が最良で、
その後3回連続で`2e-5`以上改善せず、事前規則により飽和として停止した。

## 実探索validationでの主結果

loss差は候補 minus checkpoint 510であり、負が改善である。

| 学習データ | A loss差 (95% root bootstrap) | A改善root | B loss差 (95% root bootstrap) | B改善root |
|---|---:|---:|---:|---:|
| 実探索leaf 500k、ckpt 4 | **-0.004936** `[-0.005319,-0.004596]` | 74.29% | **-0.005300** `[-0.005695,-0.004959]` | 71.85% |
| 棋譜位置500k、固定qsearch再label、ckpt 0 | -0.003260 `[-0.003512,-0.003026]` | 71.17% | -0.003590 `[-0.003837,-0.003357]` | 69.52% |
| 棋譜proxy 500k、従来score、ckpt 0 | +0.000594 `[+0.000500,+0.000689]` | 45.86% | +0.000522 `[+0.000428,+0.000615]` | 46.11% |
| 棋譜proxy 5m、従来score、ckpt 4 | +0.001156 `[+0.001027,+0.001290]` | 43.10% | +0.001067 `[+0.000938,+0.001196]` | 43.66% |

固定qsearch再label棋譜位置モデルを直接controlとすると、実探索leafモデルの追加改善はAで
`-0.001676`、95%区間`[-0.001877,-0.001496]`、Bで`-0.001710`、
`[-0.001926,-0.001515]`だった。教師score生成法を揃えた後にも残るため、これは局面分布のgapである。

## 逆方向の確認

固定qsearch再labelした棋譜位置validationでは、実探索leafモデルは棋譜位置モデルよりAで
`+0.000154`、95%区間`[+0.000084,+0.000233]`、Bで`+0.000166`、
`[+0.000093,+0.000247]`悪かった。両モデルの差は評価分布への特化として整合する。

## 判断と次の実験

棋譜先局面proxyを増量するだけでは実探索末端向けrouterを最適化できない。今後のrouter学習データは
実探索qsearch訪問分布を優先する。ただし今回の探索器はbaseline `nn.bin`固定、1000 nodesであり、
探索量やblendにより訪問分布が変わる可能性が残る。次の仮説は、nodesを変えた時の分布転移と、学習した
blend自身で再収集するDAgger型反復である。最善手一致率・対局強さは本実験の対象外なので別途評価する。

## 成果物

- 実験計画: `docs/research/experiments/proxy-gap-20260824.md`
- collector: `scripts/collect_search_leaf_groups.py`
- 固定qsearch再label: `scripts/relabel_root_grouped_qsearch.py`
- 対応root抽出: `scripts/subset_root_grouped_by_roots.py`
- 学習曲線: `results/proxy_gap_*_training_curve.json`
- paired比較: `results/proxy_gap_final_matrix_actual_val{A,B}.json`
- 直接比較: `results/proxy_gap_direct_actual_vs_relabel_proxy_val{A,B}.json`
