# 浅いroot探索統計 結果 (2026-08-27)

## 結論

`H-ROOT-SEARCH-STATS`は**支持**した。静的combined 7特徴へ固定1024-node MultiPV統計6特徴を追加すると、
選定用valBと未使用valAのroot-group lossがともに改善し、95% bootstrap区間は全域負だった。
固定validation 10,000局面・100万nodesのbestmove一致率も`61.95%`から`62.04%`へ`+0.09`ポイント
改善した。浅い探索の追加時間は本探索の`0.127%`で、事前上限10%を十分下回った。

bestmove差の正確McNemar検定は`p=0.846`であり、`+0.09`ポイント自体が0から分離したとはいえない。
ここでの支持は、事前規則どおりlossの再現、有向bestmove差、実測時間を同時に満たしたという限定的な判定である。
独立testでは`-0.01`ポイントとなり、bestmove効果量は再現しなかった。

## 条件

- 計画: [浅いroot探索統計](../../experiments/root-search-stats-20260827.md)
- shallow engine: 固定HalfKP、`Threads=1`、`MultiPV=4`、`go nodes 1024`
- 入力: 静的combined 7特徴 + score由来6特徴、計13次元
- initial: M0、追加13列を0初期化
- train: `tmp/proxy_gap_v2/train` 99,840 root x 8 leaf、1 epoch
- selection/confirmation: valB/valA各19,968 root
- matched control: 同じM0・train root・epochの静的combined 7特徴checkpoint
- experts/backbone固定、adapter-only、mean group loss、batch 256、seed 42

探索6特徴はtop1 score、top1-top2 margin、top1-last width、score標準偏差、score entropy、
得られたMultiPV本数率である。train/valA/valB cacheの先頭7列はmatched control cacheと完全一致し、
最大絶対差はすべて`0`だった。

## Root-group loss

差は探索統計ありminus静的7特徴対照で、負が改善を表す。

| split | 対照loss | 候補loss | 対応差 | bootstrap 95%区間 |
|---|---:|---:|---:|---:|
| valB | 0.0198136903 | 0.0198119618 | -0.0000017294 | [-0.0000022063, -0.0000013154] |
| valA | 0.0153250499 | 0.0153233455 | -0.0000017047 | [-0.0000021943, -0.0000012876] |

tail loss差もvalBで`-0.0000055013`、valAで`-0.0000051029`となり、両区間が全域負だった。

## C++配備と固定validation

やねうら王を7/13次元auxiliary入力へ対応させ、評価器が各rootの本探索前に浅いMultiPV探索を行って
後半6特徴をUSI optionで注入する。モデル読込時だけneutral zeroで初期weightを作り、実際の各`go`では
6特徴がない場合をエラーにする。2局面のonline smokeと、従来7次元modelを使う探索の双方を通過した。

| 条件 | 一致数 | 一致率 |
|---|---:|---:|
| 静的combined対照 | 6,195/10,000 | 61.95% |
| 静的 + 浅い探索統計 | 6,204/10,000 | **62.04%** |

対応内訳は対照のみ正解845、候補のみ正解854、差`+0.0009`、正確McNemar `p=0.846116`だった。
浅い探索は合計13.42秒、本探索は合計10,563.40秒で、平均1.342ms対1,056.340ms、追加率は
`0.001271`（`0.127%`）だった。

## 独立test追試

validationが正方向だったため、事前計画どおり固定test 10,000局面を追試した。

| 条件 | 一致数 | 一致率 |
|---|---:|---:|
| 静的combined対照 | 6,209/10,000 | **62.09%** |
| 静的 + 浅い探索統計 | 6,208/10,000 | 62.08% |

対応内訳は対照のみ正解844、候補のみ正解843、差`-0.0001`、正確McNemar `p=1.0`だった。
追加時間は平均1.443ms、本探索平均1,133.106msに対して`0.127%`だった。testは効果量の再現性を
記録する追試なので、事前固定したvalidation条件による支持判定は変更しない。ただし棋力へ進める根拠は
強まらず、loss改善が棋力へ転換するかは別仮説として未検証のままである。

## 成果物

- cache収集: `scripts/collect_root_search_statistics.py`、`scripts/collect_root_search_stats.sh`
- 特徴定義: `src/train_nnue/root_search_statistics.py`
- 学習/loss比較: `scripts/run_root_search_stats_stage{1,2}.sh`
- online accuracy: `src/train_nnue/eval_accuracy_root_stats.py`
- C++推論: `YaneuraOu` submodule commit `09d5ad27`
- loss: `results/root_search_stats_val{B,A}.json`
- accuracy: `results/accuracy_{eval,comparison}_root_search_stats_{validation,test}.json`
- logs: `/tmp/root_search_stats_*.log`、`/tmp/accuracy_eval_root_search_stats_*.log`
