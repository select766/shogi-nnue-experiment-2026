# Search utility直接蒸留 pilot 結果 (2026-08-25)

## 結論

rootで複数の固定blend候補を実探索し、その指し手を独立評価してrouterへ蒸留する機構を実装した。
候補集合には十分な指し手差と独立評価上のoracle gainがあったが、1,600 rootの蒸留でheld-out教師への
cross entropyが僅かに改善しても、固定10,000局面の100万nodes bestmove一致は改善しなかった。
事前規則に従いscale、固定test、自己対局には進めない。

## 配備前提と教師定義

現行M0のgateを基準とし、root logitsのexpert 0--7へ各々+1.0した8候補を加えた。各候補について
DNNはrootで1回だけ実行し、そのblendを探索終了まで固定した。leafごとのDNN推論は行っていない。

候補探索はThreads=1、10,000 nodesとした。各候補bestmoveを、候補から独立した固定`nn.bin`評価器で
1手ずつ100,000 nodesの制限手探索にかけ、root手番視点score cpをsearch utilityと定義した。各制限手
探索前にTTを初期化した。基準候補より最大utilityが10cp以上高い場合だけ最大候補のgateを教師とし、
それ未満では基準gateを保持した。

ここで教師cross entropyとは、この8次元教師gate分布に対する学習router gateのcross entropyである。
bestmove一致率とは別物で、後者は固定HCPE局面に保存された教師bestmoveと、100万nodesのやねうら王
bestmoveが同一である割合を指す。

実装は`ExpertBlendingGateLogitBias` USI optionで同一ONNX routerのlogit biasだけを切り替える。
全zeroまたはoption未指定時はgateの再正規化も行わず、従来production経路を保つ。教師生成は
`scripts/collect_search_utility_teachers.py`、学習は
`scripts/run_search_utility_distillation_pilot.sh`、固定評価と対応比較は
`scripts/eval_search_utility_accuracy.sh`と`scripts/compare_search_utility_accuracy.sh`で再現できる。

## データと候補診断

既存root-grouped母集団の先頭2,000 rootを使い、先頭1,600をtrain、後続400をvalidationとした。
各rootのroot record、leaf group、teacher cacheは同じ範囲でbyte一致を確認した。

| 指標 | train 1,600 | validation 400 |
|---|---:|---:|
| teacher changed率 (gain 10cp以上) | 28.31% | 26.25% |
| unique candidate bestmove平均 | 2.468 | 2.395 |
| unique candidate bestmove中央値 | 2 | 2 |
| 基準gateから教師への平均L1変化 | 0.04985 | 0.04721 |

validationのoracle gain中央値は0cp、90 percentileは130.2cpだった。mate scoreを含むためgain平均値は
外れ値に支配され、継続判断には使わなかった。C++が出した候補gateと
`softmax(log(base gate) + one-hot)`の最大誤差はsmoke rootで`2.7e-8`だった。

## 学習

M0 (`logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt`)からexpertsを固定し、adapterだけを
10 epoch学習した。router-onlyはtask weight 0 / router weight 1、combinedはtask weight 1 /
router weight 0.01とした。batch 64、adapter LR 0.003である。

validation 384 rootで全epochを比較し、cross entropy最小epochを事前規則どおり選んだ。

| model | 選択epoch | 教師cross entropy | leaf group loss |
|---|---:|---:|---:|
| M0 | - | 1.57719576 | 0.01247725 |
| router-only | 8 | 1.57712746 | 0.01247688 |
| combined | 7 | 1.57716691 | 0.01247417 |

教師への改善量はrouter-onlyで−0.00006831、combinedで−0.00002885と非常に小さい。元gateと教師の
top1一致が96.61%で、教師変更後もsoft gateの移動量が小さいため、蒸留信号自体が弱かった。

## 固定bestmove validation

固定validation 10,000局面、Threads=1、1,000,000 nodesでM0と対応比較した。

| model | 一致数 | 一致率 | M0との差 | M0のみ正解 | 候補のみ正解 | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 6,191 | 61.91% | - | - | - | - |
| router-only epoch 8 | 6,176 | 61.76% | −0.15 point | 849 | 834 | 0.733 |
| combined epoch 7 | 6,141 | 61.41% | −0.50 point | 886 | 836 | 0.238 |

どちらも正方向でなく、有意差もない。したがって「この候補半径・10k候補探索・100k独立utility・
1,600 root」というpilot条件では、search utility直接蒸留が棋力proxyへ転換する仮説を支持しない。

## 解釈と残る範囲

今回否定されたのはsearch utilityという考え全体ではなく、局所logit +1候補を10k-nodeで選び、soft
gateを保守的に蒸留する具体条件である。候補探索10kと配備評価100万nodesのhorizon差、教師gateの
平均L1変化が約0.05しかない弱い介入、1,600 rootという標本量は残る制約である。ただし、事前に定めた
bestmove継続条件を満たさないため、同じ教師の単純なデータscaleは行わない。
