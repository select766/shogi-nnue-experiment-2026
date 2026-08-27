# 浅いroot探索統計の棋力追試 結果 (2026-08-28)

## 結論

`H-ROOT-SEARCH-STATS-STRENGTH`は事前規則により**測定完了**とした。固定1024-node MultiPV統計を
オンライン入力する13次元モデルは、静的combined 7特徴対照との固定10万nodes・4,000局で
1,921勝1,942敗137分、`-1.824 Elo`、95%区間`[-12.405,+8.757]`だった。区間が0を跨ぐため棄却条件
にも支持条件にも達しないが、点推定は負であり、小さなvalidation bestmove改善の棋力転換は確認できない。

固定bestmoveはvalidationで`+0.09`ポイントだった一方、独立testでは`-0.01`ポイントだった。
自己対局も負方向だったことから、浅い統計による微小loss改善をこの条件で採用する根拠は強まらなかった。

## 条件

- 計画: [浅いroot探索統計 棋力検証](../../experiments/root-search-stats-strength-20260827.md)
- candidate: 静的7特徴 + 1024-node MultiPV統計6特徴の13次元モデル
- control: matchedな静的combined 7特徴モデル
- 本探索: 両者`Threads=1`、100,000 nodes
- 浅探索: 固定HalfKP、`Threads=1`、`MultiPV=4`、1,024 nodes
- TT: 評価値が着手ごとに変わる候補に合わせ、両者とも各着手前にhash clear
- 開始局面: validationのabs(eval)<=500をseed 20260826でshuffleした2,200--4,199番
- 2,000開始局面を先後反転、4,000局、50 opening/100局chunk、最大4並列

開始局面は過去decision-aligned matchのshuffle index 200--2,199と重ならない。全4,000局について
各SFENにcandidate先手・後手が1局ずつあることをmerge時に検査した。

## 対局結果

| 局数 | 勝 | 敗 | 分 | score rate | Elo | 95%区間 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 1,921 | 1,942 | 137 | 49.7375% | -1.824 | [-12.405, +8.757] |

途中の1,200局では`-12.17 Elo`、2,000局では`+3.65 Elo`、3,600局では`+0.48 Elo`と符号が変わり、
小効果を途中値で判定しない必要性も確認した。

## 追加計算量

candidateの着手184,195回に対し、浅探索合計240.91秒、本探索合計45,595.06秒で、浅探索時間は
本探索の`0.528%`だった。固定bestmoveの100万nodes評価では`0.127%`だったため、main nodesを
10万へ下げた今回の相対負荷は増えたが、依然1%未満だった。これは追加計算を許した情報価値の比較であり、
同一wall-clock比較ではない。

## 24時間分岐

点推定が負なので、事前計画どおりroot統計候補の追加4,000局は実行しない。既存の未確定仮説中、
validation/test bestmoveと4,000局Eloがすべて正だった`H-DECISION-ALIGNED-LOSS`を、test由来の
未使用開始局面8,000局で合計12,000局へ拡大する。

## 成果物

- 対局実装: `src/train_nnue/run_match.py`
- chunk merge: `scripts/merge_match_results.py`
- runner: `scripts/run_root_search_stats_strength_match.sh`
- aggregate: `results/match_root_search_stats_4000.json`
- logs: `/tmp/eval_match_match_root_search_stats_chunk_*.log`
