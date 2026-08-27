# 浅いroot探索統計 棋力検証計画 (2026-08-27)

対象は`H-ROOT-SEARCH-STATS-STRENGTH`である。静的combined 7特徴対照と、同じ7特徴へ固定
1024-node MultiPV統計6特徴を加えた候補を、固定node自己対局で比較する。

## 配備条件

- candidate: `root_search_stats_from_m0/checkpoints/0.ckpt`の13次元model
- control: `root_features_combined_from_m0/checkpoints/0.ckpt`の7次元model
- candidateは各着手前に固定HalfKP、Threads=1、MultiPV=4、1024 nodesを探索し、6特徴を設定する
- 本探索: 両者Threads=1、100,000 nodes
- 評価値が着手ごとに変わる候補のTT混入を避けるため、両者とも各着手前にhashをclearする
- 開始局面: `accuracy_eval_10k/validation.jsonl`のabs(eval)<=500をseed 20260826でshuffleした
  2,200--4,199番。過去decision-aligned matchの200--2,199番と重ならない2,000局面
- 各開始局面を先後反転し、4,000局。50 opening/100局単位、最大4並列

## 判定

候補のElo 95%区間下端が0より上なら`支持`、上端が0以下なら`棄却`、それ以外は`測定完了`とする。
点推定だけで判定を変更しない。オンライン浅探索時間も合計して記録する。

24時間拡張では、4,000局の点推定が正で区間が0を跨ぐ場合に限り、未使用開始局面を追加して
最大8,000局まで延長する。負方向ならこの候補へ追加局数を使わない。
