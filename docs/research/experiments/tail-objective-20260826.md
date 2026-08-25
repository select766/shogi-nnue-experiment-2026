# Tail-sensitive group objective 実験計画 (2026-08-26)

対象は`H-TAIL-OBJECTIVE`である。M0からadapterだけを追加10万root学習し、平均leaf KLを使う対照と、
root内loss上位2/8 leafを使うCVaR目的を比較する。

## 条件

- 初期値: `logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt` (M0)
- train: `tmp/proxy_gap_v2/train`先頭99,840 root、group size 8
- 選抜validation: `tmp/proxy_gap_v2/valB` 19,968 root
- 未使用判定: `tmp/proxy_gap_v2/valA` 19,968 root
- experts/backbone固定、adapter-only、LR 0.01、momentum 0.9、seed 42
- 条件: `mean`、`cvar` (top 25%=2 leaf)、`mixed` (mean 0.5 + CVaR 0.5)

位置lossは教師Bernoulli分布とモデル予測のKLで、従来の`result - teacher_entropy`と同じである。
CVaRは各rootの8 leafのうち位置lossが大きい2件を選び、その平均をroot間で平均する。学習時だけでなく
比較時も同じtop-2定義を使う。

## 事前判定

追加mean学習をmatched controlとする。候補が次をすべて満たした場合だけ固定bestmoveへ進める。

1. 未使用valAのroot別top-2 loss差についてbootstrap 95%区間上端が0未満。
2. valA平均group lossのmean対照からの悪化が0.0005以下。
3. valBとvalAでtail差が同方向。

通過候補は固定validation 10,000局面、`Threads=1`、100万nodesでmean対照と対応比較する。
bestmove一致率差が正方向なら`支持`、0以下なら`棄却`とする。loss条件を通る候補がなければ、
tail目的が平均目的より良い候補を作れなかったとして`棄却`する。
