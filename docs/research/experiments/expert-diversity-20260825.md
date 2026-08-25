# Expert多様性ボトルネック 実験計画 (2026-08-25)

対象は`H-EXPERT-DIVERSITY`である。現行M0のadapterを固定し、expertパラメータの平均を保ったまま
expert間偏差だけを拡大する制御介入により、相関低下がrouting/search utilityを増やすか検証する。

## 介入

各expertパラメータテンソル`W_k`について、expert平均`W_bar`から
`W'_k = W_bar + alpha * (W_k - W_bar)`とする。`alpha={1.5, 2.0}`を比較し、adapter、DNN backbone、
gate分布は変更しない。各テンソルのexpert平均がfloat誤差内で不変であることを変換時に検査する。

これは強いexpertを新たに学習する実験ではなく、現expert群の機能差を増やす局所的な因果介入である。
相関を下げてもutilityが増えなければ、「現在の差の方向を広げるだけではrouting効果は増えない」と判定する。

## 段階判定

1. 現行paired validation 10,000 positionsでgate診断とexpert出力相関を測り、別途
   `tmp/proxy_gap_v2/valA`の先頭10,000 rootで実探索leafのgroup lossを対応比較する。
2. expert間相関平均が0.90以下、かつM0に対する実探索leaf group loss悪化が0.002以下の最大alphaを
   候補とする。
3. 候補がある場合、held-out 64 root、候補探索・独立utilityとも1,000,000 nodesで、M0と同じ
   logit `+1.0`候補群の有効教師率、candidate bestmove多様度、oracle gain p90を比較する。
4. 有効教師率またはoracle gain p90が相対25%以上改善し、candidate bestmove多様度も10%以上改善すれば
   `支持`とする。相関を0.90以下へ下げてもこれらが改善しなければ`棄却`する。相関を下げられない、または
   loss条件を満たす候補がなければ`不確定`とする。

配備時の不変条件は維持する。DNNはrootで候補ごとに1回だけ実行し、blendは探索終了まで固定する。
