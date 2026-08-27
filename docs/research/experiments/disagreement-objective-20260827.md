# 候補手disagreement多目的選択 実験計画 (2026-08-27)

対象は`H-SU-DIVERSITY-OBJECTIVE`である。score gainだけで選んだroot別半径が候補手多様度を
低下させたため、候補bestmove coverageを直接目的に入れた9候補集合を固定半径4と比較する。

## データと候補pool

- root: `tmp/proxy_gap_v2/valA/roots.bin`の1,000--1,191番、計192件。過去のsearch utility収集で
  使った先頭64 root、および`tmp/search_utility_v1`のrootとは重ならない。
- selection/held-out: 前半96件で候補集合を1回選び、後半96件は最終判定にだけ使う。
- pool: positive-axisのlogit半径`0.5, 1, 2, 4`、8 expert方向の32候補と基準gate。
- 配備候補集合: 各expert方向につき半径を1つ選ぶため、基準を含め常に9候補。
- candidate/reference: `Threads=1`、各1,000,000 nodes。独立固定`nn.bin`の制限手scoreをutilityにする。
- 固定対照: 全8方向を半径4とする。直前実験で確定済みなので今回のselectionでは選び直さない。

## disagreement目的

全`4^8=65,536`半径ベクトルをselectionで列挙する。固定対照以上の有効教師率（10cp以上）と
平均oracle gainを持つ集合だけをeligibleとし、その中でrootあたり候補bestmove種類数の平均を
最大にする。同値は有効教師率、平均gain、半径ベクトルの辞書順で決める。eligibleがなければ
固定半径4を選ぶ。この規則にheld-outの値は使わない。

## 事前判定

held-out 96 rootで選択集合minus固定半径4を対応比較する。

1. 有効教師率差と候補手多様度差がともに正、両bootstrap 95%区間下端が0以上、かつ平均oracle
   gain差が0以上なら`支持`。
2. 有効教師率差または多様度差の95%区間上端が0以下、あるいは平均gain差が負なら`棄却`。
3. それ以外は`測定完了`。同じheld-outで目的重みや制約を再調整しない。
