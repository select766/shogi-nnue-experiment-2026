# Root別search utility候補半径 実験計画 (2026-08-26)

対象は`H-SU-ADAPTIVE-RADIUS`である。全rootへ同じ半径を与えた直前実験を対照に、rootで一度だけ
得られる低コスト特徴からlogit半径を選ぶ場合に、候補数9を保ったままsearch utility信号が増えるか測る。

## データと候補

- root: `tmp/search_utility_v1/val/roots.bin`の192--383番、計192件。直前までの実験で使った
  0--191番とは重ならない。
- selection/held-out: 前半96件で固定半径と予測器を選び、後半96件は最終判定に一度だけ使う。
- 半径: positive-axisの`0.5, 1.0, 2.0, 4.0`。各条件は基準gateと8 expert方向の計9候補。
- candidate/reference: `Threads=1`、各1,000,000 nodes。独立固定`nn.bin`の制限手探索scoreをutilityにする。

各rootでは半径を一つだけ選ぶため、探索候補数は固定半径と同じ9である。複数半径を同時に配備する
oracleは上限診断にだけ用いる。

## 予測器

特徴はgame ply、手番、盤上駒数、持駒数、成駒数、合法手数、現gateのentropy、最大weight、top1-top2
marginに限定する。探索後のscore、候補bestmove、held-outのutilityは入力しない。selection内4-foldで
深さ1--3の決定木を選び、その後selection全体へfitする。各rootの教師半径はoracle gain最大、同値なら
小さい半径とする。固定対照はselectionで有効教師率、候補手多様度、gain平均の順に最大の半径とする。

## 事前判定

held-out rootで予測半径から得た値と固定対照を対応比較する。

1. 有効教師率差とcandidate bestmove多様度差がともに正で、両方のroot bootstrap 95%区間下端が
   0以上なら`支持`。
2. いずれかの差の95%区間上端が0以下なら`棄却`。
3. それ以外は予測可能性が未確定なので`測定完了`とし、同じ標本で閾値や特徴を追加調整しない。

oracle半径でも固定対照を両指標で上回らない場合は、予測器によらず`棄却`する。
