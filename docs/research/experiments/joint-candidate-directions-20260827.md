# Joint signed候補方向 検証計画 (2026-08-27)

対象は`H-SU-JOINT-DIRECTIONS`である。expert別positive-axis半径だけでなく、複数expertを同時に
増減するbalanced signed方向をpoolへ加え、同じ9候補でutilityと候補手多様度を比較する。

## 候補poolとデータ

- root: `tmp/proxy_gap_v2/valA/roots.bin`の1,200--1,967番、fresh 768件
- selection/held-out: 前半384件 / 後半384件
- control: 基準gate + positive-axis半径4の8方向
- pool: controlの8方向と、8次Hadamardの非定数7行および各反転、計22方向
- joint方向: 4要素を`+1`、4要素を`-1`するlogit bias。softmaxの共通移動は含めない
- 配備集合: poolから8方向を選び、基準を含め常に9候補
- candidate/reference: Threads=1、各1,000,000 nodes

全候補手を共通の固定`nn.bin`・制限手100万node探索で再採点する。selectionで全
`C(22,8)=319,770`集合を列挙し、control以上の有効教師率（10cp）と平均oracle gainを持つ集合だけを
eligibleとする。その中で候補bestmove種類数平均を最大化し、同値は有効教師率、gain、候補index辞書順で
固定する。held-outは選定後に一度だけ開く。

## 判定

held-outで選択minus controlの有効教師率差と多様度差がともに正、両bootstrap 95%区間下端が0以上、
かつ平均gain差が0以上なら`支持`する。いずれかの区間上端が0以下、またはgain差が負なら`棄却`、
それ以外は`測定完了`とする。

24時間拡張では、held-outの3点推定がすべて非負だが区間だけが0を跨ぐ場合、選択集合を固定したまま
1,968番以降のfresh rootで独立追試する。明確な負方向ならjoint方向を拡大しない。
