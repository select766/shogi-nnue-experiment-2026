# Root-grouped router仮説の検証計画 (2026-08-23)

## 実行時制約と仮説

実運用では探索ルート局面 `R` に対してDNNを一度だけ実行し、8 expertsの合成重み `w(R)`を
決める。探索中はこの重みを固定し、すべての末端局面を同じNNUE関数で評価する。末端ごとにDNNを
実行したり、末端局面をrouterへ入力したりする方式は対象外とする。

検証する仮説は次である。

> ルート局面だけから、そこから到達する複数の末端局面に平均的に有効な、固定expert blendを
> 予測できる。

ルート `R` に対応する末端集合を `L(R)={L_1,...,L_K}` とし、共有teacherを次で定義する。

`w*(R) = argmin_w mean_k loss(F_w(L_k)) + lambda_KL KL(w || w_current(R))`

`F_w`は重み`w`で一度合成されたNNUE評価関数である。従来の末端ごとの
`argmin_w loss(F_w(L_k))`は、探索時に実現できないoracleなので、この仮説のteacherには使わない。

## Pilotデータ

第一段階は実探索木ではなく、ラベルを保持している棋譜上の将来局面を探索末端の代理にする。

- ルートごとに `H=min(50, 棋譜の残り手数)` とする。
- `1..H`から復元抽出で`K=8`個のoffsetを一様サンプリングする。
- 各将来局面へ現行と同じqsearchを適用する。
- 8末端は同じルート重みを共有する。
- trainとvalidationは入力ファイルを分け、さらにPacked SFENで照合してtrainと同一のルートを
  validationから除く。validation内の重複ルートも除き、ルート単位統計の独立性を改善する。
- validationは同一ルートについて独立な集合A、Bを各8末端作る。実装上は1ルート16末端を生成し、
  前半8件をA、後半8件をBとする。

実験形式は次とする。

- `roots.bin`: ルートごとに40 bytes、shape相当は`(R,)`。
- `leaves.bin`: group-majorに40 bytes、shape相当は`(R,K)`。
- `offsets.npy`: qsearch前の棋譜上offset、shapeは`(R,K)`。
- `metadata.json`: format version、K、seed、qsearch、入力、件数を保存する。

pilot規模はtrain 100,000 roots x 8 leaves、validation 10,000 roots x 16 leavesとする。

## 指標の正確な定義

本文と結果表では単独の「一致率」という語を使わず、次の完全名を使う。

### Shared-teacher argmax agreement (A/B)

同じvalidationルートについて、集合Aから最適化した共有teacherを`w_A`、集合Bから最適化した
共有teacherを`w_B`とする。`argmax(w_A) == argmax(w_B)`となるルートの割合を指す。
これは「独立な末端標本から同じ最大重みexpertが選ばれる割合」であり、指し手一致率ではない。

### Router-to-shared-teacher top-1 match

ルート局面だけを入力したrouter出力を`p(R)`、共有teacherを`w*(R)`とする。
`argmax(p(R)) == argmax(w*(R))`となるルートの割合を指す。expert番号の一致であり、
やねうら王の指し手や単一expert oracleとの一致ではない。

### Router-to-single-expert oracle match

ルートの末端集合に対し、各expertを単独で使った平均lossが最小のexpert番号と、routerの
`argmax` expert番号が一致するルートの割合を指す。blend teacherとの一致とは別に記録する。

### Best-move agreement

やねうら王が固定ノード探索で返した最善手と、評価JSONLのHCPE由来教師指し手が一致した局面の
割合だけを指す。本pilotのteacher安定性・validation loss評価では測定しない。

### Cross-set loss delta

集合Aで得たteacherを独立集合Bへ適用した平均lossと、現行router重みを同じ集合Bへ適用した
平均lossとの差をルートごとに計算する。

`delta_B = mean_loss_B(w_A) - mean_loss_B(w_current)`

負値が未知末端への改善を表す。平均だけでなく、ルート単位bootstrap 95%区間を保存する。

### Teacher distribution divergence

`w_A`と`w_B`のJensen-Shannon divergenceをルートごとに計算し、その平均と分位点を保存する。
0は同一分布を表す。

## 段階判定

1. **共有teacherの存在**: `delta_B`の平均が負で、ルートbootstrap 95%区間の上端も0未満なら、
   Aで決めた固定重みが未知末端Bへ転移すると判断する。
2. **ルートからの予測可能性**: expertsを固定してrouterを学習し、独立末端B上のgroup平均task
   lossが同一初期値・同一データ量のtask-only対照より低い場合に候補とする。
3. 上記を通った候補だけ、別validation窓、最善手一致率、固定ノード自己対局へ進める。

棋譜上の将来局面は実探索末端の代理にすぎない。第一段階を通った場合、固定した探索器から
qsearch末端をオフライン収集する第二段階で追試する。探索中にDNNを追加実行する設計にはしない。
