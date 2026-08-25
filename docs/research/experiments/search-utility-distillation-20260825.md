# Search utility直接蒸留 実験計画 (2026-08-25)

## 仮説

leaf score lossではなく、rootで固定blend候補を実際に探索させて得た指し手のutilityを教師にすると、
routerのbestmove指標へ転換する。DNNは各候補のrootで1回だけ実行し、その候補のblendは探索終了まで
固定する。leafごとにDNNを実行する方式は使用しない。

## 候補とutilityの正確な定義

基準候補は現行M0 (`logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt`) がrootから出す
8次元gateである。追加候補は同じroot logitsのexpert 0から7の各1成分へ順に+1.0を加え、softmaxを
取り直した8個である。したがって計9候補となり、各候補はroot依存性を保ったsoft blendである。

候補探索はThreads=1、同一node数で行う。候補が返したunique bestmoveだけを、候補から独立した固定
`nn.bin`評価器で1手ずつ同一node数の制限手探索へ入力する。各探索前にTTを初期化する。ある候補の
search utilityとは、その候補のbestmoveに対してこの独立探索が返したroot手番視点のscore cpを指す。
候補自身が返すscoreはutilityに使用しない。

最大utilityと基準候補utilityとの差をoracle gainと呼ぶ。gainが10cp未満なら蒸留対象は基準gateの
ままとする。10cp以上なら、最大utilityのbestmoveを選んだ候補群のgate平均を蒸留対象とする。同一手・
同一utilityの候補を別々の正解として水増ししない。この教師と学習router gateの一致とは8次元分布の
cross entropyを指し、argmax一致だけを主目的にはしない。

## 段階評価

1. 10 root smokeでUSI候補切替、gate正規化、制限手score対応、cache再読込を検証する。
2. 2,000 root pilotをtrain 1,600 / validation 400に分ける。候補探索10,000 nodes、独立教師
   100,000 nodesを初期値とし、teacher changed率、candidate move多様度、oracle gain分布を測る。
3. teacher changedが5%以上あり、validationで教師cross entropyが初期M0より改善する場合だけ、
   adapter-only蒸留を実行する。task lossを併用する条件とrouter-only条件を比較する。
4. 学習候補は固定validation 10,000局面、Threads=1、1,000,000 nodesでM0と対応比較する。

validation bestmoveが正方向でなければscaleしない。正方向ならroot数を拡大し、固定testはscale後の
最終候補だけに使用する。candidate move多様度がほぼ1、またはteacher changedが5%未満なら候補半径を
増やす前に、探索node数を上げても同じかを小標本で確認する。

pilot学習はM0から開始し、experts固定、batch 64、train 1,600 root、validation 400 root、adapter
LR 0.003、10 epochとする。router-onlyはtask weight 0 / router weight 1、combinedはtask weight 1 /
router weight 0.01とする。validation 400 rootのうちbatch境界に揃えた先頭384 rootで教師cross
entropy最小epochを条件ごとに1つ選び、固定bestmove validationへ進める。
