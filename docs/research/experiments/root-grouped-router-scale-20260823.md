# Root-grouped router大規模追試計画 (2026-08-23)

## 目的

pilotの10万rootだけで見えたtask-only router改善が、同一棋譜の反復学習による偶然ではなく、
未知rootと独立に標本化したleaf集合へ転移するかを検証する。実運用上の制約はpilotと同じで、
DNNへ入力するのは探索rootだけであり、そこで決めたblendを全leafへ固定する。

## データ

- train: 元棋譜thread 002、500万root、各root 8 leaf、seed 20260824
- validation: 元棋譜thread 010、10万root、各root 16 leaf、seed 20260825
- validationの16 leafは独立評価用にA/B各8 leafへ分割する
- trainとのPacked SFEN重複4,611 rootとvalidation内重複14 rootを除き、95,375 rootを保持した
- batch境界を揃えた正式評価は先頭95,232 rootを使う
- trainの固有Packed SFENは4,810,544 / 5,000,000、root重複率は3.78912%
- trainのoffset平均は20.4688手、group内の異なるoffset数は平均7.0219 / 8

生成物は`tmp/root_grouped_router_large_v1/`に置き、容量の大きいbinaryはGitへ追加しない。

## 学習条件

- 初期値: checkpoint 510
- expertsとDNN backboneを固定し、adapterだけをtask lossで学習する
- batch 256、adapter LR 0.01、momentum 0.9、最大20 evaluation intervals
- 1 intervalは100万root、5 intervalsで500万root全体を一巡する
- validationはBの95,232 rootで各interval測定する
- validation悪化時はNewBobによりLRを0.5倍にする
- 各epoch checkpointを保存する

train iteratorはinterval境界でresetせず、500万rootの末尾まで進んでから循環する。したがって、
100万root間隔のvalidationは学習範囲を先頭100万rootへ狭めない。

batch 64、LR 0.003でも速度を測定したが、約1,400 root/s、GPU memory 2.1 GiBであり、
batch 256でも約1,480 root/s、4.6 GiBだった。rootごとのDNNと8 leafのNNUE計算が支配的で、
batch拡大によるwall-clock短縮は小さい。正式runには既存fine-tuneと同じbatch 256、LR 0.01を使う。

## 飽和と選抜の定義

TensorBoardの`val_loss`について、直前の意味のある最良値から`2e-5`以上更新したepochを
「意味のある改善」と呼ぶ。最後の意味のある改善から3 epoch連続でこの幅を更新しなければ、
学習曲線上は暫定的に飽和とする。この判定は`scripts/summarize_training_curve.py`でJSON化する。

最終選抜はepoch平均だけで決めない。初期checkpointと全epoch候補を同じrootに適用し、rootごとの
group平均loss差を作る。validation Bで最良候補を選び、学習中に選抜へ使っていないvalidation Aで
平均差とroot bootstrap 95%区間を確認する。Aでも区間上端が0未満なら大規模追試を通過とする。

## 飽和後の仮説

優先順位は次のとおりとする。

1. **proxy gap**: 棋譜上の将来局面へのqsearchは実探索末端と分布が違う。固定した探索器から、
   1 rootにつき複数の実探索qsearch leafを収集し、同じroot-grouped task lossで追試する。
2. **平均lossによるtailの希薄化**: 少数の大損失leafが平均で薄まるなら、group内CVaRまたは
   rootごとの改善上限・局所曲率で重み付けしたtask損失を比較する。
3. **expert多様性不足**: routerを十分学習しても共有teacherとの差が大きく、expert出力相関が
   約0.97のままなら、expert出力のdecorrelationや異なる初期値を先に試し、16 experts化は後にする。

第一候補はproxy gapである。これは探索中にleafごとにDNNを呼ばず、rootで一度決めたblendを固定する
制約を保ったまま、学習データだけを実運用分布へ近づけるためである。

## 実施結果

500万rootを1巡した候補はvalidation A/Bの両方で初期checkpointを約`0.00024`改善した。
2巡目は反転悪化し、700万rootで上記の飽和条件を満たした。詳細と次のproxy gapデータ仕様は
[大規模追試結果](../results/root-grouped-router-scale-20260824/README.md)を参照する。
