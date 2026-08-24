# Search-aware目的関数とon-policy router検証計画 (2026-08-25)

## 仮説と不変条件

H1は、等重みのexact-qsearch leaf lossが探索の指し手目的とずれており、独立した深い探索教師へ
置き換えると指し手指標への転換が改善する、である。H2は、baseline `nn.bin`の探索木でなく配備候補
自身の固定blend探索木からleafを集めると、そのpolicy分布でrouterが改善する、である。

全条件でDNNはrootへ1回だけ実行し、その8 expert blendを探索終了まで固定する。leafごとにDNNを
再実行しない。教師は候補から独立した固定`nn.bin`評価器を使う。

## 2x2 pilot

| 条件 | leafを訪問する探索policy | 教師 |
|---|---|---|
| C00 | baseline `nn.bin` | exact qsearch |
| C10 | baseline `nn.bin` | leafから10,000 nodes探索 |
| C01 | 現候補M0 | exact qsearch |
| C11 | 現候補M0 | leafから10,000 nodes探索 |

M0は`logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt`である。全pilotはcheckpoint 510から
開始し、backboneとexpertsを固定してadapterだけを100,000 rootで1 epoch学習する。batch 256、
adapter LR 0.01、momentum 0.9、score scaling 361、task-onlyとする。

train rootは既存baseline実探索データを母集団に16個の非重複範囲から合計100,000件を使う。
on-policyで採用されたrootへbaselineデータを40-byte root recordで正確にsubsetする。validationも
同様に18,000 root x 16 leafへ揃える。これによりpolicy以外のroot集合差を除く。
学習loaderはtrain/validationのgroup size一致を要求するため、学習中のvalidationには各rootの
先頭8 leafを使い、交差評価には情報量を保った16 leaf版を使う。

## 深い教師の定義

現在のqsearch PV終端であるquiet leafを固定`nn.bin`で`go nodes 10000`し、最後のUSI `score`を
教師値とする。そのPV終端へ進めたPackedSfenValueを学習recordとし、終端の手番が教師探索開始時と
異なる場合だけscore符号を反転する。mate scoreは符号を保って学習可能範囲へclipする。

この教師はrootの最終bestmoveを直接最適化するものではないが、静的qsearchより探索結果に近いscoreへ
置き換える安価なH1 pilotである。これで指し手改善がなければ、固定gate候補を実探索しdeep teacherで
選ぶsearch-utility蒸留へ進む。

## 評価

4モデルを4 validation分布すべてで交差評価する。差は候補 minus checkpoint 510のroot group平均loss、
95% root bootstrap、改善root割合で報告する。またbaseline/on-policyの同一root leaf overlap、Jaccard、
global unique率、score分布を保存する。

pilot checkpointの選抜には固定validation 10,000局面を使う。test 10,000局面は最終候補が出た場合だけ
一度使用する。最善手一致はHCPE教師手と100万nodes `bestmove`の一致であり、対応ありMcNemar検定を使う。

## 継続・打ち切り

- H2継続: C01がC00よりon-policy validationを改善し、その差のbootstrap上端が0未満。
- H1継続: C10またはC11が対応するqsearch教師条件よりdeep validationを改善し、固定validationの
  最善手差が正方向。
- combined継続: C11がon-policy deep validationと最善手の両方で最良。
- scale: 上記を満たす最大2条件だけ500,000 rootまたは次policy DAggerへ進める。
- 打ち切り: 学習教師だけ改善してqsearch・最善手が反転する、または未知policy分布で改善が消える。

最終候補が出た場合は固定test 10,000局面と、必要効果量に応じたpaired自己対局へ進める。
