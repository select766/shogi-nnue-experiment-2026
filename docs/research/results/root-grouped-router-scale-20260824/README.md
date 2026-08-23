# Root-grouped router大規模追試 (2026-08-24)

## 結論

pilotの50倍にあたる500万root x 8 leafでrouterをtask-only学習した。500万rootを1巡した
checkpointは、初期checkpoint 510に対して、選抜用validation Bと未使用validation Aの両方で
group平均lossを約`0.00024`改善した。どちらもroot bootstrap 95%区間の上端が0未満であり、
大規模な独立データでもrootだけから決めた固定blendの改善が再現した。

一方、2巡目は大きく悪化し、学習率を半減しても回復しなかった。事前定義した
「最後の`2e-5`以上の改善から3 interval更新なし」を満たしたため、同一データ・同一目的での学習は
700万rootで飽和と判定した。最良候補は500万root時点の`checkpoints/4.ckpt`である。

## データと分離

- train: 元棋譜thread 002、500万root、各root 8 leaf
- validation: 別の元棋譜thread 010、各root 16 leafをA/B各8 leafへ分割
- trainとPacked SFENが一致する4,611 root、validation内重複14 rootを除外
- A/Bとも先頭95,232 rootを正式評価に使用
- trainの固有rootは4,810,544、重複率3.78912%
- group内の異なる棋譜offsetは平均7.0219 / 8

validation Bは学習曲線とcheckpoint選抜に使い、validation AはBで選んだ1候補へ一度だけ使った。

## 学習条件

- 初期値: `logs/expert_blending_8experts_v4_paired_uniform50_noise0/checkpoints/510.ckpt`
- expertsとDNN backboneを固定、adapterのみ学習
- batch 256、adapter LR 0.01、momentum 0.9
- 100万rootごとにvalidationとcheckpoint保存
- train iteratorはinterval境界でresetせず、5 intervalsで500万root全体を一巡
- validation悪化時はNewBobでLRを0.5倍

## 学習曲線と飽和

| 累積学習root | validation B loss | 判断 |
|---:|---:|---|
| 1,000,000 | 0.0380238444 | 最初の基準点 |
| 2,000,000 | 0.0380217955 | `2e-5`未満の改善 |
| 3,000,000 | 0.0380156673 | `2e-5`未満の改善 |
| 4,000,000 | 0.0380013026 | 基準点から`2e-5`以上改善、カウンタreset |
| 5,000,000 | **0.0379991122** | 最良、横ばい1/3 |
| 6,000,000 | 0.0393837728 | 反転悪化、LRを0.005へ半減、2/3 |
| 7,000,000 | 0.0394366421 | 回復せず、3/3で飽和 |

600万rootではtrain lossも`0.0372760`から`0.0395751`へ悪化した。validationだけの揺れではなく、
2巡目に同じ学習率で進めたことによる最適点通過と判断する。700万root後に同条件runを停止した。

## Root単位paired比較

### 選抜用validation B

- 初期checkpoint平均loss: `0.0382365361`
- 500万root候補平均loss: `0.0379991233`
- root別loss差の平均: `-0.0002374107`
- root bootstrap 95%区間: `[-0.0003174885, -0.0001605608]`
- 改善root割合: `58.201%`

### 未使用validation A

- 初期checkpoint平均loss: `0.0382287167`
- 500万root候補平均loss: `0.0379880592`
- root別loss差の平均: `-0.0002406610`
- root bootstrap 95%区間: `[-0.0003225570, -0.0001616117]`
- 改善root割合: `58.260%`

A/Bで効果量と区間がほぼ一致した。したがって、特定のleaf標本Bへの選抜過適合では説明しにくい。

## 次に検証する仮説

同じ棋譜先局面proxyを反復学習する追加runは優先しない。次の第一候補は**proxy gap**である。
棋譜上の将来局面は実探索が実際に訪れる末端分布と一致しないため、固定した探索器から
rootごとに複数のqsearch leafを収集し、同じroot固定blend制約のまま追試する。

具体的には次のデータを作る。

1. train/validationで重複しないroot SFENを別棋譜ファイルから抽出する。
2. `Threads=1`、固定nodes、固定探索器で各rootを探索し、探索中に到達したqsearch leafを記録する。
3. rootごとに訪問分布からK=8 leafを復元抽出し、root SFENとgroupを一体でshuffleする。
4. leaf教師scoreは固定した現行`nn.bin`のqsearch値とし、学習中のrouter候補から切り離す。
5. A/B各8 leafを独立に収集し、今回と同じroot paired bootstrapで判定する。

この方法は探索中にleafごとにDNNを呼ばない。DNNはrootで一度だけ実行し、選んだblendを全leafへ
固定する実行時制約を完全に維持する。次点は大損失leafを重視するgroup内CVaR、第三候補は相関約
0.97のexpert出力へ多様性を導入する学習である。

## 成果物

- `results/root_grouped_large5m_training_curve.json`
- `results/root_grouped_large5m_best_comparison_valB.json`
- `results/root_grouped_large5m_best_confirmation_valA.json`
- 実験計画: `docs/research/experiments/root-grouped-router-scale-20260823.md`
