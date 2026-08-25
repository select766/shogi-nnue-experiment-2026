# Search utility探索量安定性 結果 (2026-08-25)

## 結論

`H-SU-HORIZON`は事前判定規則により**棄却**した。局所logit `+1.0`の9候補について、10,000 nodesで
得た独立utility順位は1,000,000 nodesの順位と安定しなかった。informative rootは60/128 (46.88%)、
tie補正Spearman相関のroot平均は`0.0865`、bootstrap 95%区間は
`[-0.0043, 0.1818]`で、支持閾値0.30を区間上端でも下回った。

ただし1,000,000 nodesでも候補0を10cp以上上回る有効教師は27/128 (21.09%)、Wilson 95%区間
`[14.92%, 28.95%]`だった。search utility候補そのものが消えるのではなく、短い探索で選んだ候補順位を
配備horizonへ外挿できない。したがって次の`H-SU-GEOMETRY`は、10,000 nodes教師の単純な候補拡大ではなく、
候補選抜を配備に近い探索量へ揃えて検証する。

## 条件

- 計画: [Search utility探索量安定性](../../experiments/search-utility-horizon-20260825.md)
- root: `tmp/search_utility_v1/val/roots.bin`先頭128件
- 候補: M0 root gateと各expert logitへ個別に`+1.0`した計9候補
- candidate nodes: 10,000、100,000、1,000,000
- utility: 全horizonのunique候補手を独立`nn.bin`で1,000,000 nodes制限手探索したroot手番score
- `Threads=1`、4 workers、各探索後の`isready` barrierで出力とTTを区切った
- bootstrap: root単位10,000回、seed 42

DNNは候補ごとにrootで1回だけ実行し、得たblendを各探索の終了まで固定した。leafごとのDNN実行はない。

## Horizon別の信号

| candidate nodes | 有効教師 | Wilson 95%区間 | 候補手多様度平均 | oracle gain p90 |
|---:|---:|---:|---:|---:|
| 10,000 | 37/128 (28.91%) | [21.76%,37.28%] | 2.469 | 215.6cp |
| 100,000 | 30/128 (23.44%) | [16.94%,31.48%] | 2.180 | 91.2cp |
| 1,000,000 | 27/128 (21.09%) | [14.92%,28.95%] | 1.922 | 100.0cp |

探索量を増やすと候補が返す指し手の多様度は下がったが、1mでも事前条件の5%を明確に上回る教師信号が
残った。

## Horizon間比較

| 比較 | informative root | Spearman平均 (95% CI) | oracle候補集合重複 | 同候補の指し手一致 | 教師L1平均 |
|---|---:|---:|---:|---:|---:|
| 10k vs 100k | 67/128 | 0.157 `[0.074,0.241]` | 89.84% | 62.24% | 0.0734 |
| 10k vs 1m | 60/128 | 0.087 `[-0.004,0.182]` | 96.88% | 59.29% | 0.0703 |
| 100k vs 1m | 58/128 | 0.286 `[0.179,0.391]` | 93.75% | 72.57% | 0.0538 |

oracle候補集合重複率が高い一方で順位相関が低いのは、多くの候補が同じ指し手を返して最大utilityへtie
する一方、指し手差が生じたrootで候補の細かな順位がhorizon間で入れ替わるためである。候補番号を一つに
決める一致率だけではこの構造を表せないため、事前どおりinformative rootの全順位相関を主判定にした。

## 再現性上の修正

開始時の`bin/YaneuraOu-expert-blending`が4月版で、サブモジュールに追加済みの
`ExpertBlendingGateLogitBias`を含まないことをsmokeで検出した。現サブモジュール`d83c48f4`から再ビルドし、
option文字列を含むことを確認してから本結果を生成した。古いバイナリによる失敗runは結果に含めていない。

また、C++ workerが`bestmove`直後に出すgateログを次候補が読む可能性をなくすため、候補探索後に
`isready`まで読み切るbarrierを共通collectorへ追加した。候補gateはlogit biasの解析式とも照合した。

## 成果物

- 集計: `results/search_utility_horizon_128roots.json`
- raw details: `tmp/search_utility_horizon_128roots/details.jsonl` (128行、SHA-256
  `447d0f91122df0dfd0a611bfc2d0544dc67e3e4f4abdbf3fe594e2c2af1ac594`)
- 診断: `scripts/diagnose_search_utility_horizons.py`
- 共通collector修正: `scripts/collect_search_utility_teachers.py`
- log: `/tmp/search_utility_horizon_128roots.log`
