# Tail-sensitive group objective 結果 (2026-08-26)

## 結論

`H-TAIL-OBJECTIVE`は、事前判定規則の範囲で**支持**した。root内8 leafの平均lossで学習する対照に対し、
loss上位2 leafの平均（CVaR）で学習した候補は、独立なvalAでtail lossと平均lossをともに改善した。
固定10,000局面・100万nodesの最善手一致率も`62.00%`から`62.21%`へ`+0.21`ポイントとなり、
事前規則の「正方向」を満たした。

ただし対応McNemar検定は`p=0.6325`であり、bestmove改善の大きさは統計的に確定していない。
この判定は「tail-sensitive目的が候補を作れる」という方向性への支持であり、棋力向上の証明ではない。

## 条件

- 計画: [Tail-sensitive group objective](../../experiments/tail-objective-20260826.md)
- 初期値: `logs/proxy_gap_actual500k_curve_from510/checkpoints/4.ckpt` (M0)
- train: 実探索leaf 99,840 root、各root 8 leaf
- 選抜validation: valB 19,968 root
- 未使用判定: valA 19,968 root
- experts/backbone固定、adapter-only、1 epoch
- 対照: root内8 leafの平均loss
- 候補: CVaR（loss上位2/8 leaf）およびmean/CVaR 50:50 mixed

## Loss結果

差は候補からmean対照を引いた値であり、負が改善を表す。

| validation | 候補 | 平均loss差 | tail loss差 | tail差 bootstrap 95%区間 |
|---|---|---:|---:|---:|
| valA | CVaR | -0.0000853 | **-0.0003209** | **[-0.0003769, -0.0002663]** |
| valA | mixed | -0.0000446 | -0.0001647 | [-0.0001930, -0.0001369] |
| valB | CVaR | -0.0001023 | **-0.0004011** | **[-0.0004568, -0.0003459]** |
| valB | mixed | -0.0000530 | -0.0002043 | [-0.0002327, -0.0001762] |

CVaRはvalAのtail差95%区間上端が0未満、平均lossも悪化せず、valBと同方向という3条件を満たし、
固定bestmove評価へ進めた。

## 固定10,000局面評価

`Threads=1`、100万nodes、同一10,000局面でmean対照とCVaR候補を対応比較した。

| 指標 | mean対照 | CVaR候補 | 差 |
|---|---:|---:|---:|
| 最善手一致率 | 62.00% (6,200/10,000) | 62.21% (6,221/10,000) | **+0.21pt** |
| 対照のみ正解 / 候補のみ正解 | 864 | 885 | +21局面 |

対応McNemar検定は`p=0.6325`だった。互角層では`+0.596`ポイント、優勢層では
`-1.791`ポイントであり、層別差も標本誤差を含む。この小さな正方向を棋力へ外挿せず、今後の候補選抜では
tail lossと固定bestmoveを併記する。

## 成果物

- 学習: `scripts/run_tail_objective_pilot.sh`
- 学習実装: `src/train_nnue/train_expert_blending.py`
- loss比較: `src/train_nnue/compare_root_grouped_checkpoints.py`
- loss集計: `results/tail_objective_val{A,B}.json`
- bestmove集計: `results/accuracy_comparison_tail_objective_cvar_vs_mean_validation.json`
- 評価設定: `configs/accuracy_eval_tail_objective_{mean,cvar}.json`
- logs: `/tmp/tail_objective_*.log`、`/tmp/accuracy_eval_tail_objective_*_validation.log`
