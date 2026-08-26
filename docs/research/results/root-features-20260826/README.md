# 明示root特徴ablation 結果 (2026-08-26)

## 結論

`H-ROOT-FEATURES`は**棄却**した。game plyとSFENから安価に計算できるphase特徴をadapterへ連結すると、
ply、phase、combinedの全条件で独立valA group lossが改善し、combinedのCPU wall-clockも悪化しなかった。
しかし固定validation 10,000局面・100万nodesのbestmove一致率はmatched controlの`62.00%`から
`61.95%`へ`-0.05`ポイント低下したため、事前の正方向条件を満たさなかった。

否定した範囲はgame ply、盤上/持駒率、成駒率、合法手率、駒数差、玉間距離という静的7特徴である。
探索を一度行って得るroot統計までは同じ判定に含めない。

## 条件

- 計画: [明示root特徴ablation](../../experiments/root-features-20260826.md)
- initial: M0、追加adapter列を0初期化して初期logitを保存
- train: `tmp/proxy_gap_v2/train` 99,840 root x 8 leaf
- validation: valB/valA各19,968 root
- experts/backbone固定、adapter-only、mean group loss、1 epoch
- 条件: `ply` 1次元、`phase` 6次元、`combined` 7次元
- matched control: `tail_objective_mean_from_m0/checkpoints/0.ckpt`

学習cacheはshuffle bufferの先読みを含め116,224 rootまで作成した。最初の99,840件だけではepoch終端前に
cache境界へ達したためで、学習に数えるroot数は全条件99,840のままである。

## Group loss

差は候補minus controlの対応root差で、負が改善を表す。

| validation | 特徴 | group loss差 | bootstrap 95%区間 |
|---|---|---:|---:|
| valB | ply | -0.000000194 | [-0.000000233, -0.000000157] |
| valB | phase | -0.000001063 | [-0.000001277, -0.000000871] |
| valB | combined | **-0.000001258** | [-0.000001504, -0.000001034] |
| valA | ply | -0.000000190 | [-0.000000240, -0.000000148] |
| valA | phase | -0.000000960 | [-0.000001186, -0.000000766] |
| valA | combined | **-0.000001144** | [-0.000001413, -0.000000914] |

全条件で95%区間上端が0未満だった。combinedが両splitで最良なので配備候補とした。

## C++配備と速度

exporterを2入力/3入力ONNXの両方へ対応させ、やねうら王側で同じ7特徴をSFENから計算した。開始局面の
gateはPythonとC++で表示8桁まで一致した。従来の2入力モデルも100 nodesのsmoke testを通過している。

1000 nodesの20 go-cycle（10局面系列を2巡、先頭2 cycleはwarmup）の測定値:

| 条件 | 測定18 cycle平均 | median |
|---|---:|---:|
| control | 150.1ms | 150.2ms |
| combined | 149.0ms | 149.0ms |

差は約`-0.7%`で、追加時間10%以内という事前条件を通過した。

## 固定bestmove

| 条件 | 一致数 | 一致率 |
|---|---:|---:|
| matched control | 6,200/10,000 | 62.00% |
| combined | 6,195/10,000 | 61.95% |

対応内訳はcontrolのみ正解847、combinedのみ正解842、差`-0.0005`、正確McNemar
`p=0.922468`だった。loss改善は固定bestmoveへ転換せず、testと自己対局へは進めない。

## 成果物

- feature cache: `scripts/build_root_feature_cache.py`
- 学習/比較: `scripts/run_root_features_stage{1,2}.sh`
- export: `src/train_nnue/export_for_yaneuraou.py`
- C++推論: `YaneuraOu` submodule commit `29475544`
- loss: `results/root_features_{ply,phase,combined}_val{B,A}.json`
- accuracy: `results/accuracy_{eval,comparison}_root_features_combined_validation.json`
- logs: `/tmp/root_features_*.log`、`/tmp/accuracy_eval_root_features_combined_validation.log`
