# 24時間 深掘り検証 結果 (2026-08-28)

## 結論

`H-DECISION-ALIGNED-LOSS`を**支持**へ更新する。root候補手utilityの勝者expert軸を分類した候補は、
事前登録した既存4,000局と独立test由来8,000局の合計12,000局で5,917勝5,679敗404分、
`+6.892 Elo`、95%区間`[+0.780,+13.004]`となり、成功条件の区間下端`>0`を満たした。
効果は小さく、追加8,000局単独では`+6.689 Elo`、95%区間`[-0.789,+14.167]`と0を跨ぐため、
支持範囲は固定10万nodes、Threads=1、既存hard teacher/checkpointの合算精度に限定する。
支持判定を固定した後の未使用2,000局感度追試は`+5.386 Elo`、合計14,000局は`+6.677 Elo`、
95%区間`[+1.017,+12.336]`で、効果量と区間の正方向を維持した。

残り時間に実施した`H-ROOT-SEARCH-STATS`の重複なし1,000局面追加splitでは、静的7特徴対照
65.60%に対して浅い探索統計候補62.90%、差`-2.70`ポイント、McNemar `p=0.038878`だった。
既存の独立10,000 testも`-0.01`ポイントだったため、validationの`+0.09`ポイントは再現せず、
「固定blend予測に必要」という仮説は累積証拠により**棄却**へ更新する。

## 事前計画と分岐

- 計画: [24時間 学習量・検証深さ拡張計画](../../experiments/deep-validation-24h-20260827.md)
- 実行枠: 2026-08-27 22:50--2026-08-28 22:50 JST
- 一次結果: joint方向はheld-outで負、root統計棋力は4,000局で`-1.82 Elo`
- 分岐: 一次対象が棋力で負だったため、最も棋力に近い既存正方向信号である
  `H-DECISION-ALIGNED-LOSS`を12,000局へ拡張
- 残り6時間未満: 長い自己対局は開始せず、root統計の固定bestmove追加splitを完了
- 結果依存の重み、方向、checkpoint再選択は行っていない

## Decision-aligned 追加対局

- candidate/control: 既存4,000局と同じrelease、checkpoint、engine options
- search: 10万nodes、Threads=1
- 既存標本: `accuracy_eval_10k/validation.jsonl`由来2,000開始局面、先後反転4,000局
- 追加標本: `accuracy_eval_10k/test.jsonl`からseed 20260827で固定した4,000開始局面、
  先後反転8,000局。既存標本とは別split
- 実行: 50開始局面・100局/chunk、4 chunk並列、全80 chunk完了

| 標本 | 勝 | 敗 | 分 | score rate | Elo | 標準誤差 | 95%区間 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 既存4,000局 | 1,982 | 1,898 | 120 | 51.05% | +7.297 | 5.412 | [-3.309,+17.904] |
| 追加8,000局 | 3,935 | 3,781 | 284 | 50.9625% | +6.689 | 3.816 | [-0.789,+14.167] |
| **合計12,000局** | **5,917** | **5,679** | **404** | **50.9917%** | **+6.892** | **3.118** | **[+0.780,+13.004]** |
| 感度追試2,000局 | 985 | 954 | 61 | 50.7750% | +5.386 | 7.650 | [-9.609,+20.380] |
| **感度追試後14,000局** | **6,902** | **6,633** | **465** | **50.9607%** | **+6.677** | **2.888** | **[+1.017,+12.336]** |

追加split単独の区間は0を跨ぐが、点推定は既存splitと同方向・同程度だった。事前登録した合算では
区間下端が0を上回ったため支持とする。これは12,000局を実行した後の選択的な停止ではなく、固定した
最大局数での一回の判定である。残り時間に別途事前登録した2,000局は判定を反転しない感度分析で、
未使用test shuffle 4,000--4,999番を用いた。単独区間は広いが、点推定は既存標本と同方向だった。

## Root統計 固定bestmove追加split

- dataset: `data/accuracy_eval/test.jsonl`の全1,000局面
- 現行10,000 validation/testとのSFEN重複: いずれも0
- search: 100万nodes、Threads=1、4 workers
- baseline: 静的combined 7特徴
- candidate: 静的7特徴＋1024-node MultiPV統計6特徴

| 条件 | 一致 | 一致率 |
|---|---:|---:|
| 静的7特徴 | 656/1,000 | 65.60% |
| 浅いroot統計 | 629/1,000 | 62.90% |

対応内訳は両方正解563、対照のみ93、候補のみ66、両方不正解278で、差は`-2.70`ポイント、
exact McNemar `p=0.038878`だった。特に`abs(eval)<=500`の659局面で差`-5.159`ポイント、
`p=0.001382`だった。浅い探索の追加時間は本探索比0.1286%で速度制約内だが、予測改善は転移しない。

## 24時間枠と完了状態

主要計算は2026-08-27 22:50に開始し、追加8,000局は2026-08-28 18:17、追加固定bestmoveは
18:29、残時間の感度追試2,000局は21:44に完了した。結果文書、台帳、検査、コミットまでを
同日22:50までに完了した。
未完了またはバックグラウンドで残した実験ジョブはない。

## 成果物

- 追加対局runner: `scripts/run_decision_aligned_extension_match.sh`
- 感度追試runner: `scripts/run_decision_aligned_timebox_extension.sh`
- 合算: `results/match_decision_aligned_12000.json`
- 感度追試後合算: `results/match_decision_aligned_14000.json`
- root統計追加split:
  `results/accuracy_eval_root_features_combined_legacy_test.json`、
  `results/accuracy_eval_root_search_stats_legacy_test.json`、
  `results/accuracy_comparison_root_search_stats_legacy_test.json`
- 対局logs: `/tmp/eval_match_match_decision_aligned_extension_chunk_*.log`
- accuracy logs: `/tmp/accuracy_eval_root_{features_combined,search_stats}_legacy_test.log`
